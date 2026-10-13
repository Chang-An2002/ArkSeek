# -*- coding: utf-8 -*-
"""
Chroma接入的专项回归测试（接入Chroma时新增，见 chroma_index.py / README"接入Chroma"一节）

跟 run_orchestration_regression_tests.py 的分工：那边验证编排层的LLM调用接线，
这里验证"稠密检索从手写内存版换成Chroma"这件事本身：
  1. 等价性：同一个BGE模型下，Chroma版跟改动前内存版的检索结果一致（L3/L2各一个）
  2. 增量更新：知识库文件新增/修改/删除时只重新嵌入变化的部分
  3. embedding模型换了：旧集合被识别出来并重建，不会新旧向量混用
  4. L2对账：补写（含绕过对话流程直接写SQLite的历史）/更新is_general_chat标记/
     删除（reset_user之后）三种情况
  5. 隔离：两个 ":memory:" 会话即使user_id相同也互相看不到
  6. 持久化：重新打开同一个目录能直接复用，不重新嵌入
  7. 降级可见：chromadb不可用时自动退回内存版，但一定打印提示、记进
     DEGRADE_NOTICES、.name里写明"降级"
  8. 索引损坏（HNSW文件丢失，真机在坚果云同步目录里遇到过）：删掉集合按原件
     重建，不算降级

需要真实BGE模型（跟 run_orchestration_regression_tests.py 一样在本地跑），
不需要LLM、不需要联网（建议 HF_HUB_OFFLINE=1）。所有持久化目录都建在临时
目录里，不碰真实的向量索引目录（chroma_index.default_chroma_dir()）和 data/memory.db。

用法：
    cd src
    python run_chroma_tests.py
"""

import os
import shutil
import sys
import tempfile
import traceback

from chunking import Chunk, load_knowledge_base
import chroma_index
from chroma_index import ChromaDenseRetriever, HistoryVectorIndex
from dense_retriever import (
    DEFAULT_EMBED_MODEL, EmbeddingDenseRetriever, build_dense_retriever, load_embedding_model,
)
from memory_store import MemoryStore


HERE = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(HERE, "..", "data", "knowledge_base")

_MODEL = {}


def _model():
    if "m" not in _MODEL:
        _MODEL["m"] = load_embedding_model(DEFAULT_EMBED_MODEL)
    return _MODEL["m"]


def _tmpdir():
    return tempfile.mkdtemp(prefix="chroma_test_")


def _rmtree(path):
    # Windows上Chroma可能还占着文件句柄，删不掉不算测试失败
    shutil.rmtree(path, ignore_errors=True)


class _FakeDense:
    """给ConversationSession用的最小kb_pipeline替身：只需要 .dense.model /
    .dense.model_name / .reranker.is_real 这几个属性。"""

    def __init__(self, model):
        self.dense = type("D", (), {"model": model, "model_name": DEFAULT_EMBED_MODEL})()
        self.reranker = type("R", (), {"is_real": False})()


def _history_chunks(store, user_id):
    # 跟生产代码 _search_history() 用同一个函数算候选集合（L2检索范围放开后
    # 是用户+助手全部发言），否则测试自己的候选集合跟Chroma里的记录对不上
    from conversation_memory import ConversationSession
    rows = ConversationSession._history_index_rows(store.get_all_turns(user_id)[:-1])
    return [Chunk(chunk_id=r["turn_index"], source_file="", text=r["text"]) for r in rows]


# ============================================================
# 1. 等价性
# ============================================================

def test_l3_chroma_matches_in_memory():
    chunks = load_knowledge_base(KB_DIR)
    d = _tmpdir()
    try:
        old = EmbeddingDenseRetriever(chunks, model=_model())
        new = ChromaDenseRetriever(chunks, _model(), DEFAULT_EMBED_MODEL, d)
        for q in ["多种分子层面的数据怎么放到一起分析", "谱系追踪用了什么技术",
                  "肝纤维化", "单细胞测序的数据质控怎么做"]:
            a, b = old.search(q, 10), new.search(q, 10)
            assert [i for i, _ in a] == [i for i, _ in b], f"{q}: 排名不一致 {a} vs {b}"
            assert max(abs(float(x[1]) - y[1]) for x, y in zip(a, b)) < 1e-4, f"{q}: 分数不一致"
    finally:
        _rmtree(d)


def test_l2_chroma_matches_in_memory():
    from conversation_memory import ConversationSession
    store = MemoryStore(":memory:")
    session = ConversationSession(_FakeDense(_model()), store, user_id="u")
    for role, text in [
        ("user", "我们组决定用Chroma做向量数据库"), ("assistant", "好的"),
        ("user", "今天天气有点冷，中午吃了碗泡面"), ("assistant", "注意保暖"),
        ("user", "实验室的烛龙分析流程已经跑完质控了"), ("assistant", "了解"),
        ("user", "我下周要去上海开会"), ("assistant", "祝顺利"),
        ("user", "我们打算用什么做向量数据库来着？"),
    ]:
        store.append_turn("u", role, text)
    chunks = _history_chunks(store, "u")
    query = "我们打算用什么做向量数据库来着？"

    via_chroma = session._dense_search(query, chunks, top_k=len(chunks))
    index, session._history_index = session._history_index, None
    via_memory = session._dense_search(query, chunks, top_k=len(chunks))
    session._history_index = index

    assert [p for p, _ in via_chroma] == [p for p, _ in via_memory], f"{via_chroma} vs {via_memory}"
    assert max(abs(float(x[1]) - float(y[1])) for x, y in zip(via_chroma, via_memory)) < 1e-4
    # 检索范围已放开到用户+助手全部发言：4句历史用户发言+4句助手回复，
    # 当前这一轮（最后一句）不能出现在结果里
    assert len(via_chroma) == 8, via_chroma


# ============================================================
# 2/3/6. L3 增量更新 / 模型更换 / 持久化复用
# ============================================================

def test_l3_incremental_update():
    kb, d = _tmpdir(), _tmpdir()
    try:
        def write(name, text):
            with open(os.path.join(kb, name), "w", encoding="utf-8") as f:
                f.write(text)

        write("a.txt", "苹果是一种水果。\n\n香蕉也是水果。")
        write("b.txt", "Python是一种编程语言。")
        r1 = ChromaDenseRetriever(load_knowledge_base(kb), _model(), DEFAULT_EMBED_MODEL, d)
        assert r1.collection.count() == 3

        # 加一个排在最前面的新文件（全局chunk_id会整体错位），改b，删a
        write("0_new.txt", "长城在中国北方。")
        write("b.txt", "Python是一种编程语言。\n\nRust也是编程语言。")
        os.remove(os.path.join(kb, "a.txt"))
        chunks = load_knowledge_base(kb)
        ids_before = set(r1.collection.get()["ids"])
        r2 = ChromaDenseRetriever(chunks, _model(), DEFAULT_EMBED_MODEL, d)
        got = r2.collection.get(include=["metadatas"])
        assert sorted(m["source_file"] for m in got["metadatas"]) == ["0_new.txt", "b.txt", "b.txt"]
        assert "a.txt::0" in ids_before and "a.txt::0" not in got["ids"]

        # 检索返回的位置下标要对应新的chunks列表
        pos, _ = r2.search("编程语言Rust", 1)[0]
        assert "Rust" in chunks[pos].text, chunks[pos].text
        pos, _ = r2.search("中国的长城", 1)[0]
        assert "长城" in chunks[pos].text

        # 什么都没变：再打开一次不应重新嵌入（复用）
        r3 = ChromaDenseRetriever(chunks, _model(), DEFAULT_EMBED_MODEL, d)
        assert r3.collection.count() == 3
    finally:
        _rmtree(kb)
        _rmtree(d)


def test_model_change_rebuilds_collection():
    d = _tmpdir()
    try:
        client = chroma_index.get_chroma_client(d)
        col = chroma_index.open_collection(client, "some_col", "model-A")
        col.add(ids=["x"], embeddings=[[1.0, 0.0]], documents=["t"], metadatas=[{"k": 1}])
        same = chroma_index.open_collection(client, "some_col", "model-A")
        assert same.count() == 1
        rebuilt = chroma_index.open_collection(client, "some_col", "model-B")
        assert rebuilt.count() == 0 and rebuilt.metadata["embed_model"] == "model-B"
    finally:
        _rmtree(d)


# ============================================================
# 4/5/6. L2 对账 / 隔离 / 持久化
# ============================================================

def test_l2_reconcile_add_retag_delete():
    store = MemoryStore(":memory:")
    idx = HistoryVectorIndex(_model(), DEFAULT_EMBED_MODEL, None)
    store.append_turn("u", "user", "你好")
    store.append_turn("u", "assistant", "你好呀")
    store.append_turn("u", "user", "我下周要去上海开会")
    idx.reconcile("u", store.get_all_turns("u"))
    assert idx.collection.count() == 3  # 所有发言都存（含助手、含之后会被标成闲聊的）

    store.mark_turn_general_chat("u", 0)
    idx.reconcile("u", store.get_all_turns("u"))
    meta = idx.collection.get(ids=["u::0"], include=["metadatas"])["metadatas"][0]
    assert meta["is_general_chat"] == 1

    # 检索范围放开后，闲聊不再被where排除（降权在conversation_memory里做），
    # 助手发言也能查到
    hits = idx.search("u", "打招呼", {0: 0, 1: 1, 2: 2}, top_k=5)
    assert {pos for pos, _ in hits} == {0, 1, 2}, hits

    store.reset_user("u")
    store.append_turn("u", "user", "全新的第一句话")  # turn_index 0，文本变了
    idx.reconcile("u", store.get_all_turns("u"))
    got = idx.collection.get(include=["documents"])
    assert got["ids"] == ["u::0"] and got["documents"] == ["全新的第一句话"], got


def test_l2_memory_sessions_isolated():
    a = HistoryVectorIndex(_model(), DEFAULT_EMBED_MODEL, None)
    b = HistoryVectorIndex(_model(), DEFAULT_EMBED_MODEL, None)
    s = MemoryStore(":memory:")
    s.append_turn("same_user", "user", "只属于A的话")
    a.reconcile("same_user", s.get_all_turns("same_user"))
    assert a.collection.count() == 1 and b.collection.count() == 0


def test_l2_persistent_reuse():
    d = _tmpdir()
    try:
        s = MemoryStore(":memory:")
        for t in ["第一句", "第二句", "第三句"]:
            s.append_turn("u", "user", t)
        HistoryVectorIndex(_model(), DEFAULT_EMBED_MODEL, d).reconcile("u", s.get_all_turns("u"))
        again = HistoryVectorIndex(_model(), DEFAULT_EMBED_MODEL, d)
        again._model_calls = 0
        orig = chroma_index._encode

        def counting(model, texts):
            again._model_calls += 1
            return orig(model, texts)

        chroma_index._encode = counting
        try:
            again.reconcile("u", s.get_all_turns("u"))
        finally:
            chroma_index._encode = orig
        assert again._model_calls == 0, "重新打开后不应该重新嵌入已有记录"
        assert again.collection.count() == 3
    finally:
        _rmtree(d)


# ============================================================
# 7. 降级可见
# ============================================================

def test_degrade_is_visible_when_chromadb_unavailable():
    from conversation_memory import ConversationSession
    chunks = load_knowledge_base(KB_DIR)
    saved = sys.modules.get("chromadb", "absent")
    sys.modules["chromadb"] = None  # 让 import chromadb 抛 ImportError，模拟没装
    before = len(chroma_index.DEGRADE_NOTICES)
    d = _tmpdir()
    try:
        r = build_dense_retriever(chunks, chroma_dir=d)
        assert isinstance(r, EmbeddingDenseRetriever), type(r)
        assert "降级" in r.name, r.name
        # L2同样退回，并且同样留下记录
        session = ConversationSession(_FakeDense(_model()), MemoryStore(":memory:"), user_id="u")
        assert session._history_index is None
        new_notices = chroma_index.DEGRADE_NOTICES[before:]
        assert len(new_notices) == 2, new_notices
        assert any("L3" not in n and "向量检索" in n for n in new_notices)
        assert any("L2" in n for n in new_notices)
        # 退回之后的检索依然可用
        assert len(r.search("谱系追踪", 3)) == 3
    finally:
        if saved == "absent":
            del sys.modules["chromadb"]
        else:
            sys.modules["chromadb"] = saved
        del chroma_index.DEGRADE_NOTICES[before:]
        _rmtree(d)


# ============================================================
# 8. 索引损坏自动重建
# ============================================================

class _HashModel:
    """确定性的假embedding（16维，按文本哈希）：损坏测试要造1200个片段，用真实
    bge-m3算太慢，而这个测试只关心索引文件本身，跟向量是不是有语义无关。"""

    def encode(self, texts, normalize_embeddings=True):
        import hashlib
        import numpy as np
        out = []
        for t in texts:
            v = np.frombuffer(hashlib.sha256(t.encode("utf-8")).digest(), dtype=np.uint8)[:16].astype(float) + 1
            out.append(v / np.linalg.norm(v))
        return np.array(out)


def _hash_chunks():
    return [Chunk(chunk_id=i, source_file=f"f{i // 100}.txt", text=f"片段{i}") for i in range(1200)]


def test_corrupted_index_is_rebuilt_not_degraded():
    """真机遇到过：索引放在坚果云同步目录里，HNSW二进制文件（*.bin）被同步弄丢，
    只剩index_metadata.pickle，下次启动报"Error loading hnsw index"、退回内存版。
    修复后应该删掉集合按原件重建，而且不算降级。

    复现条件（实测）：Chroma要攒够1000条（hnsw:sync_threshold默认值）才把HNSW
    落盘并写pickle；条数少时没有pickle，删掉/改坏*.bin它会从日志重放、不报错，
    测不到这条路径——所以这里造1200个片段。建索引放在子进程里：进程退出后文件
    句柄才释放（Windows上占用中的文件删不掉），也跟真实情况一致。"""
    import glob
    import subprocess
    d = _tmpdir()
    try:
        code = ("import sys; sys.path.insert(0, %r)\n"
                "from run_chroma_tests import _HashModel, _hash_chunks\n"
                "from chroma_index import ChromaDenseRetriever\n"
                "r = ChromaDenseRetriever(_hash_chunks(), _HashModel(), 'hash-model', %r)\n"
                "print('TOP', [i for i, _ in r.search('片段7', 3)])\n") % (HERE, d)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
        expected = eval(out.split("TOP", 1)[1].strip())
        assert glob.glob(os.path.join(d, "*", "index_metadata.pickle")), "测试前提：HNSW应该已经落盘"
        for f in glob.glob(os.path.join(d, "*", "*.bin")):
            os.remove(f)

        before = len(chroma_index.DEGRADE_NOTICES)
        r = ChromaDenseRetriever(_hash_chunks(), _HashModel(), "hash-model", d)  # 修复前这里会抛异常
        assert len(chroma_index.DEGRADE_NOTICES) == before
        assert r.collection.count() == 1200
        assert [i for i, _ in r.search("片段7", 3)] == expected
    finally:
        _rmtree(d)


TESTS = [
    test_l3_chroma_matches_in_memory,
    test_l2_chroma_matches_in_memory,
    test_l3_incremental_update,
    test_model_change_rebuilds_collection,
    test_l2_reconcile_add_retag_delete,
    test_l2_memory_sessions_isolated,
    test_l2_persistent_reuse,
    test_degrade_is_visible_when_chromadb_unavailable,
    test_corrupted_index_is_rebuilt_not_degraded,
]


def main():
    results = []
    for t in TESTS:
        try:
            t()
            results.append((t.__name__, True, ""))
            print(f"[PASS] {t.__name__}")
        except Exception:
            results.append((t.__name__, False, traceback.format_exc()))
            print(f"[FAIL] {t.__name__}\n{traceback.format_exc()}")
    passed = sum(ok for _, ok, _ in results)
    print(f"\n{passed}/{len(results)} 通过。")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
