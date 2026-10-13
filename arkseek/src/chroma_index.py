# -*- coding: utf-8 -*-
"""
Chroma向量数据库封装：替换L3（本地知识库）和L2（对话历史）里手搓的稠密检索

完整设计讨论见 README.md"接入Chroma"一节，这里只记录跟代码直接相关的要点。

【范围：只替换"稠密检索"这一路】
BM25Index、RRF融合、精排、L2时间衰减全部不动。Chroma只负责"向量存在哪、
怎么按语义相似度找最近的几条"——这是标准混合检索架构里向量库的本职工作。
embedding仍然由我们自己用BGE模型算好再交给Chroma（不用Chroma自带的默认
embedding函数），保证向量分数跟改动前是同一个模型、同一种算法（余弦相似度），
检索行为可以直接跟改动前对照。

【Chroma和SQLite/知识库文件的关系：原件 vs 索引】
- L3：原件是 data/knowledge_base/*.txt，Chroma集合"knowledge_base"是它的索引。
  每个文件记一份"指纹"（该文件切出的全部片段文本的哈希，内容或切分参数变了
  都会变），启动时逐文件比对，只重新嵌入新增/变化的文件，删掉磁盘上已经不存在
  的文件对应的记录。
- L2：原件是 memory.db 的 conversation_turns 表（SQLite），Chroma集合
  "conversation_turns"是它的索引。所有发言都写进去（用户的、助手的、闲聊的），
  带 user_id/role/is_general_chat/turn_index 元数据；查询时用 where 过滤决定
  查哪些。接入Chroma那一步刻意保持了改动前的查询范围（只查role=user、排除
  闲聊），把"换底层存储"和"改检索范围"分成两步做、分别验证；第二步已经
  放开到全部发言（闲聊降权而不是排除），见conversation_memory.py"L2检索
  范围放开"一节。助手回复存进来之前会先去掉系统自己加的格式文字。
- Chroma整个目录删掉也没关系，下次启动会从原件完整重建；反过来Chroma里的
  内容永远以原件为准。落到代码上就是 HistoryVectorIndex.reconcile()：拿SQLite
  全部记录跟Chroma比对，补写缺的、更新 is_general_chat 变了的、删掉原件里已经
  没有的——只有这一个同步入口，启动时、每轮对话结束时、每次检索前都调它，
  不在各个写入点各写各的（测试脚本会绕过对话流程直接往SQLite里塞历史，
  各写各的方案会漏掉这种写入）。

【换了embedding模型，旧向量就作废】
不同模型的向量不能混在一起算相似度，而且混了也不会报错，只会悄悄检索变差。
每个集合的元数据里记下 embed_model，启动时发现对不上就明确打印提示、删掉
整个集合重建，见 open_collection()。

【降级必须看得见】
chromadb装不上/初始化失败时退回改动前的手搓实现（行为跟以前完全一致），
但一定要让人知道：announce_degrade() 会打印醒目的提示，同时记进
DEGRADE_NOTICES，chat_cli.py 的 /status 命令和批量测试报告里的"当前向量
检索后端"一行都能看到——不能只在启动时打印一行、被后面几十行日志淹没。
"""

import hashlib
import os
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

from chunking import Chunk


# 所有降级提示汇总在这里，供 chat_cli.py 的 /status 命令随时查看。
DEGRADE_NOTICES: List[str] = []


def announce_degrade(component: str, reason: str, consequence: str) -> str:
    """打印一条醒目的降级提示并记下来。返回提示文本，方便调用方拼进 .name。"""
    notice = f"{component}：{reason}；{consequence}"
    DEGRADE_NOTICES.append(notice)
    bar = "!" * 60
    print(f"\n{bar}\n[降级] {notice}\n{bar}\n")
    return notice


def default_chroma_dir() -> str:
    """向量索引的默认位置：本机目录，不放进项目文件夹。

    一开始放在 data/chroma_db/，跟项目一起在坚果云同步目录里。真机发现同步把
    Chroma的HNSW二进制文件弄丢了（索引目录里只剩index_metadata.pickle），下次
    启动加载失败、退回内存版。活动中的数据库文件不能放在网盘/同步盘里，这是
    SQLite和Chroma官方都提醒过的；而且索引本来就是能从原件（知识库文件、
    memory.db）完整重建的派生数据，没有同步的必要——每台电脑各建各的。
    于是搬到了%LOCALAPPDATA%\\ArkSeek\\chroma_db。

    2026-09-30项目整体搬出坚果云（D:\\Desktop\\mihoyo\\memory_system，本地磁盘、不同步），
    又搬回项目里的 data/chroma_db/（.gitignore排除，不提交）：放在项目里，终端和Claude
    应用看到的是同一份（Claude桌面应用写%LOCALAPPDATA%会被系统重定向到它自己的目录）。
    环境变量 ARKSEEK_CHROMA_DIR 可以覆盖。"""
    env = os.environ.get("ARKSEEK_CHROMA_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "chroma_db")


def get_chroma_client(persist_dir: Optional[str]):
    """persist_dir 为 None 时用内存版（EphemeralClient），用于 ":memory:"
    数据库的测试会话。注意内存版在同一进程里是共享的（实测确认），所以调用方
    要给集合起互不冲突的名字，见 HistoryVectorIndex.__init__()。
    """
    import chromadb  # 延迟导入：装不上时由调用方捕获异常、走降级路径
    from chromadb.config import Settings

    # 关掉Chroma自带的匿名使用统计上报——纯本地项目，不需要往外发任何东西
    settings = Settings(anonymized_telemetry=False)
    if persist_dir is None:
        return chromadb.EphemeralClient(settings=settings)
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir, settings=settings)


def open_collection(client, name: str, embed_model: str):
    """打开（或新建）一个集合；如果集合里存的向量是另一个embedding模型算的，
    删掉重建——新旧模型的向量混在一起算相似度没有意义。"""
    existing = {c.name for c in client.list_collections()}
    if name in existing:
        col = client.get_collection(name, embedding_function=None)
        stored = (col.metadata or {}).get("embed_model")
        if stored == embed_model:
            return col
        print(f"[chroma_index] 集合「{name}」里的向量是用「{stored}」算的，当前模型是"
              f"「{embed_model}」，两者的向量不能混用，删掉整个集合按当前模型重建。")
        client.delete_collection(name)
    return client.create_collection(
        name, embedding_function=None,
        metadata={"hnsw:space": "cosine", "embed_model": embed_model},
    )


def _encode(model, texts: Sequence[str]) -> List[List[float]]:
    return model.encode(list(texts), normalize_embeddings=True).tolist()


def _cosine_from_distance(distance: float) -> float:
    # hnsw:space=cosine 时 Chroma 返回的是余弦距离 = 1 - 余弦相似度。
    # 换回相似度，跟改动前 EmbeddingDenseRetriever 的分数是同一个量。
    return 1.0 - float(distance)


# ============================================================
# L3：本地知识库
# ============================================================

class ChromaDenseRetriever:
    """L3稠密检索的Chroma版，接口跟 EmbeddingDenseRetriever 完全一致：
    .search(query, top_k) -> [(chunks列表里的位置下标, 余弦相似度)]。
    同时暴露 .model，conversation_memory.py 靠它复用同一个已加载的模型实例。
    """

    COLLECTION = "knowledge_base"

    def __init__(self, chunks: List[Chunk], model, model_name: str, persist_dir: str):
        self.chunks = chunks
        self.model = model
        self.model_name = model_name
        self.name = f"Chroma持久化向量库 + {model_name}（真实语义embedding）"
        self.client = get_chroma_client(persist_dir)
        # Chroma记录id -> 当前chunks列表里的位置下标。id用"文件名::文件内序号"
        # 而不是全局chunk_id：全局编号会因为前面插入一个新文件而整体错位，
        # 那样没变过的文件也得重新嵌入，失去增量更新的意义。
        self._id_to_pos: Dict[str, int] = {}
        try:
            self.collection = open_collection(self.client, self.COLLECTION, model_name)
            self._sync(persist_dir)
            self._probe()
        except Exception as e:
            # 索引文件损坏（比如被同步盘弄丢了一部分）：索引是派生数据，删掉这个
            # 集合按原件重建即可，不必每次启动都退回内存版。重建也失败才往上抛，
            # 由build_dense_retriever()走降级路径
            print(f"[chroma_index] 知识库向量索引读取失败（{type(e).__name__}: {e}），"
                  f"判断为索引损坏，删掉重建（首次重建需要重新计算全部向量，会比较久）。")
            try:
                self.client.delete_collection(self.COLLECTION)
            except Exception:
                pass
            self._id_to_pos = {}
            self.collection = open_collection(self.client, self.COLLECTION, model_name)
            self._sync(persist_dir)
            self._probe()

    def _probe(self) -> None:
        """真的查一次：HNSW文件缺失时get()/count()照常返回，只有query会报错。"""
        if self._id_to_pos:
            self.collection.query(query_embeddings=_encode(self.model, ["probe"]), n_results=1,
                                  include=["distances"])

    def _sync(self, persist_dir: str) -> None:
        # 按文档分组：知识库扩充后用meta里的doc_id（不同文档可能有相同的展示标签），
        # 老的Chunk没有meta，退回用source_file
        by_file: Dict[str, List[int]] = {}
        for pos, c in enumerate(self.chunks):
            by_file.setdefault(c.meta.get("doc_id", c.source_file), []).append(pos)

        stored = self.collection.get(include=["metadatas"])
        stored_fp: Dict[str, str] = {}
        for meta in stored["metadatas"]:
            stored_fp[meta["source_file"]] = meta["file_fingerprint"]

        added_files, skipped_files, removed_files = [], [], []
        for source_file, positions in by_file.items():
            texts = [self.chunks[p].text for p in positions]
            fingerprint = hashlib.sha256("\x00".join(texts).encode("utf-8")).hexdigest()
            ids = [f"{source_file}::{i}" for i in range(len(positions))]
            for rid, pos in zip(ids, positions):
                self._id_to_pos[rid] = pos
            if stored_fp.get(source_file) == fingerprint:
                skipped_files.append(source_file)
                continue
            # 新文件或者内容/切分方式变了：先删掉这个文件的旧记录（旧版本切出的
            # 片段数可能更多，只upsert会留下多余的旧片段），再整文件重新嵌入
            if source_file in stored_fp:
                self.collection.delete(where={"source_file": source_file})
            self.collection.add(
                ids=ids,
                embeddings=_encode(self.model, texts),
                documents=texts,
                metadatas=[{"source_file": source_file, "file_fingerprint": fingerprint, "local_idx": i,
                            **{k: v for k, v in self.chunks[pos].meta.items() if isinstance(v, (str, int, float, bool))}}
                           for i, pos in enumerate(positions)],
            )
            added_files.append(source_file)

        for source_file in set(stored_fp) - set(by_file):
            self.collection.delete(where={"source_file": source_file})
            removed_files.append(source_file)

        print(f"[chroma_index] 知识库向量索引（{persist_dir}）：复用{len(skipped_files)}个文件，"
              f"新嵌入{len(added_files)}个文件{added_files if added_files else ''}，"
              f"删除{len(removed_files)}个已不存在的文件{removed_files if removed_files else ''}，"
              f"共{self.collection.count()}个片段。")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        top_k = min(top_k, len(self._id_to_pos))
        if top_k <= 0:
            return []
        res = self.collection.query(
            query_embeddings=_encode(self.model, [query]), n_results=top_k,
            include=["distances"],
        )
        return [
            (self._id_to_pos[rid], _cosine_from_distance(d))
            for rid, d in zip(res["ids"][0], res["distances"][0])
            if rid in self._id_to_pos
        ]


# ============================================================
# L2：对话历史
# ============================================================

class HistoryVectorIndex:
    """L2对话历史的Chroma索引。原件是SQLite，这里只是它的向量索引，
    reconcile() 是唯一的同步入口，见模块开头说明。"""

    COLLECTION = "conversation_turns"

    def __init__(self, model, model_name: str, persist_dir: Optional[str]):
        self.model = model
        self.client = get_chroma_client(persist_dir)
        # 内存版在同一进程里共享，每个 ":memory:" 会话用独立的集合名，
        # 避免多个测试用例（常常用同一个 user_id）互相看到对方的数据。
        name = self.COLLECTION if persist_dir else f"{self.COLLECTION}_{uuid.uuid4().hex[:12]}"
        self.collection = open_collection(self.client, name, model_name)
        self.location = persist_dir or "内存（本次进程结束即消失）"
        # 每个用户在Chroma里现有的记录：turn_index -> (text, role, is_general_chat)。
        # 第一次对某个用户reconcile时从Chroma读一次，之后在内存里跟着维护，
        # 这样每次检索前的reconcile只是一次内存比对，没有额外的Chroma读取。
        self._known: Dict[str, Dict[int, Tuple[str, str, int]]] = {}

    @staticmethod
    def _rid(user_id: str, turn_index: int) -> str:
        return f"{user_id}::{turn_index}"

    def _load_known(self, user_id: str) -> Dict[int, Tuple[str, str, int]]:
        if user_id not in self._known:
            got = self.collection.get(where={"user_id": user_id}, include=["metadatas", "documents"])
            self._known[user_id] = {
                meta["turn_index"]: (doc, meta["role"], meta["is_general_chat"])
                for meta, doc in zip(got["metadatas"], got["documents"])
            }
        return self._known[user_id]

    def reconcile(self, user_id: str, rows) -> None:
        """rows：memory_store.get_all_turns() 的返回值（SQLite原件）。
        让Chroma里这个用户的记录跟原件完全一致。"""
        known = self._load_known(user_id)
        desired = {r["turn_index"]: (r["text"], r["role"], int(r["is_general_chat"])) for r in rows}

        to_add = [ti for ti, v in desired.items() if ti not in known or known[ti][0] != v[0]]
        to_retag = [ti for ti, v in desired.items()
                    if ti in known and known[ti][0] == v[0] and known[ti][1:] != v[1:]]
        to_delete = [ti for ti in known if ti not in desired]

        if to_delete:
            self.collection.delete(ids=[self._rid(user_id, ti) for ti in to_delete])
            for ti in to_delete:
                del known[ti]
        if to_add:
            texts = [desired[ti][0] for ti in to_add]
            self.collection.upsert(
                ids=[self._rid(user_id, ti) for ti in to_add],
                embeddings=_encode(self.model, texts),
                documents=texts,
                metadatas=[self._meta(user_id, ti, desired[ti]) for ti in to_add],
            )
        if to_retag:
            self.collection.update(
                ids=[self._rid(user_id, ti) for ti in to_retag],
                metadatas=[self._meta(user_id, ti, desired[ti]) for ti in to_retag],
            )
        for ti in to_add + to_retag:
            known[ti] = desired[ti]

        # 日常每轮只会新增2条（一问一答），不打印；批量补写（首次导入旧历史/
        # 上次写入失败后补上）或者删除（reset之后）才打印，方便确认同步真的发生了
        if len(to_add) > 2 or to_delete:
            print(f"[chroma_index] 对话历史向量索引同步（用户「{user_id}」，{self.location}）："
                  f"补写{len(to_add)}条，更新标记{len(to_retag)}条，删除{len(to_delete)}条。")

    @staticmethod
    def _meta(user_id: str, turn_index: int, v: Tuple[str, str, int]) -> dict:
        _, role, is_gc = v
        return {"user_id": user_id, "turn_index": turn_index, "role": role, "is_general_chat": is_gc}

    def search(self, user_id: str, query: str, allowed_turn_indices: Dict[int, int],
               top_k: int) -> List[Tuple[int, float]]:
        """allowed_turn_indices：turn_index -> 调用方history_chunks列表里的位置。
        返回 [(位置, 余弦相似度)]，形状跟改动前的 _dense_search() 一致。

        在向量检索前限定群/会话和允许的轮次，避免其他群友挤占个人问题的top_k。
        查询后仍核对SQLite认可的候选集合，防止不同步的索引产生越界证据。
        """
        if top_k <= 0 or not allowed_turn_indices:
            return []
        res = self.collection.query(
            query_embeddings=_encode(self.model, [query]),
            n_results=min(top_k, len(allowed_turn_indices)),
            where={"$and": [{"user_id": user_id}, {"turn_index": {"$in": list(allowed_turn_indices)}}]},
            include=["metadatas", "distances"],
        )
        out = []
        for meta, d in zip(res["metadatas"][0], res["distances"][0]):
            pos = allowed_turn_indices.get(meta["turn_index"])
            if pos is not None:
                out.append((pos, _cosine_from_distance(d)))
        return out[:top_k]
