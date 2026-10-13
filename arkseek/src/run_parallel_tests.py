# -*- coding: utf-8 -*-
"""
并行执行的结构性测试（2026-09-28：推测执行 + 子问题按依赖图调度 + LLM并发上限 +
摘要挪到回复之后）。

run_orchestration_regression_tests.py 的剧本按调用顺序写，跑在串行模式下，守的是
原来的接线逻辑；这里测并行版本身：挑结果的规则对不对、依赖图调度的先后对不对、
失败时有没有停、是不是真的同时在等。用 KeyedLLMClient（按调用方+请求内容认领
回复，不看顺序）和"睡一会儿再返回"的假子问题模拟网络等待，靠时间戳断言重叠。

不需要网络、不需要API key、不加载真实模型（知识库pipeline用一个空壳），几秒跑完。
运行：cd arkseek/src && python run_parallel_tests.py
"""

import os
import threading
import time
from unittest import mock

from concurrency import set_parallel
set_parallel(True)

import llm_client
from conversation_memory import ConversationSession
from fake_llm_client import KeyedLLMClient
from memory_store import MemoryStore
from query_planner import SubAnswer


class _StubPipeline:
    """这里测的路径都不碰知识库：没有embedding模型（L2索引不建），没有精排。"""

    class dense:
        model = None
        model_name = "stub"

    class reranker:
        is_real = False


class _NoWeb:
    is_real = True
    name = "no-web"

    def search(self, query, max_results=5):
        raise AssertionError("并行测试不应该触发联网搜索")


def _session(client, window_size=6):
    s = ConversationSession(_StubPipeline(), MemoryStore(":memory:"), user_id="parallel_test",
                            llm_client=client, window_size=window_size)
    s.web_search = _NoWeb()
    return s


L1_NONE = {"updates": []}
SINGLE_PLAN = lambda q: {"sub_queries": [{"id": "q1", "query": q, "depends_on": []}]}


def _general(reply):
    return {"route": "general", "keyword": "", "reply": reply, "is_chitchat": True,
            "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": ""}


def _overlap(a, b):
    return a["start"] < b["end"] and b["start"] < a["end"]


# ============================================================
# 推测执行
# ============================================================

def test_speculative_three_judgements_overlap():
    """最常见的情况：没有要记的、不拆分、整句路由判成闲聊。三次调用同时在等，总耗时
    约等于一次（每次0.3秒：串行要0.9秒），最终用的是路由的回复。"""
    client = KeyedLLMClient([
        {"caller": "llm_l1_extract()", "response": L1_NONE, "delay": 0.3},
        {"caller": "decompose()", "response": SINGLE_PLAN("今天好冷"), "delay": 0.3},
        {"caller": "route_query()", "response": _general("是啊，多穿点"), "delay": 0.3},
    ])
    s = _session(client)
    t0 = time.monotonic()
    reply = s.user_says("今天好冷")
    dt = time.monotonic() - t0
    client.assert_exhausted()
    assert reply == "是啊，多穿点", reply
    assert client.max_active == 3, f"三次判断应该同时在等，实际同时最多{client.max_active}个"
    assert dt < 0.6, f"总耗时{dt:.2f}秒，没有重叠起来（串行约0.9秒）"


def test_memory_write_runs_alongside_answer_and_reply_waits_for_it():
    """记忆写入和回答分开（2026-09-29）：画像写入跟拆分/路由同时开始，谁也不截住谁；
    回复返回之前等画像写完（这里写入最慢，0.4秒），末尾带"（已记下：…）"。"""
    client = KeyedLLMClient([
        {"caller": "llm_l1_extract()", "response": {"updates": [{"key": "称呼", "value": "小周"}]}, "delay": 0.4},
        {"caller": "decompose()", "response": SINGLE_PLAN("我叫小周，今天好冷"), "delay": 0.2},
        {"caller": "route_query()", "response": _general("是啊，多穿点"), "delay": 0.2},
    ])
    s = _session(client)
    t0 = time.monotonic()
    reply = s.user_says("我叫小周，今天好冷")
    dt = time.monotonic() - t0
    client.assert_exhausted()
    assert reply.startswith("是啊，多穿点") and reply.endswith("（已记下：称呼：小周）"), reply
    assert s.store.get_profile("parallel_test").get("称呼") == "小周"
    write, route = client.calls_of("llm_l1_extract()")[0], client.calls_of("route_query()")[0]
    assert _overlap(write, route), "画像写入和路由应该同时在等"
    assert 0.39 <= dt < 0.6, f"总耗时{dt:.2f}秒：应该等写入完成（0.4秒），又不该把三段串起来"


def test_decompose_split_discards_speculative_route():
    """拆分判断说要拆：提前做的整句路由用不上，丢掉；子问题各自再路由。"""
    client = KeyedLLMClient([
        {"caller": "llm_l1_extract()", "response": L1_NONE},
        {"caller": "decompose()", "response": {"sub_queries": [
            {"id": "q1", "query": "甲是什么", "depends_on": []},
            {"id": "q2", "query": "乙是什么", "depends_on": []}]}},
        {"caller": "route_query()", "when": "甲是什么和乙是什么", "response": _general("整句路由-不该出现"),
         "optional": True},
        {"caller": "synthesize()", "response": "甲和乙的合并回答"},
    ])
    s = _session(client)
    fake_sub = lambda q, deps, ctx, results, detect_candidates=False: SubAnswer(display=q + "的答案",
                                                                                core=q + "的答案", ok=True)
    with mock.patch.object(s, "_execute_sub_query", side_effect=fake_sub):
        reply = s.user_says("甲是什么和乙是什么")
    time.sleep(0.1)
    client.assert_exhausted()
    assert reply == "甲和乙的合并回答", reply


# ============================================================
# 子问题按依赖图调度
# ============================================================

class _TimedSubQueries:
    """假的 _execute_sub_query：按子问题文本睡一会儿、返回预设结果，并记下开始/结束时间。"""

    def __init__(self, spec):
        self.spec = spec  # query -> (delay, SubAnswer or None)
        self.log = {}
        self._lock = threading.Lock()

    def __call__(self, query, depends_on, recent_context, dep_results, detect_candidates=False):
        start = time.monotonic()
        delay, answer = self.spec[query]
        time.sleep(delay)
        with self._lock:
            self.log[query] = {"start": start, "end": time.monotonic(), "deps_seen": sorted(dep_results)}
        return answer


def _ok(text, candidates=None):
    return SubAnswer(display=text, core=text, ok=True, candidates=candidates)


def test_dag_starts_dependent_as_soon_as_its_dependency_is_done():
    """q1、q2独立，q3依赖q1：q1和q2同时开始；q1一完成q3就开始，不等q2（LLMCompiler的
    做法，不是按层一批批等）。合并时子问题结果按plan顺序排，不按完成先后。"""
    plan = {"sub_queries": [
        {"id": "q1", "query": "A", "depends_on": []},
        {"id": "q2", "query": "B", "depends_on": []},
        {"id": "q3", "query": "C", "depends_on": ["q1"]},
    ]}
    fake = _TimedSubQueries({"A": (0.2, _ok("a")), "B": (0.6, _ok("b")), "C": (0.2, _ok("c"))})
    s = _session(KeyedLLMClient([]))
    captured = {}

    def fake_merge(original, plan_, sub_results):
        captured["order"] = list(sub_results)
        return "merged"

    with mock.patch.object(s, "_execute_sub_query", side_effect=fake), \
            mock.patch.object(s, "_merge_sub_results", side_effect=fake_merge):
        t0 = time.monotonic()
        reply = s._answer_multi_hop("原问题", plan, None)
        dt = time.monotonic() - t0
    log = fake.log
    assert reply == "merged"
    assert abs(log["A"]["start"] - log["B"]["start"]) < 0.1, "q1和q2应该同时开始"
    assert log["C"]["start"] >= log["A"]["end"] - 0.01, "q3必须等q1完成"
    assert log["C"]["start"] < log["B"]["end"], "q3不应该等跟它无关的q2"
    assert log["C"]["deps_seen"] == ["q1"] or "q1" in log["C"]["deps_seen"]
    assert captured["order"] == ["q1", "q2", "q3"], captured["order"]
    assert dt < 0.75, f"总耗时{dt:.2f}秒（按依赖图约0.6秒，全串行1.0秒）"


def test_dag_abort_when_depended_on_subquery_fails():
    """被依赖的q1失败：依赖它的q3不再启动；已经在跑的q2等它结束再返回终止提示。"""
    plan = {"sub_queries": [
        {"id": "q1", "query": "A", "depends_on": []},
        {"id": "q2", "query": "B", "depends_on": []},
        {"id": "q3", "query": "C", "depends_on": ["q1"]},
    ]}
    fake = _TimedSubQueries({"A": (0.1, SubAnswer(display="没查到", core="", ok=False)),
                             "B": (0.4, _ok("b")), "C": (0.1, _ok("c"))})
    s = _session(KeyedLLMClient([]))
    with mock.patch.object(s, "_execute_sub_query", side_effect=fake):
        t0 = time.monotonic()
        reply = s._answer_multi_hop("原问题", plan, None)
        dt = time.monotonic() - t0
    assert "得先查清楚「A」" in reply, reply
    assert "C" not in fake.log, "q1失败后不该再启动依赖它的q3"
    assert "B" in fake.log and dt >= 0.39, "已经在跑的q2要等它结束再返回"


def test_fanout_branches_run_in_parallel_and_keep_candidate_order():
    """q1查出多个候选（歧义），q2依赖q1：每个候选一个分支，同时跑；合并时按候选顺序排。"""
    plan = {"sub_queries": [
        {"id": "q1", "query": "谁", "depends_on": []},
        {"id": "q2", "query": "他们的X", "depends_on": ["q1"]},
    ]}
    spec = {"谁": (0.05, _ok("甲乙丙", candidates=["甲", "乙", "丙"])), "他们的X": (0.3, _ok("x"))}
    fake = _TimedSubQueries(spec)
    branch_log = []
    orig = fake.__call__

    def per_branch(query, depends_on, recent_context, dep_results, detect_candidates=False):
        branch_log.append(time.monotonic())
        return orig(query, depends_on, recent_context, dep_results, detect_candidates)

    s = _session(KeyedLLMClient([]))
    captured = {}

    def fake_merge(original, plan_, sub_results):
        captured["ids"] = [sq["id"] for sq in plan_["sub_queries"]]
        return "merged"

    with mock.patch.object(s, "_execute_sub_query", side_effect=per_branch), \
            mock.patch.object(s, "_merge_sub_results", side_effect=fake_merge):
        t0 = time.monotonic()
        reply = s._answer_multi_hop("原问题", plan, None)
        dt = time.monotonic() - t0
    assert reply == "merged"
    assert captured["ids"] == ["q1", "q2__甲", "q2__乙", "q2__丙"], captured["ids"]
    assert dt < 0.6, f"3个分支各0.3秒，总耗时{dt:.2f}秒，没有同时跑（串行约0.95秒）"


# ============================================================
# LLM并发上限、摘要挪到回复之后
# ============================================================

def test_llm_concurrency_cap():
    """同时打到LLM接口的请求不超过上限（默认4，套餐写的3-4个），多的在本机排队。"""
    state = {"active": 0, "max": 0}
    lock = threading.Lock()

    class _FakeCompletions:
        def create(self, **kwargs):
            with lock:
                state["active"] += 1
                state["max"] = max(state["max"], state["active"])
            time.sleep(0.15)
            with lock:
                state["active"] -= 1

            class _Delta:  # 2026-10-01起LLM请求是流式接收的，假接口也按流式返回
                content = "ok"
                model_extra = {}

            class _Chunk:
                usage = None
                choices = [type("Choice", (), {"delta": _Delta()})()]

            class _Stream:
                def __iter__(self):
                    return iter([_Chunk()])

                def close(self):
                    pass
            return _Stream()

    client = object.__new__(llm_client.OpenAIStyleLLMClient)
    client.provider_label, client.model, client.name = "fake", "fake", "fake"
    client.client = type("C", (), {"chat": type("Chat", (), {"completions": _FakeCompletions()})()})()
    threads = [threading.Thread(target=client.chat, args=([{"role": "user", "content": "x"}],),
                                kwargs={"caller": f"t{i}"}) for i in range(10)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert state["max"] == llm_client.LLM_MAX_CONCURRENCY, \
        f"同时最多{state['max']}个，上限是{llm_client.LLM_MAX_CONCURRENCY}"


def test_summary_deferred_until_after_reply():
    """defer_summary=True时，回复返回之前不调摘要；after_reply()里才调。2026-10-01起
    滑出窗口的轮次攒够SUMMARY_BATCH_TURNS条（4问4答）才合并：不够时不调摘要、
    没合并的原文留在上下文窗口里；够了才调一次。"""
    from conversation_memory import SUMMARY_BATCH_TURNS
    client = KeyedLLMClient([
        {"caller": "llm_l1_extract()", "response": L1_NONE},
        {"caller": "decompose()", "response": SINGLE_PLAN("第一句")},
        {"caller": "route_query()", "response": _general("好的")},
        {"caller": "llm_l1_extract()", "response": L1_NONE},
        {"caller": "decompose()", "response": SINGLE_PLAN("最后一句")},
        {"caller": "route_query()", "response": _general("好的")},
        {"caller": "llm_update_summary()", "response": {"summary": "之前聊了几句"}},
    ])
    s = _session(client, window_size=2)
    # 先聊到"滑出窗口的还差一轮才够一批"
    for i in range(SUMMARY_BATCH_TURNS // 2 - 1):
        s.store.append_turn(s.conversation_id, "user", f"第{i}句")
        s.store.append_turn(s.conversation_id, "assistant", f"收到{i}")
    assert s.user_says("第一句", defer_summary=True) == "好的"
    s.after_reply()
    assert not client.calls_of("llm_update_summary()"), "不够一批不该更新摘要"
    texts = [r["text"] for r in s.get_context_window()]
    assert "第0句" in texts, f"没合并的轮次应该留在上下文里：{texts}"
    # 再聊一轮，够一批了
    assert s.user_says("最后一句", defer_summary=True) == "好的"
    assert not client.calls_of("llm_update_summary()"), "回复返回之前不该更新摘要"
    s.after_reply()
    assert len(client.calls_of("llm_update_summary()")) == 1
    assert s.get_session_summary() == "之前聊了几句"
    assert len(s.get_context_window()) == 2, "合并之后上下文回到窗口大小"
    client.assert_exhausted()


TESTS = [
    test_speculative_three_judgements_overlap,
    test_memory_write_runs_alongside_answer_and_reply_waits_for_it,
    test_decompose_split_discards_speculative_route,
    test_dag_starts_dependent_as_soon_as_its_dependency_is_done,
    test_dag_abort_when_depended_on_subquery_fails,
    test_fanout_branches_run_in_parallel_and_keep_candidate_order,
    test_llm_concurrency_cap,
    test_summary_deferred_until_after_reply,
]


if __name__ == "__main__":
    results = []
    for t in TESTS:
        try:
            t()
            results.append((t.__name__, True, ""))
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            results.append((t.__name__, False, f"{type(e).__name__}: {e}"))
            print(f"[FAIL] {t.__name__}：{type(e).__name__}: {e}")
    n_ok = sum(ok for _, ok, _ in results)
    print(f"\n{n_ok}/{len(results)} 通过。")
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "parallel_test_results.md"), "w", encoding="utf-8") as f:
        f.write(f"# 并行执行结构性测试结果\n\n{n_ok}/{len(results)} 通过（run_parallel_tests.py）\n\n")
        for name, ok, err in results:
            f.write(f"- {'✅' if ok else '❌'} `{name}`{('：' + err) if err else ''}\n")
