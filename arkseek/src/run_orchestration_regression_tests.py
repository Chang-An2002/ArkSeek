# -*- coding: utf-8 -*-
"""
编排层（2.2）结构性回归测试脚手架。

【为什么现在才有这个文件】
2026-09架构审查（跟用户一起审视"整套系统有没有变成屎山代码"那次）发现的
唯一一处真正的结构性风险：`conversation_memory.py`/`query_planner.py`/
`wiki_router.py`这个编排层，在HANDOFF.md记录的①-㉖约26轮真机修复背后，
"没有引入回归"的证据全部来自当时现写、跑完就跟着会话一起消失的沙盒测试
脚本（`test_fanout_sandbox.py`等），从来没有一份留在这个坚果云同步文件夹
里、可以随时重新跑一遍的东西。阶段一/2.1有`run_tests.py`+`test_cases.json`、
`run_multiturn_tests.py`+`multiturn_test_cases.json`这种真正带断言的回归
套件，2.2的编排层完全没有对应物——这个文件就是补这个洞。

【这层测试验证什么，不验证什么——这条边界必须说清楚，别自己骗自己】
完整设计哲学见同目录`fake_llm_client.py`模块开头的说明，这里只重复结论：
这里验证的是"代码收到LLM某个回复之后做什么"（调用顺序对不对、参数传递
对不对、分支/降级逻辑对不对），不验证"真实LLM会不会真的给出那个回复"——
decompose()/route_query()这些prompt本身的分类准不准，只能靠`run_*_demo.py`
这批脚本在你自己电脑上用真实模型验证。两层测试是互补关系：这层测试通过
只说明"这次改动没有破坏已知的接线逻辑"，不能替代真机验证；真机验证通过
也不代表以后的改动不会重新破坏接线逻辑，这正是这层测试要接住的部分。

【明确写清楚：HANDOFF里哪些问题这套脚手架覆盖不了，不要误以为漏测】
- ㉒（decompose()把上一轮没查到答案的旧问题当成新子查询混进来）——纯粹
  是LLM会不会照着prompt做的语义问题，ScriptedLLMClient的返回内容是我们
  自己写死的，测不出"LLM会不会凭空多拆一个子查询"这件事，只能靠
  run_l0_summary_context_demo.py配合真实LLM验证。
- ㉕（route_query()的history/local_kb边界不稳定）——同上，是分类prompt
  本身在具体语境下的泛化问题，不是代码逻辑问题。
- ⑬⑭⑮（synthesize()会不会拿训练记忆瞎编答案）——同上，是LLM会不会遵守
  prompt里"没查到就老实说没查到"这条指令的语义问题；这里能测的只是
  synthesize()有没有把"没有查到可信的答案"这几个字如实传给了LLM（结构
  正确），测不出LLM看到之后会不会瞎编。
以上三类问题的性质完全相同：都是"prompt写得对不对、LLM听不听话"，不是
"代码逻辑对不对"，所以都不在这套脚手架的覆盖范围内，是有意的边界，不是
遗漏。

【运行方式】
    cd arkseek/src && python run_orchestration_regression_tests.py
不需要网络、不需要真实API key（全程用本文件下面的ScriptedLLMClient），
几秒钟跑完；结果打印到终端，同时写入同目录的
`orchestration_regression_test_results.md`。

【怎么添加新的回归用例】
每次以后再修一个编排层的坑，照着下面已有的测试函数抄一份：构造场景
（`_build_session()`起个新会话，`_seed_history()`塞几条历史，或者直接调用
更底层的私有方法而不经过完整的`user_says()`）、给`ScriptedLLMClient`写清楚
"这次期望按什么顺序、被谁调用、返回什么"，最后断言結果——重点不是覆盖率
数字，是把"这次真机测试发现的这个具体坑"变成一条以后几秒钟就能重新确认
一遍的用例，加进`TESTS`列表就行。
"""

import os
from unittest import mock

# 这套用例的剧本是"按调用顺序"写的，跑在串行模式下（2026-09-28起默认并行：推测
# 执行+子问题按依赖图调度，调用顺序不固定）。串行模式逐字就是并行改动前的代码路径，
# 所以这里守的仍是原来那些接线逻辑；并行版本身的调度/挑结果逻辑由
# run_parallel_tests.py（按调用方+请求内容认领回复的假LLM）单独测。
from concurrency import set_parallel
set_parallel(False)

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession, _HISTORY_MISS_REPLY
from query_planner import SubAnswer
from fake_llm_client import ScriptedLLMClient


# 知识库只需要加载一次，多个测试用例共享同一份（真正隔离测试之间状态的是
# 每个测试各自新建的 MemoryStore(":memory:") 和 ScriptedLLMClient，不是
# 知识库这个只读的部分）。
_KB_PIPELINE_CACHE = {}


def _get_kb_pipeline():
    if "pipeline" not in _KB_PIPELINE_CACHE:
        here = os.path.dirname(os.path.abspath(__file__))
        kb_dir = os.path.join(here, "..", "data", "knowledge_base")
        _KB_PIPELINE_CACHE["pipeline"] = MemorySystemPipeline(kb_dir)
    return _KB_PIPELINE_CACHE["pipeline"]


class _FakeWebSearch:
    """联网搜索的测试替身：按预设返回结果，并记录被调用的问题。ScriptedLLMClient
    的is_real=True，会话初始化时会建出真实的MiniMax搜索客户端——_build_session()
    统一换成这个替身，保证任何测试都不会真的联网、不会消耗用户的套餐额度。"""

    is_real = True
    name = "fake-web-search"

    def __init__(self, results=None):
        self.results = results
        self.queries = []

    def search(self, query, max_results=5):
        self.queries.append(query)
        return self.results


def _build_session(script, user_id="regression_test", web_results=None):
    """新建一个用干净临时数据库+ScriptedLLMClient武装起来的会话，
    每个测试用例互相隔离（各自的sqlite ":memory:" 数据库、各自的脚本）。
    web_results：假联网搜索要返回的结果（None=搜索失败/没结果）。
    """
    store = MemoryStore(db_path=":memory:")
    client = ScriptedLLMClient(script)
    session = ConversationSession(_get_kb_pipeline(), store, user_id=user_id, llm_client=client)
    session.web_search = _FakeWebSearch(web_results)
    return session


def _seed_turns(session, turns):
    """turns：[(role, text), ...]，按顺序写进历史。"""
    for role, text in turns:
        session.store.append_turn(session.user_id, role, text)


# ============================================================
# 回归用例
# ============================================================

def test_search_history_filters_low_score_candidates():
    """回归 ㉓：`_search_history()` 排序之后要先用及格线过滤掉没达标的
    候选，不能为了凑够 top_n 条硬塞不相关的陪跑内容进"引用来源"。

    构造：历史里一条跟 query 关键词高度重合（该留下），一条完全不相关的
    闲聊（该被过滤掉，即使凑不够 top_n=2 条也不能硬塞）。这个方法本身不
    应该发起任何 LLM 调用（纯 BM25/向量检索 + 过滤逻辑）——脚本给空
    列表，如果这次改动意外让它多打了一次 LLM 调用，ScriptedLLMClient 会
    在第一次 chat() 时就报错，这本身也是有意义的回归信号。

    `_search_history()`按`reranker.is_real`二选一走`L2_RERANK_FLOOR`或
    `L2_BM25_FLOOR`其中一条门槛（见该方法实现），生产环境这两个值（0.5/
    5.0）都是拿真实多轮对话语料（几十条历史）调出来的。这条测试的语料
    只有两三条，实测过真机上两种分数（诊断数据：相关候选 bm25=1.647/
    rerank=0.477，不相关候选 bm25≈0/rerank≈0.0000137——两者差距悬殊，
    只是"相关候选"这个绝对值本身够不上生产环境5.0/0.5这两个门槛，纯粹
    是语料规模太小的缘故，不是过滤逻辑本身有问题）。这里把两条门槛都
    临时调到"相关候选能过线、不相关候选依旧不过线"的安全区间内，两条
    都patch是为了不管当前环境`reranker.is_real`是True还是False都能
    正确验证过滤逻辑，不需要复刻生产语料的规模。
    """
    import conversation_memory as cm_module

    session = _build_session(script=[])
    _seed_turns(session, [
        ("user", "我们实验室这周启动了一个新的分析流程，内部代号叫烛龙，专门处理超长读长测序数据的组装"),
        ("assistant", "好的，了解"),
        ("user", "今天天气有点冷，中午吃了碗泡面"),
        ("assistant", "哈哈，泡面确实很适合冷天"),
        ("user", "烛龙这个项目现在进展到哪一步了"),  # 当前这句话，会被 _search_history 排除在候选之外
    ])

    with mock.patch.object(cm_module, "L2_BM25_FLOOR", 0.5), \
         mock.patch.object(cm_module, "L2_RERANK_FLOOR", 0.1):
        results = session._search_history("烛龙这个分析流程现在进展到哪一步了", top_n=2)
    session.llm_client.assert_exhausted()

    texts = [p.chunk.text for p in results]
    assert any("烛龙" in t for t in texts), f"应该翻到提到烛龙的那条历史，实际结果：{texts}"
    assert not any("泡面" in t for t in texts), f"不相关的闲聊不该混进结果（㉓回归点），实际结果：{texts}"


def test_dispatch_router_general_skips_resolve_for_retrieval():
    """回归 ㉔：route="general" 时完全不该调用 resolve_for_retrieval()——
    这正是㉔真机测试暴露的"闲聊被烛龙污染"那次回归的根因（第一版实现把
    resolve_for_retrieval() 放在路由判断之前，对 general 也无差别生效）。
    脚本只给一条 route_query() 的响应；如果代码又调用了
    resolve_for_retrieval()，ScriptedLLMClient 会在下一次 chat() 时因为
    脚本耗尽而报错。

    `recent_context` 故意给非空（复刻真实㉔场景：闲聊前一句是"烛龙"这类
    更详细的陈述）——如果 `recent_context` 是空列表，`resolve_for_retrieval()`
    自己会在真正发起 LLM 调用之前就短路返回（见该函数"context_rows为空
    ...直接原样返回text"的说明），那样即使这次改动重新调用了它，这条
    测试也观察不到任何区别，等于白测；只有 `recent_context` 非空时，这条
    测试才是在真正验证"general 路由不触碰 resolve_for_retrieval()"，而不是
    凑巧靠空上下文绕过去。
    """
    script = [
        {"caller": "route_query()", "response": {
            "route": "general", "keyword": "", "reply": "哈哈，大冷天来碗热泡面确实舒服",
            "is_chitchat": True, "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
        }},
    ]
    session = _build_session(script=script)
    result = session._dispatch_with_router(
        "今天天气有点冷，中午吃了碗泡面",
        recent_context=[{"role": "user", "text": "我们实验室这周启动了一个新的分析流程，内部代号叫烛龙"}],
    )
    session.llm_client.assert_exhausted()

    assert result is not None
    reply, is_chitchat = result
    assert reply == "哈哈，大冷天来碗热泡面确实舒服"
    assert is_chitchat is True
    assert session.llm_client.caller_sequence() == ["route_query()"]


def test_dispatch_router_wiki_skips_resolve_for_retrieval():
    """回归 ㉔ 的另一半：route="wiki" 同样不该调用 resolve_for_retrieval()
    ——keyword 已经是 route_query() 自己解析好的，wiki 页面正文本身就能
    帮 LLM 确定代词指代的是谁，完整理由见 `_dispatch_with_router()` 文档
    字符串。用 monkeypatch 顶掉 `fetch_page_for_answer()`，断言它拿到的
    还是 route_query() 给出的 keyword（没有被某次意外插入的改写步骤动过）。
    """
    script = [
        {"caller": "route_query()", "response": {
            "route": "wiki", "keyword": "能天使", "reply": "",
            "is_chitchat": True, "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
        }},
        {"caller": "llm_generate_answer(wiki:能天使)", "response": "能天使精二需要的材料是技巧概要·卷2。"},
    ]
    session = _build_session(script=script)

    captured = {}

    def fake_fetch(keyword, disambiguator=None):
        captured["keyword"] = keyword
        return {"title": "能天使", "text": "能天使精二所需材料：技巧概要·卷2 x4"}

    with mock.patch("conversation_memory.fetch_page_for_answer", side_effect=fake_fetch):
        result = session._dispatch_with_router(
            "她精二的材料是什么",
            recent_context=[{"role": "user", "text": "能天使是谁"}],
        )

    session.llm_client.assert_exhausted()
    assert captured["keyword"] == "能天使"
    assert result is not None
    reply, is_chitchat = result
    assert "技巧概要" in reply
    assert is_chitchat is False
    assert session.llm_client.caller_sequence() == ["route_query()", "llm_generate_answer(wiki:能天使)"]


def test_dispatch_router_history_uses_resolved_query_for_search():
    """回归 ⑳+㉑：route="history" 时必须调用 resolve_for_retrieval()，
    并且要真正拿改写后的文本去做 L2 检索——不是"调用了但结果没被用上"
    这种半吊子修复。这里不满足于只检查调用顺序，而是构造一个只有"改写
    真的生效"才能翻到内容的场景：历史里存的是"烛龙"，当前问题是纯代词
    "它"，如果代码退化回改动前那样直接拿"它"去检索，`_search_history()`
    的BM25会因为字面零重合查不到任何东西，最终落到 `_HISTORY_MISS_REPLY`
    固定文案；只有真的用上 resolve_for_retrieval() 的改写结果去检索，
    才能翻到"烛龙"那条历史、生成一句真正的回答。
    """
    import conversation_memory as cm_module

    script = [
        {"caller": "route_query()", "response": {
            "route": "history", "keyword": "", "reply": "",
            "is_chitchat": True, "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
        }},
        {"caller": "resolve_for_retrieval()", "response": "烛龙这个分析流程现在进展到哪一步了"},
        {"caller": "llm_generate_answer(history)", "response": "目前还没有提到具体进展。"},
    ]
    session = _build_session(script=script)
    _seed_turns(session, [
        ("user", "我们实验室这周启动了一个新的分析流程，内部代号叫烛龙，专门处理超长读长测序数据的组装"),
        ("assistant", "好的，了解"),
    ])

    # 理由同 test_search_history_filters_low_score_candidates()：这条测试
    # 语料太小，生产环境的 L2_BM25_FLOOR=5.0 在这里天然达不到，临时调低
    # 只是为了让"改写后的文本确实被用于检索"这件事能被观察到，不是在
    # 验证 5.0 这个具体数值。
    with mock.patch.object(cm_module, "L2_BM25_FLOOR", 0.5):
        result = session._dispatch_with_router(
            "它现在进展到哪一步了？",
            recent_context=[{"role": "user", "text": "我们实验室这周启动了一个新的分析流程，内部代号叫烛龙，专门处理超长读长测序数据的组装"}],
        )

    session.llm_client.assert_exhausted()
    assert result is not None
    reply, is_chitchat = result
    assert reply != _HISTORY_MISS_REPLY, "退化成没查到的固定兜底文案了，说明改写后的文本没有真正被用于检索"
    assert "目前还没有提到具体进展" in reply
    assert is_chitchat is False


def test_execute_sub_query_with_dependency_always_resolves_before_routing():
    """回归 Fix A（HANDOFF ⑩-⑪）：只要子查询有 depends_on，不管最终路由
    到哪一条分支，都必须先用 resolve_for_retrieval() 把依赖信息显式替换
    进查询文本——不能像修复前那样只在 route="local_kb"/"history" 时才补，
    让 route_query() 自己在"依赖注入的信息"和"更早的多轮历史"之间竞争
    着猜，猜错了就查错页面（真实案例：三个候选查到的全是同一份"能天使"
    资料）。
    """
    dep_results = {"q1": SubAnswer(display="上一跳的完整展示文本", core="德克萨斯", ok=True)}
    script = [
        {"caller": "resolve_for_retrieval()", "response": "德克萨斯的种族是什么？"},
        {"caller": "route_query()", "response": {
            "route": "wiki", "keyword": "德克萨斯", "reply": "",
            "is_chitchat": True, "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
        }},
        {"caller": "llm_generate_answer(wiki:德克萨斯)", "response": "德克萨斯是狼族。"},
    ]
    session = _build_session(script=script)

    captured = {}

    def fake_fetch(keyword, disambiguator=None):
        captured["keyword"] = keyword
        return {"title": "德克萨斯", "text": "德克萨斯，狼族，精英干员..."}

    with mock.patch("conversation_memory.fetch_page_for_answer", side_effect=fake_fetch):
        answer = session._execute_sub_query(
            "他们的种族分别是什么？", depends_on=["q1"], recent_context=[],
            dep_results=dep_results, detect_candidates=False,
        )

    session.llm_client.assert_exhausted()
    assert captured["keyword"] == "德克萨斯", f"应该查德克萨斯，实际查了：{captured.get('keyword')}"
    assert answer is not None and answer.ok
    assert answer.core == "德克萨斯是狼族。"


def test_execute_sub_query_without_dependency_skips_resolve():
    """负向对照，跟上一条搭配着看：没有 depends_on 的独立子查询本身已经
    是完整问句，不需要（也不应该）额外调用一次 resolve_for_retrieval()
    ——多打这一次调用既没必要，也有被无关历史带偏的风险，跟上一条是
    同一个教训的两个方向。
    """
    script = [
        {"caller": "route_query()", "response": {
            "route": "wiki", "keyword": "能天使", "reply": "",
            "is_chitchat": True, "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
        }},
        {"caller": "llm_generate_answer(wiki:能天使)", "response": "能天使是医疗干员。"},
    ]
    session = _build_session(script=script)
    with mock.patch("conversation_memory.fetch_page_for_answer",
                     return_value={"title": "能天使", "text": "能天使，医疗干员..."}):
        answer = session._execute_sub_query(
            "能天使是什么职业？", depends_on=[], recent_context=[], dep_results={}, detect_candidates=False,
        )
    session.llm_client.assert_exhausted()
    assert answer is not None and answer.ok
    assert session.llm_client.caller_sequence() == ["route_query()", "llm_generate_answer(wiki:能天使)"]


def test_multi_hop_fanout_end_to_end():
    """F（多候选页面消歧，HANDOFF ⑨-⑪）端到端回归——项目里最复杂的一段
    编排逻辑，也是这套脚手架里最有价值的一条用例。用真实踩过的案例本身：
    "玛恩纳的侄女的职业是什么"，plan 拆成 q1（"玛恩纳的侄女是谁？"，
    无依赖）-> q2（"她的职业是什么？"，依赖 q1）。

    完整期望的调用序列（10次LLM调用，顺序本身就是这条测试要保护的东西）：
        1. route_query()                     —— q1，判到 wiki/玛恩纳
        2. llm_generate_answer(wiki:玛恩纳)   —— q1 生成"侄女是瑕光"这句话
        3. detect_answer_candidates()        —— 核对原文发现其实有两个候选
        （此时main循环发现q1歧义，当场跳出，不会再执行一次q2——见
        `_answer_multi_hop()` 文档字符串"发现歧义前会先白跑一次"那段
        历史教训，这条测试如果又开始白跑，会在这里因为脚本错位报错）
        4-6. 瑕光分支：resolve_for_retrieval() -> route_query() ->
             llm_generate_answer(wiki:瑕光)，且这个分支不传recent_context
             （只用branch_context里写死的候选名字，不传多轮历史）
        7-9. 临光分支：同上三步
        10. synthesize()                     —— 合并成最终回答
    """
    plan = {
        "sub_queries": [
            {"id": "q1", "query": "玛恩纳的侄女是谁？", "depends_on": []},
            {"id": "q2", "query": "她的职业是什么？", "depends_on": ["q1"]},
        ]
    }
    script = [
        {"caller": "route_query()", "response": {
            "route": "wiki", "keyword": "玛恩纳", "reply": "",
            "is_chitchat": True, "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
        }},
        {"caller": "llm_generate_answer(wiki:玛恩纳)", "response": "玛恩纳的侄女是瑕光。"},
        {"caller": "detect_answer_candidates()", "response": {"candidates": ["瑕光", "临光"]}},

        {"caller": "resolve_for_retrieval()", "response": "瑕光的职业是什么？"},
        {"caller": "route_query()", "response": {
            "route": "wiki", "keyword": "瑕光", "reply": "",
            "is_chitchat": True, "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
        }},
        {"caller": "llm_generate_answer(wiki:瑕光)", "response": "瑕光是术师干员。"},

        {"caller": "resolve_for_retrieval()", "response": "临光的职业是什么？"},
        {"caller": "route_query()", "response": {
            "route": "wiki", "keyword": "临光", "reply": "",
            "is_chitchat": True, "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "",
        }},
        {"caller": "llm_generate_answer(wiki:临光)", "response": "临光是狙击干员。"},

        {"caller": "synthesize()", "response": "玛恩纳有两个侄女：瑕光是术师干员，临光是狙击干员。"},
    ]
    session = _build_session(script=script)

    fetch_calls = []

    def fake_fetch(keyword, disambiguator=None):
        fetch_calls.append(keyword)
        pages = {
            "玛恩纳": {"title": "玛恩纳", "text": "玛恩纳有两个侄女：瑕光和临光。"},
            "瑕光": {"title": "瑕光", "text": "瑕光，术师干员..."},
            "临光": {"title": "临光", "text": "临光，狙击干员..."},
        }
        return pages[keyword]

    # 顶层这句问题之前，对话里其实聊过一件不相关的事——用来验证 fan-out
    # 分支不会把这段"更早的多轮历史"带进它自己的 resolve_for_retrieval()
    # 调用（⑪的回归点：以前正是被这类更早、更详细的历史带偏，解析出错的
    # 候选）。`format_recent_context()` 对任何背景行都会加"最近对话：\n"
    # 这个统一前缀（哪怕背景里只有一行"已查到的信息"），所以不能拿这个
    # 前缀本身当区分标志，只能直接检查这句不相关旧话的具体文字有没有
    # 泄漏进去。
    unrelated_earlier_turn = "我们之前聊过一次防御塔的强度排行"

    with mock.patch("conversation_memory.fetch_page_for_answer", side_effect=fake_fetch):
        reply = session._answer_multi_hop(
            "玛恩纳的侄女的职业是什么", plan,
            recent_context=[{"role": "user", "text": unrelated_earlier_turn}],
        )

    session.llm_client.assert_exhausted()

    assert fetch_calls == ["玛恩纳", "瑕光", "临光"], f"实际抓取顺序：{fetch_calls}"
    assert "术师" in reply and "狙击" in reply, f"合并结果里应该同时带上两个候选各自的职业，实际：{reply}"

    resolve_calls = [c for c in session.llm_client.calls if c["caller"] == "resolve_for_retrieval()"]
    assert len(resolve_calls) == 2
    for call in resolve_calls:
        user_content = call["messages"][-1]["content"]
        assert "已查到的信息" in user_content
        assert unrelated_earlier_turn not in user_content, (
            "fan-out分支不该把_answer_multi_hop()顶层的recent_context带进自己的"
            "resolve_for_retrieval()调用（⑪的回归点）"
        )


def test_disambiguate_wiki_candidates_rejects_titles_outside_candidate_set():
    """回归 ⑱ 的防御性校验：LLM 选的标题必须真的在候选集合里，不能盲目
    信任。模拟"LLM返回了一个不在候选列表里的标题"这种异常输出，确认
    `disambiguate_wiki_candidates()` 老实返回 None，不会原样把一个假标题
    透传出去。
    """
    from wiki_router import disambiguate_wiki_candidates

    client = ScriptedLLMClient(script=[
        {"caller": "disambiguate_wiki_candidates()", "response": {"title": "这个标题压根不在候选列表里"}},
    ])
    candidates = [
        {"title": "能天使", "snippet": "..."},
        {"title": "能天使的中坚信物", "snippet": "..."},
    ]
    chosen = disambiguate_wiki_candidates("能天使是什么职业", "能天使", candidates, client)
    client.assert_exhausted()
    assert chosen is None


def test_detect_answer_candidates_truncates_to_max_candidates():
    """回归 ⑩ 的 MAX_CANDIDATES 安全阀（3→6）：候选数超过上限时要截断，
    不能让 fan-out 的规模无限膨胀（这属于④"索引/聚合/动态更新"该管的
    枚举类问题，F 本身不该被撑到那个规模）。
    """
    from query_planner import detect_answer_candidates, MAX_CANDIDATES

    many = [f"候选{i}" for i in range(MAX_CANDIDATES + 3)]
    client = ScriptedLLMClient(script=[
        {"caller": "detect_answer_candidates()", "response": {"candidates": many}},
    ])
    result = detect_answer_candidates("问题", "原文", "生成的回答", client)
    client.assert_exhausted()
    assert len(result) == MAX_CANDIDATES, f"应该截断到{MAX_CANDIDATES}个，实际{len(result)}个"


def _local_kb_mocks(session, gate_results):
    """_try_answer_from_local_kb() 的结构性测试辅助：_gate() 按给定顺序
    返回True/False（不依赖真实精排分数，只验证接线），retrieve() 用wraps
    保留真实行为、同时记录每次调用的参数；reranker.is_real 固定成True
    （rewrite只在精排可用时才会触发）。"""
    kb = session.kb_pipeline
    return (
        mock.patch.object(kb, "_gate", side_effect=list(gate_results)),
        mock.patch.object(kb, "retrieve", wraps=kb.retrieve),
        mock.patch.object(kb.reranker, "is_real", True),
    )


def test_local_kb_gate_fail_triggers_rewrite_and_retrieves_with_it():
    """L3查询改写：第一次gate没过时，必须调用 rewrite_for_kb()，并且把改写
    出的问法真的传进第二次 retrieve()（而不是算出来又丢掉）；第二次过了
    gate才去生成回答。"""
    script = [
        {"caller": "rewrite_for_kb()", "response": {"rewrites": ["谱系追踪课题使用的是什么动物？"]}},
        {"caller_prefix": "llm_generate_answer(", "response": "这是生成的回答[1]"},
    ]
    session = _build_session(script=script)
    p_gate, p_retrieve, p_real = _local_kb_mocks(session, [False, True])
    with p_gate as gate_mock, p_retrieve as retrieve_mock, p_real:
        result = session._try_answer_from_local_kb("lineage tracing课题使用的是什么动物？")
    session.llm_client.assert_exhausted()

    # 靠改写才过线时必须用更高的门槛（见 pipeline.REWRITE_RERANK_FLOOR）
    kb = session.kb_pipeline
    assert gate_mock.call_args_list[1].kwargs.get("rerank_floor") == kb.REWRITE_RERANK_FLOOR,         gate_mock.call_args_list

    calls = retrieve_mock.call_args_list
    assert len(calls) == 2, f"应该检索两次（原问题一次、带改写一次），实际{len(calls)}次"
    assert not calls[0].kwargs.get("rewrites"), "第一次检索不该带改写"
    assert calls[1].kwargs.get("rewrites") == ["谱系追踪课题使用的是什么动物？"], calls[1]
    assert result.ok is True


def test_local_kb_gate_pass_skips_rewrite():
    """gate第一次就过：不该调用 rewrite_for_kb()（多花一次LLM调用），脚本里
    只有生成那一次。"""
    script = [{"caller_prefix": "llm_generate_answer(", "response": "这是生成的回答[1]"}]
    session = _build_session(script=script)
    p_gate, p_retrieve, p_real = _local_kb_mocks(session, [True])
    with p_gate, p_retrieve as retrieve_mock, p_real:
        result = session._try_answer_from_local_kb("多组学数据整合有哪些方法？")
    session.llm_client.assert_exhausted()
    assert retrieve_mock.call_count == 1
    assert result.ok is True


def test_local_kb_no_rewrite_keeps_rejecting_without_generation():
    """rewrite_for_kb() 判断不需要改写（返回空列表）：不重新检索，维持拒答，
    也绝不能调用生成——gate没过时不生成是防幻觉的硬性防线。"""
    script = [{"caller": "rewrite_for_kb()", "response": {"rewrites": []}}]
    session = _build_session(script=script)
    p_gate, p_retrieve, p_real = _local_kb_mocks(session, [False])
    with p_gate, p_retrieve as retrieve_mock, p_real:
        result = session._try_answer_from_local_kb("今天晚饭吃什么？")
    session.llm_client.assert_exhausted()
    assert retrieve_mock.call_count == 1
    assert result.ok is False


def test_pipeline_retrieve_actually_uses_rewrites():
    """上面三条只验证接线（改写有没有被传进 retrieve()），这条验证 retrieve()
    内部真的用上了改写问法：传入改写后，至少有一条返回的片段是靠改写问法
    拿到最高精排分的（matched_query 是改写问法而不是原问题）。自我验证时
    发现只测接线守不住"传进去了但在 retrieve() 里被丢掉"这种回归，所以补上。

    依赖真实精排模型（NoOp占位版没有精排分数，matched_query 永远是原问题），
    跟这个文件其它用例一样需要在本地真实模型环境跑。
    """
    kb = _get_kb_pipeline()
    assert kb.reranker.is_real, "这条用例需要真实精排模型"
    q = "lineage tracing课题使用的是什么动物？"
    rewrite = "谱系追踪课题使用的是什么动物？"
    passages = kb.retrieve(q, top_n=3, rewrites=[rewrite])
    assert any(p.matched_query == rewrite for p in passages),         [(p.matched_query, round(p.rerank_score, 3)) for p in passages]


def test_clean_assistant_text():
    """L2检索范围放开：助手回复进检索语料之前要去掉系统自己加的格式文字，
    但不能误删wiki正文里本身带【】的有效内容；纯拒答/没翻到这类固定文案
    清洗后为空、不进语料。"""
    from conversation_memory import clean_assistant_text, _HISTORY_MISS_REPLY

    raw = ("【L3知识库——明日方舟wiki《德克萨斯》，LLM生成回答】\n"
           "德克萨斯的种族是鲁珀[1]，档案里标注了【种族】鲁珀[1]。\n\n"
           "引用来源：\n[1] prts.wiki/德克萨斯")
    cleaned = clean_assistant_text(raw)
    assert cleaned == "德克萨斯的种族是鲁珀，档案里标注了【种族】鲁珀。", repr(cleaned)

    multi_hop = ("（这个问题拆成了2部分分别查证）\n- 玛恩纳的侄女是谁？\n"
                 "  【L3知识库——明日方舟wiki《玛恩纳》，LLM生成回答】\n瑕光和临光[1]。\n\n引用来源：\n[1] prts.wiki/玛恩纳")
    cleaned = clean_assistant_text(multi_hop)
    assert "引用来源" not in cleaned and "拆成了" not in cleaned and "【L3" not in cleaned
    assert "瑕光和临光" in cleaned, repr(cleaned)

    assert clean_assistant_text("【未找到可靠信息】没有检索到置信度足够的相关内容，不做猜测性回答。") == ""
    assert clean_assistant_text(_HISTORY_MISS_REPLY) == ""
    assert clean_assistant_text("好的，用Chroma做向量数据库这个决定我记下了") == "好的，用Chroma做向量数据库这个决定我记下了"


def test_search_history_retrieves_assistant_reply_with_assistant_label():
    """L2检索范围放开的动机场景："今天吃什么"（请求建议，被标成闲聊）->
    助手推荐 -> 之后问"你上次推荐我吃什么来着"。放开之前两边都被过滤掉、
    必然翻不到；放开之后助手那条推荐要能翻到，而且来源标签必须写明是
    "我（助手）的回答"——生成步骤靠这个标签把它表述成"我之前推荐过"，
    而不是"你说过"。不应该发起任何LLM调用（纯检索）。"""
    import conversation_memory as cm_module

    session = _build_session(script=[])
    _seed_turns(session, [
        ("user", "今天晚饭吃什么？"),
        ("assistant", "可以试试番茄炒蛋配青菜豆腐汤，简单又下饭；想吃肉的话红烧鸡翅也不错"),
        ("user", "我们实验室这周启动了一个新的分析流程，内部代号叫烛龙"),
        ("assistant", "好的，了解"),
        ("user", "你上次推荐我晚饭吃什么来着？"),  # 当前这句
    ])
    session.store.mark_turn_general_chat(session.user_id, 0)

    with mock.patch.object(cm_module, "L2_BM25_FLOOR", 0.5), \
         mock.patch.object(cm_module, "L2_RERANK_FLOOR", 0.1):
        results = session._search_history("你上次推荐我晚饭吃什么来着？", top_n=2)
    session.llm_client.assert_exhausted()

    hit = [p for p in results if "番茄炒蛋" in p.chunk.text]
    assert hit, f"助手那条推荐应该能翻到，实际：{[(p.chunk.source_file, p.chunk.text) for p in results]}"
    assert "我（助手）的回答" in hit[0].chunk.source_file, hit[0].chunk.source_file


def test_search_history_downweights_general_chat_instead_of_excluding():
    """闲聊降权而不是排除：同一句话，标成闲聊之后分数应该正好乘以
    GENERAL_CHAT_WEIGHT，而不是从结果里消失。直接比较同一条候选在标记前后
    的分数，不依赖任何绝对阈值。"""
    import conversation_memory as cm_module

    def score_of_turn0(mark_general):
        session = _build_session(script=[])
        _seed_turns(session, [
            ("user", "我们组决定用Chroma做向量数据库"),
            ("assistant", "好的，记下了"),
            ("user", "我们组打算用什么做向量数据库来着？"),
        ])
        if mark_general:
            session.store.mark_turn_general_chat(session.user_id, 0)
        with mock.patch.object(cm_module, "L2_BM25_FLOOR", 0.0), \
             mock.patch.object(cm_module, "L2_RERANK_FLOOR", 0.0):
            results = session._search_history("我们组打算用什么做向量数据库来着？", top_n=5)
        p = next(p for p in results if p.chunk.chunk_id == 0)
        return p.rerank_score if p.rerank_score is not None else p.bm25_score

    normal, downweighted = score_of_turn0(False), score_of_turn0(True)
    assert abs(downweighted - normal * cm_module.GENERAL_CHAT_WEIGHT) < 1e-6, (normal, downweighted)


def test_search_history_gate_uses_pre_decay_relevance():
    """相关性和时间衰减分开用：gate看衰减前的原始相关性，衰减只影响排序。
    把DECAY_RATE临时调到0.9（隔两句就只剩1%权重），一条真正相关的旧发言
    衰减后的分数远低于及格线，但原始相关性过线，应该照样被翻出来——回归
    "你上次推荐我吃什么来着"那次翻不到的问题（原始约0.69，衰减到0.425被
    0.5的门槛挡掉）。用生产环境的及格线，不patch门槛。"""
    import conversation_memory as cm_module

    session = _build_session(script=[])
    _seed_turns(session, [
        ("user", "我们组决定用Chroma做向量数据库"),
        ("assistant", "好的，这个我记下了"),
        ("user", "中午好"),
        ("assistant", "中午好呀"),
        ("user", "今天天气不错"),
        ("assistant", "是呀"),
        ("user", "我们组打算用什么做向量数据库来着？"),
    ])
    with mock.patch.object(cm_module, "DECAY_RATE", 0.9):
        results = session._search_history("我们组打算用什么做向量数据库来着？", top_n=2)
    session.llm_client.assert_exhausted()

    hit = [p for p in results if "Chroma" in p.chunk.text and p.chunk.chunk_id == 0]
    assert hit, [(p.chunk.text, p.relevance_score, p.rerank_score) for p in results]
    p = hit[0]
    floor = cm_module.L2_RERANK_FLOOR if session.kb_pipeline.reranker.is_real else cm_module.L2_BM25_FLOOR
    decayed = p.rerank_score if p.rerank_score is not None else p.bm25_score
    assert decayed < floor <= p.relevance_score, (decayed, floor, p.relevance_score)


def test_auto_intro_only_on_first_contact_and_not_stored():
    """开场白（bot_profile.py）：auto_intro=True 时只在这个用户第一次对话时
    加在回复前面，第二句不再加；而且开场白不能存进对话历史（否则会被L2
    翻旧账检索到、挤占L0窗口）。auto_intro 默认关闭，批量测试不受影响。"""
    from bot_profile import BOT_NAME
    route_general = lambda reply: {"caller": "route_query()", "response": {
        "route": "general", "keyword": "", "reply": reply, "is_chitchat": True,
        "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": ""}}
    script = [
        {"caller": "llm_l1_extract()", "response": {"updates": []}},
        {"caller": "decompose()", "response": {"sub_queries": [{"id": "q1", "query": "你好", "depends_on": []}]}},
        route_general("你好呀～"),
        {"caller": "llm_l1_extract()", "response": {"updates": []}},
        {"caller": "decompose()", "response": {"sub_queries": [{"id": "q1", "query": "在吗", "depends_on": []}]}},
        route_general("在的～"),
    ]
    store = MemoryStore(db_path=":memory:")
    client = ScriptedLLMClient(script)
    session = ConversationSession(_get_kb_pipeline(), store, user_id="intro_test",
                                  llm_client=client, auto_intro=True)
    first = session.user_says("你好")
    second = session.user_says("在吗")
    client.assert_exhausted()

    assert BOT_NAME in first and first.endswith("你好呀～"), first
    assert BOT_NAME not in second and second == "在的～", second
    stored = [r["text"] for r in store.get_all_turns("intro_test")]
    assert not any(BOT_NAME in t for t in stored), stored


# ============================================================
# QQ群压力测试准备（二）：查不到时 联网搜索 -> 大模型自身知识，分级标注
# ============================================================

_WEB_RESULTS = [{"title": "小狼XF", "link": "https://example.com/a", "snippet": "明日方舟攻略UP主", "date": ""}]


def _route(route, keyword="", reply="", needs_search=False, is_chitchat=True):
    return {"caller": "route_query()", "response": {
        "route": route, "keyword": keyword, "reply": reply, "is_chitchat": is_chitchat,
        "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "", "needs_search": needs_search}}


def test_general_needs_search_goes_to_web_with_label():
    """客观问题（general+needs_search）：联网搜索->按搜索结果生成->带参考链接和
    用户定的标注文字，逐字；不算闲聊。"""
    from llm_generate import WEB_SEARCH_LABEL
    script = [_route("general", reply="我帮你查查", needs_search=True),
              {"caller": "llm_answer_from_web()", "response": "可以看看小狼XF[1]。"}]
    session = _build_session(script, web_results=_WEB_RESULTS)
    reply, is_chitchat = session._dispatch_with_router("推荐个明日方舟攻略up主")
    session.llm_client.assert_exhausted()
    assert session.web_search.queries == ["推荐个明日方舟攻略up主"]
    assert "小狼XF" in reply and "https://example.com/a" in reply
    assert reply.endswith(WEB_SEARCH_LABEL), reply
    assert is_chitchat is False


def test_general_search_uses_router_search_query():
    """2026-10-04真机：聊着KFC问"现在有什么新品吗"，原句里没有"KFC"，拿原句搜结果对不上。
    路由给了search_query时拿它去搜（整理结果用的还是原句）；没给（旧的路由输出）时照旧用原句。"""
    r = _route("general", reply="我帮你查查", needs_search=True)
    script = [{**r, "response": {**r["response"], "search_query": "肯德基 2026年10月 新品"}},
              {"caller": "llm_answer_from_web()", "response": "有新品X[1]。"}]
    session = _build_session(script, web_results=_WEB_RESULTS)
    session._dispatch_with_router("现在有什么正在卖的新品吗？")
    session.llm_client.assert_exhausted()
    assert session.web_search.queries == ["肯德基 2026年10月 新品"], session.web_search.queries


def test_general_subjective_does_not_search():
    """主观闲聊（needs_search=False）：直接用路由给的回复，不搜索、不加标注。"""
    session = _build_session([_route("general", reply="我喜欢能天使～", needs_search=False)],
                             web_results=_WEB_RESULTS)
    reply, is_chitchat = session._dispatch_with_router("你觉得哪个干员最帅")
    session.llm_client.assert_exhausted()
    assert session.web_search.queries == []
    assert reply == "我喜欢能天使～" and is_chitchat is True


def test_search_failure_falls_back_to_own_knowledge_with_label():
    """搜索失败/没结果：退到大模型自身知识，带另一句标注；搜索结果答不了
    （NO_ANSWER）同样退到这一级。"""
    from llm_generate import LLM_KNOWLEDGE_LABEL
    script = [_route("general", needs_search=True, reply="x"),
              {"caller": "llm_answer_from_own_knowledge()", "response": "一般认为是某某。"}]
    session = _build_session(script, web_results=None)
    reply, _ = session._dispatch_with_router("明日方舟哪个干员最强")
    session.llm_client.assert_exhausted()
    assert reply.endswith(LLM_KNOWLEDGE_LABEL) and "一般认为是某某" in reply

    script = [_route("general", needs_search=True, reply="x"),
              {"caller": "llm_answer_from_web()", "response": "NO_ANSWER"},
              {"caller": "llm_answer_from_own_knowledge()", "response": "一般认为是某某。"}]
    session = _build_session(script, web_results=_WEB_RESULTS)
    reply, _ = session._dispatch_with_router("明日方舟哪个干员最强")
    session.llm_client.assert_exhausted()
    assert reply.endswith(LLM_KNOWLEDGE_LABEL) and "NO_ANSWER" not in reply


def test_wiki_miss_goes_to_web_not_local_kb():
    """wiki查不到页面：先联网搜索兜底，不再退回作者的生信课题资料（明日方舟的
    问题在那里不可能有答案）。"""
    from llm_generate import WEB_SEARCH_LABEL
    script = [_route("wiki", keyword="最强干员"),
              {"caller": "llm_answer_from_web()", "response": "社区常提到某某[1]。"}]
    session = _build_session(script, web_results=_WEB_RESULTS)
    with mock.patch("conversation_memory.fetch_page_for_answer", return_value=None):
        reply, _ = session._dispatch_with_router("明日方舟最强干员是谁")
    session.llm_client.assert_exhausted()
    assert reply.endswith(WEB_SEARCH_LABEL)
    assert "rewrite_for_kb()" not in session.llm_client.caller_sequence()
    # 原句不带游戏名时，搜索词要补上"明日方舟"（真机实测不补会搜到别的游戏）
    session2 = _build_session([_route("wiki", keyword="第五章"),
                               {"caller": "llm_answer_from_web()", "response": "是某某[1]。"}],
                              web_results=_WEB_RESULTS)
    with mock.patch("conversation_memory.fetch_page_for_answer", return_value=None):
        session2._dispatch_with_router("主线第五章的最终boss是谁")
    assert session2.web_search.queries == ["明日方舟 主线第五章的最终boss是谁"], session2.web_search.queries
    assert "问的是游戏《明日方舟》" in session2.llm_client.calls[-1]["messages"][-1]["content"]


def test_local_kb_no_answer_falls_back_with_kb_miss_note():
    """本地知识库gate过了、但生成步骤判断完全答不了（NO_ANSWER，C15那类）：
    不再展示"L3知识库"的回答，转去联网搜索，并在前面说明"作者的课题资料里
    没有找到这部分内容"。"""
    from llm_generate import WEB_SEARCH_LABEL
    kb = _get_kb_pipeline()
    script = [_route("local_kb"),
              {"caller_prefix": "llm_generate_answer(", "response": "NO_ANSWER"},
              {"caller": "llm_answer_from_web()", "response": "空间转录组分辨率一般更低[1]。"}]
    session = _build_session(script, web_results=_WEB_RESULTS)
    with mock.patch.object(kb, "retrieve_with_gate", return_value=(kb.retrieve("单细胞", top_n=3), True)):
        reply, _ = session._dispatch_with_router("空间转录组和单细胞测序哪个分辨率更高")
    session.llm_client.assert_exhausted()
    assert reply.startswith("（长安的资料里没有找到这部分内容") and reply.endswith(WEB_SEARCH_LABEL), reply
    assert "【L3知识库" not in reply


def test_history_miss_never_searches_web():
    """L2翻旧账查不到：维持固定的"没翻到"回复，不联网搜索（网上不可能有用户
    之前说过什么）。"""
    import conversation_memory as cm_module
    session = _build_session([_route("history")], web_results=_WEB_RESULTS)
    reply, _ = session._dispatch_with_router("我上次说的那件事来着？")
    session.llm_client.assert_exhausted()
    assert reply == cm_module._HISTORY_MISS_REPLY
    assert session.web_search.queries == []


def test_merge_labels_each_unverified_part_separately():
    """复合问题分段标注：知识来源的部分交给synthesize()合并（只给它这几部分的
    问题），联网搜索/大模型自身知识的部分各自单独成段、原样保留自己的标注。"""
    from query_planner import SubAnswer
    from llm_generate import WEB_SEARCH_LABEL, LLM_KNOWLEDGE_LABEL
    plan = {"sub_queries": [
        {"id": "q1", "query": "能天使是什么职业？", "depends_on": []},
        {"id": "q2", "query": "推荐个攻略up主", "depends_on": []},
        {"id": "q3", "query": "哪个干员最强", "depends_on": []},
    ]}
    results = {
        "q1": SubAnswer(display="狙击干员", core="能天使是狙击干员", ok=True, source="wiki",
                        refs=["prts.wiki/能天使"]),
        "q2": SubAnswer(display=f"小狼XF\n\n{WEB_SEARCH_LABEL}", core="小狼XF", ok=True, source="web"),
        "q3": SubAnswer(display=f"一般认为某某\n\n{LLM_KNOWLEDGE_LABEL}", core="某某", ok=True, source="llm"),
    }
    session = _build_session([{"caller": "synthesize()", "response": "能天使是狙击干员。"}])
    merged = session._merge_sub_results("能天使什么职业，再推荐个up主，顺便说说哪个干员最强", plan, results)
    session.llm_client.assert_exhausted()

    synth_input = session.llm_client.calls[0]["messages"][-1]["content"]
    assert "推荐个攻略up主" not in synth_input and "哪个干员最强" not in synth_input, synth_input
    assert merged.startswith("能天使是狙击干员。")
    # 知识来源那一段也要有自己的来源标注，排在网络搜索那段之前（用户的要求）
    wiki_label = "【以上内容来自明日方舟wiki：prts.wiki/能天使】"
    assert wiki_label in merged, merged
    assert merged.index(wiki_label) < merged.index("关于「推荐个攻略up主」")
    assert "关于「推荐个攻略up主」：\n小狼XF" in merged and "关于「哪个干员最强」：\n一般认为某某" in merged
    assert merged.index(WEB_SEARCH_LABEL) < merged.index("关于「哪个干员最强」")
    assert merged.endswith(LLM_KNOWLEDGE_LABEL)


def test_merge_all_verified_keeps_citations():
    """复合问题全部来自知识来源：synthesize()合并后要补回引用列表（去重），
    但不加【以上内容来自…】（那是多来源回答才用的，用户的要求）。"""
    plan = {"sub_queries": [
        {"id": "q1", "query": "作者做过几次JC？", "depends_on": []},
        {"id": "q2", "query": "都讲了哪些文章？", "depends_on": []},
    ]}
    results = {
        "q1": SubAnswer(display="8次", core="8次", ok=True, source="kb", refs=["长安的Zotero文献目录"]),
        "q2": SubAnswer(display="ArchR等", core="ArchR等", ok=True, source="kb",
                        refs=["长安的Zotero文献目录", "长安2025年1月4日 JC讲稿"]),
    }
    session = _build_session([{"caller": "synthesize()", "response": "作者做过8次JC，讲过ArchR等。"}])
    merged = session._merge_sub_results("作者做过几次JC？都讲了哪些文章？", plan, results)
    session.llm_client.assert_exhausted()
    assert merged.startswith("作者做过8次JC，讲过ArchR等。"), merged
    assert "引用来源：\n[1] 长安的Zotero文献目录\n[2] 长安2025年1月4日 JC讲稿" in merged, merged
    assert "【以上内容来自" not in merged


def test_catalog_question_answered_from_catalog_without_generation():
    """知识库扩充：列清单类问题（kb_query_kind="catalog"）直接查文献目录、确定性
    拼接——不调用检索生成，文献一篇不漏，讲过/没讲过分组，JC日期写对。"""
    fake_catalog = {"papers": [
        {"key": "A", "title": "Paper Alpha", "first_author": "Li", "year": "2023", "journal": "Cell",
         "summary": "第一句简介。第二句。", "jc": ["2025-07-05"]},
        {"key": "B", "title": "Paper Beta", "first_author": "Wang", "year": "2022", "journal": "NBT",
         "summary": "", "jc": []},
    ], "jc_sessions": []}
    session = _build_session([_route("local_kb") | {"response": {**_route("local_kb")["response"],
                                                                  "kb_query_kind": "catalog"}}])
    with mock.patch.object(session.kb_pipeline, "catalog", fake_catalog, create=True):
        reply, _ = session._dispatch_with_router("作者都读过哪些文献？")
    session.llm_client.assert_exhausted()
    assert session.llm_client.caller_sequence() == ["route_query()"]
    assert "共2篇" in reply and "Paper Alpha" in reply and "Paper Beta" in reply, reply
    assert "2025年7月5日讲" in reply and "第一句简介。" in reply and "第二句" not in reply
    assert reply.index("Paper Alpha") < reply.index("没在JC上讲过") < reply.index("Paper Beta")


def test_catalog_scope_lists_only_what_was_asked():
    """用户看测试题时指出：问"做过几次JC、都讲了哪些"也把没讲过的文献列了出来。
    路由给catalog_scope之后，"jc"只列讲过的（带场次数），"not_jc"只列没讲过的；
    路由给的范围要真的传到目录那一步。"""
    fake_catalog = {"papers": [
        {"key": "A", "title": "Paper Alpha", "first_author": "Li", "year": "2023", "journal": "Cell",
         "summary": "简介A。", "jc": ["2025-07-05"]},
        {"key": "B", "title": "Paper Beta", "first_author": "Wang", "year": "2022", "journal": "NBT",
         "summary": "", "jc": []},
    ], "jc_sessions": [{"date": "2025-07-05"}]}
    r = _route("local_kb")
    session = _build_session([{**r, "response": {**r["response"], "kb_query_kind": "catalog",
                                                 "catalog_scope": "jc"}}])
    with mock.patch.object(session.kb_pipeline, "catalog", fake_catalog, create=True):
        reply, _ = session._dispatch_with_router("长安做过几次JC？都讲了哪些文章？")
        not_jc = session._try_answer_from_catalog("not_jc").display
    session.llm_client.assert_exhausted()
    assert "一共做过1次" in reply and "Paper Alpha" in reply and "Paper Beta" not in reply, reply
    assert "Paper Beta" in not_jc and "Paper Alpha" not in not_jc, not_jc


def test_project_scope_passes_doc_filter_to_local_kb():
    """路由判断问的是作者自己的课题（kb_scope="project"）：本地知识库检索要带上
    "只查课题报告"的过滤条件；没给这个字段（旧的路由输出）时照旧全库检索。"""
    seen = []

    def fake_local_kb(query, extra_context=None, scope="all", jc_session="", retrieval_query=None):
        seen.append(scope)
        return SubAnswer(display="答案", core="答案", ok=True, source="kb")

    for route_extra, expected in [({"kb_scope": "project"}, "project"), ({}, "all")]:
        r = _route("local_kb")
        session = _build_session([{**r, "response": {**r["response"], **route_extra}}])
        with mock.patch.object(session, "_try_answer_from_local_kb", side_effect=fake_local_kb):
            session._dispatch_with_router("这个课题的scRNA-seq数据用什么软件做alignment？")
        assert seen[-1] == expected, (route_extra, seen)


def test_jc_session_resolved_from_catalog_by_date():
    """2026-10-02真机："最近一次文献分享会讲的文献"答成了6月那次（实际9月12日）。路由只给
    jc_session（"latest"/"earliest"/日期前缀），哪一次由代码按目录日期算：最近=今天及以前
    日期最大的那次（还没到的不算）；对不上就返回None照常全库检索。路由给的值要传到检索那一步，
    检索只在那篇文献的片段里查，问题后面写上日期和篇名。"""
    fake_catalog = {"papers": [{"key": "A", "title": "Paper Alpha"}, {"key": "B", "title": "Paper Beta"},
                               {"key": "C", "title": "Paper Future"}],
                    "jc_sessions": [{"date": "2025-07-05", "match": {"key": "A"}},
                                    {"date": "2026-09-12", "match": {"key": "B"}},
                                    {"date": "2099-01-01", "match": {"key": "C"}},
                                    {"date": "2026-06-13"}]}  # 没对上文献的那次不参与
    r = _route("local_kb")
    session = _build_session([{**r, "response": {**r["response"], "jc_session": "latest"}}])
    with mock.patch.object(session.kb_pipeline, "catalog", fake_catalog, create=True):
        assert session._jc_session_filter("latest") == (["B"], "2026-09-12 JC讲的文献：Paper Beta")
        assert session._jc_session_filter("earliest") == (["A"], "2025-07-05 JC讲的文献：Paper Alpha")
        assert session._jc_session_filter("2025-07")[0] == ["A"]
        assert session._jc_session_filter("2026-03") is None and session._jc_session_filter("") is None
        seen = {}

        def fake_gate(query, query_rewriter=None, doc_types=None, paper_keys=None):
            seen.update(query=query, paper_keys=paper_keys)
            return [], False
        with mock.patch.object(session.kb_pipeline, "retrieve_with_gate", side_effect=fake_gate),              mock.patch.object(session, "_answer_beyond_sources", return_value=None):
            session._dispatch_with_router("最近一次JC讲的是哪篇文献？")
    assert seen["paper_keys"] == ["B"] and seen["query"].endswith("（2026-09-12 JC讲的文献：Paper Beta）"), seen


def test_paper_scope_searches_only_paper_text():
    """2026-10-02真机："那原文里是怎么说的"拿到的是讲稿（原文是英文，中文讲稿笔记总排在前面）。
    路由给kb_scope="paper"时只查文献原文/补充材料；跟jc_session一起给时两个条件都生效。"""
    seen = {}

    def fake_gate(query, query_rewriter=None, doc_types=None, paper_keys=None):
        seen.update(doc_types=doc_types, paper_keys=paper_keys)
        return [], False
    fake_catalog = {"papers": [{"key": "B", "title": "Paper Beta"}],
                    "jc_sessions": [{"date": "2026-09-12", "match": {"key": "B"}}]}
    r = _route("local_kb")
    session = _build_session([{**r, "response": {**r["response"], "kb_scope": "paper", "jc_session": "latest"}}])
    with mock.patch.object(session.kb_pipeline, "catalog", fake_catalog, create=True),          mock.patch.object(session.kb_pipeline, "retrieve_with_gate", side_effect=fake_gate),          mock.patch.object(session, "_answer_beyond_sources", return_value=None):
        session._dispatch_with_router("最近一次JC讲的那篇，原文里提到了哪些局限")
    assert tuple(seen["doc_types"]) == ("文献原文", "文献补充材料") and seen["paper_keys"] == ["B"], seen


def test_pipeline_doc_types_filter_keeps_only_project_reports():
    """真机验证里"这个课题用什么软件做alignment"答成了文献里的Bowtie1。带上
    doc_types=["课题报告"]之后，返回的片段只能来自两份课题报告，而且要找到
    原文里的CellRanger。依赖真实模型，跟这个文件其它真实检索用例一样。"""
    kb = _get_kb_pipeline()
    q = "这个课题的scRNA-seq数据用什么软件做alignment？"
    passages = kb.retrieve(q, top_n=3, doc_types=["课题报告"])
    assert passages and all(p.chunk.meta.get("doc_type") == "课题报告" for p in passages), \
        [p.chunk.source_file for p in passages]
    assert any("CellRanger" in p.chunk.text.replace(" ", "") for p in passages), \
        [p.chunk.text[:80] for p in passages]


def test_group_chat_shares_conversation_but_not_profile():
    """阶段三群聊（用户2026-09-25定的方案）：画像按人、对话按群。
    ①同一个群的两个成员共享最近对话，用户发言前标上说话人（没有称呼时按
    出现顺序叫群友A/B）；②画像只属于各自；③群友说过称呼后改用称呼；
    ④翻旧账时来源标签分清"你说的话"和"别的群友说的话"，助手那条的问答对
    写的是当时提问的人；⑤同一个人的私聊跟群聊不相通；⑥规划背景里说明当前
    提问的是谁，而且不带上别人的画像。"""
    import conversation_memory as cm_module
    store = MemoryStore(db_path=":memory:")
    kb = _get_kb_pipeline()
    gid = "qq_group:g1"
    a_id, b_id = f"{gid}:A", f"{gid}:B"
    a = ConversationSession(kb, store, user_id=a_id, conversation_id=gid, llm_client=ScriptedLLMClient([]))
    b = ConversationSession(kb, store, user_id=b_id, conversation_id=gid, llm_client=ScriptedLLMClient([]))
    private_a = ConversationSession(kb, store, user_id="qq_c2c:A", llm_client=ScriptedLLMClient([]))

    store.append_turn(gid, "user", "能天使是什么职业？", speaker=a_id)
    store.append_turn(gid, "assistant", "能天使是狙击干员。")
    store.append_turn(gid, "user", "那她的天赋呢", speaker=b_id)

    texts = [r["text"] for r in b.get_context_window()]
    labels = b._participant_labels(b._speaker_names(store.get_all_turns(gid)))
    assert texts == [f"[{labels[a_id]}] 能天使是什么职业？", f"[助手回复给{labels[a_id]}] 能天使是狙击干员。", f"[{labels[b_id]}] 那她的天赋呢"], texts
    assert [r["text"] for r in a.get_context_window()] == texts  # 同一个群看到的是同一段对话
    assert private_a.get_context_window() == []  # 私聊跟群聊不相通

    store.upsert_profile_fact(a_id, "称呼", "小王")
    texts = [r["text"] for r in b.get_context_window()]
    labels = b._participant_labels(b._speaker_names(store.get_all_turns(gid)))
    assert texts[0] == f"[{labels[a_id]}] 能天使是什么职业？" and texts[2] == f"[{labels[b_id]}] 那她的天赋呢", texts
    assert store.get_profile(b_id) == {} and store.get_profile("qq_c2c:A") == {}

    rows = b._build_planner_context(b.get_context_window())
    assert f"当前提问的是【{labels[b_id]}】" in rows[0]["text"], rows[0]
    assert not any("（用户画像，代表当前已确认的最新状态）称呼：小王" in r["text"] for r in rows), rows

    with mock.patch.object(cm_module, "L2_BM25_FLOOR", 0.0), \
         mock.patch.object(cm_module, "L2_RERANK_FLOOR", 0.0):
        by_b = {p.chunk.chunk_id: p for p in b._search_history("能天使是什么职业", top_n=5)}
        by_a = {p.chunk.chunk_id: p for p in a._search_history("能天使是什么职业", top_n=5)}
    assert by_b[0].chunk.source_file.endswith("前小王说的话"), by_b[0].chunk.source_file  # 2026-10-01起标签是"x分钟前…"
    assert by_a[0].chunk.source_file.endswith("前你说的话"), by_a[0].chunk.source_file
    assert by_b[1].chunk.text.startswith("（小王当时问：能天使是什么职业？）我回答："), by_b[1].chunk.text
    # 2026-10-02：回答别人的那条，标签写明回答的是谁；回答自己的照旧
    assert by_b[1].chunk.source_file.endswith("前我（助手）回答小王的话"), by_b[1].chunk.source_file
    assert by_a[1].chunk.source_file.endswith("前我（助手）的回答"), by_a[1].chunk.source_file


def test_group_speaker_names_priority_and_owner():
    """群里说话人怎么称呼（2026-09-28真机测试后）：绑定的主人→"长安"；自己说过的
    称呼 > 平台给的QQ昵称 > 群友A/B（编号按所有人出现顺序，保持稳定）。主人本人提问时，
    规划背景里要说明"这就是长安本人"，"我的开题报告"才会去查长安的资料。"""
    store = MemoryStore(db_path=":memory:")
    kb = _get_kb_pipeline()
    gid = "qq_group:g2"
    owner, a, b, c = f"{gid}:OWNERID", f"{gid}:A", f"{gid}:B", f"{gid}:C"
    for sp, text in [(owner, "我的开题报告讲了什么"), (a, "能天使是什么职业"), (b, "那她的天赋呢"), (c, "你好")]:
        store.append_turn(gid, "user", text, speaker=sp)
        store.append_turn(gid, "assistant", "（回答）")
    store.set_display_name(a, "昵称甲")
    store.set_display_name(b, "昵称乙")
    store.upsert_profile_fact(b, "称呼", "小王")  # 自己说过的称呼优先于QQ昵称
    with mock.patch.dict(os.environ, {"ARKSEEK_OWNER_IDS": "OWNERID"}):
        sess_c = ConversationSession(kb, store, user_id=c, conversation_id=gid, llm_client=ScriptedLLMClient([]))
        names = sess_c._speaker_names(store.get_all_turns(gid))
        assert [names[x] for x in (owner, a, b, c)] == ["长安", "昵称甲", "小王", "群友D"], names
        texts = [r["text"] for r in sess_c.get_context_window() if r["role"] == "user"]
        assert texts[-1] == f"[{sess_c._participant_labels(names)[c]}] 你好", texts
        sess_owner = ConversationSession(kb, store, user_id=owner, conversation_id=gid,
                                         llm_client=ScriptedLLMClient([]))
        rows = sess_owner._build_planner_context(sess_owner.get_context_window())
        assert f"当前提问的是【{sess_owner._participant_labels(names)[owner]}】" in rows[0]["text"]
        assert any("就是长安本人" in r["text"] for r in rows), rows
        # 非主人提问时没有"本人"这一行
        assert not any("本人" in r["text"] for r in sess_c._build_planner_context(sess_c.get_context_window()))


# ---------- 2026-09-29：记忆写入和回答分开（用户选的方案B） ----------

def _route_general(reply, chitchat=True):
    return {"caller": "route_query()", "response": {
        "route": "general", "keyword": "", "reply": reply, "is_chitchat": chitchat,
        "wiki_query_kind": "entity", "cargo_profession": "", "cargo_rarity": "", "needs_search": False}}


def _single(q):
    return {"caller": "decompose()", "response": {"sub_queries": [{"id": "q1", "query": q, "depends_on": []}]}}


def test_statement_plus_question_answers_both():
    """真机发现的问题（3/3）："说自己的信息+顺带一个问题"以前被画像判断当场截住，
    只回"已记住"，问题没人回答。现在画像照写，问题照常走回答流程，两样都有；
    "（已记下：…）"由代码加在末尾，不存进对话历史。"""
    q = "我现在改做肝纤维化了，这个方向有什么常用的分析方法？"
    script = [
        {"caller": "llm_l1_extract()", "response": {"updates": [{"key": "研究方向", "value": "肝纤维化"}]}},
        _single(q),
        _route_general("常用的有单细胞转录组、空间转录组和谱系追踪等。"),
    ]
    session = _build_session(script)
    session.store.upsert_profile_fact(session.user_id, "研究方向", "单细胞")
    reply = session.user_says(q)
    session.llm_client.assert_exhausted()
    assert "常用的有单细胞转录组" in reply, reply
    assert reply.endswith("（已记下：研究方向改为肝纤维化）"), reply
    assert session.store.get_profile(session.user_id)["研究方向"] == "肝纤维化"
    stored = session.store.get_all_turns(session.user_id)[-1]["text"]
    assert "已记下" not in stored, "（已记下：…）是给用户看的提示，不该存进对话历史"


def test_pure_statement_writes_profile_and_notes_only_real_changes():
    """纯陈述句：路由自然回应，末尾"（已记下：称呼：小周）"；再说一遍同样的内容，
    画像没变，不加提示（提示只反映实际写入）。"""
    script = [
        {"caller": "llm_l1_extract()", "response": {"updates": [{"key": "称呼", "value": "小周"}]}},
        _single("我叫小周"),
        _route_general("你好小周，很高兴认识你！", chitchat=False),
        {"caller": "llm_l1_extract()", "response": {"updates": [{"key": "称呼", "value": "小周"}]}},
        _single("对，我叫小周"),
        _route_general("嗯嗯，记得的～", chitchat=False),
    ]
    session = _build_session(script)
    first = session.user_says("我叫小周")
    second = session.user_says("对，我叫小周")
    session.llm_client.assert_exhausted()
    assert first == "你好小周，很高兴认识你！\n\n（已记下：称呼：小周）", first
    assert second == "嗯嗯，记得的～", second


def test_profile_question_answered_by_router_with_profile_in_context():
    """"我是谁"不再由画像判断截住直接列字段，而是交给路由：路由看到的背景里有
    "用户画像"那一行，读着它直接回答（route=general，一次调用）。这里验证画像
    确实在路由的输入里，且没写入东西时不加"已记下"。"""
    script = [
        {"caller": "llm_l1_extract()", "response": {"updates": []}},
        _single("我是谁"),
        _route_general("你是长安呀～"),
    ]
    session = _build_session(script)
    session.store.upsert_profile_fact(session.user_id, "称呼", "长安")
    reply = session.user_says("我是谁")
    client = session.llm_client
    client.assert_exhausted()
    assert reply == "你是长安呀～", reply
    route_input = [c for c in client.calls if c["caller"] == "route_query()"][0]["messages"][-1]["content"]
    assert "（用户画像，代表当前已确认的最新状态）称呼：长安" in route_input, route_input


def test_memory_extract_failure_falls_back_to_regex_and_still_answers():
    """画像抽取的LLM调用失败：退回2.1的正则抽取（只写不答），回答流程不受影响。"""
    script = [
        {"caller": "llm_l1_extract()", "response": None},
        _single("我叫小周，今天好冷"),
        _route_general("是啊，多穿点～"),
    ]
    session = _build_session(script)
    reply = session.user_says("我叫小周，今天好冷")
    session.llm_client.assert_exhausted()
    assert reply == "是啊，多穿点～\n\n（已记下：称呼：小周）", reply


def test_rewrite_that_answers_or_drifts_falls_back_to_original():
    """第二道防线（2026-09-29）：检索前的改写（resolve_for_retrieval）偶尔会把请求句
    写成一段回答、或者编出新内容（真机6次里3次）。改写结果像回答、或者原句的实词大多
    不见了，就丢掉改写、用原句检索；正常的代词补全照常采用。"""
    from query_planner import resolve_for_retrieval
    ctx = [{"role": "system", "text": "（当前跟你说话的就是长安本人）"}]
    q = "那你介绍一下我的开题报告吧，100字以内"
    cases = [
        ("抱歉，我这里没有你的开题报告内容，你方便贴一下吗？", q),
        ("长安的开题报告题目是《某某理论视域下的某某研究》，研究某某机制。", q),
        ("那你介绍一下长安的开题报告吧，100字以内", "那你介绍一下长安的开题报告吧，100字以内"),
    ]
    for rewritten, expected in cases:
        client = ScriptedLLMClient([{"caller": "resolve_for_retrieval()", "response": rewritten}])
        got = resolve_for_retrieval(q, ctx, client)
        client.assert_exhausted()
        assert got == expected, f"改写={rewritten!r} 实际用了={got!r}"
    # 群里别人问同一句（背景里没有主人说明，最近对话里长安刚问过）："我"不能被换成长安
    friend_ctx = [{"role": "user", "text": "[长安] 那你介绍一下我的开题报告吧，100字以内"}]
    client = ScriptedLLMClient([{"caller": "resolve_for_retrieval()",
                                 "response": "那你介绍一下长安的开题报告吧，100字以内"}])
    got = resolve_for_retrieval(q, friend_ctx, client)
    assert got == q, f"群友的'我'被换成了长安：{got!r}"


# ---------- 2026-09-29：用户的字数要求（见length_limit.py） ----------

def test_length_request_parsing_and_body_split():
    """识别字数要求的各种说法；没有明确数字/"一句话"的不算；拆出正文时来源标题、
    引用列表、来源标注都不算字数，三段拼回去等于原文。"""
    from length_limit import parse_length_request as p, split_reply, count_units
    cases = {
        "那你介绍一下我的开题报告吧，100字以内": (100, None, "字"), "不超过200字说说DARLIN": (200, None, "字"),
        "用一百字介绍能天使": (100, None, "字"), "控制在八十字": (80, None, "字"), "50字左右": (50, None, "字"),
        "至少300字讲讲谱系追踪": (None, 300, "字"), "500字以上": (None, 500, "字"), "200到300字介绍一下": (300, 200, "字"),
        "Explain CoSpar in 50 words": (50, None, "words"), "at least 200 words please": (None, 200, "words"),
        "请用100内介绍玛恩纳的技能": (100, None, "字"),  # 少写了"字"（2026-10-01真机）
    }
    for text, (mx, mn, unit) in cases.items():
        r = p(text)
        assert r and (r.max_units, r.min_units, r.unit) == (mx, mn, unit), f"{text!r} -> {r}"
    assert p("一句话说说CoSpar是干什么的").one_sentence
    for text in ["what do these 3 words mean", "这篇文章有3000字吗", "你的名字是什么", "简短点说",
                 "费用10以内的干员有哪些", "三天内能到吗"]:
        assert p(text) is None, f"{text!r}不该被当成字数要求：{p(text)}"
    reply = "【L3知识库——LLM生成回答，引用来源见下】\n长安的开题报告里用的是**10x Genomics**[1]。\n\n引用来源：\n[1] 长安的课题开题报告"
    head, body, tail = split_reply(reply)
    assert head + body + tail == reply and tail.lstrip().startswith("引用来源") and head.startswith("【L3")
    assert count_units(body) == len("长安的开题报告里用的是10xGenomics。"), count_units(body)
    # 检索用的查询去掉字数要求（原样进检索会拉低相关性），生成时仍用完整原句
    from length_limit import strip_length_request as strip
    assert strip("至少300字讲讲这个课题的实验设计") == "讲讲这个课题的实验设计"
    assert strip("介绍一下这个课题的开题报告，100字以内") == "介绍一下这个课题的开题报告"
    assert strip("Explain what CoSpar does in 40 words") == "Explain what CoSpar does"
    assert strip("能天使是什么职业？") == "能天使是什么职业？"


_LONG_BODY = "长安的开题报告" + "围绕肝脏损伤修复展开，用CRISPR条形码结合单细胞转录组做谱系追踪[1]。" * 6
_KB_REPLY = f"【L3知识库——LLM生成回答，引用来源见下】\n{_LONG_BODY}\n\n引用来源：\n[1] 长安的课题开题报告"


def _length_session(script, dispatched_reply):
    session = _build_session(script)
    return session, mock.patch.object(session, "_dispatch_with_planner", return_value=(dispatched_reply, False))


def test_over_limit_reply_is_not_condensed():
    """正文超出上限很多：2026-10-01起不再调LLM压缩（推理模型压缩时反复逐字数、经常
    超时），原样回复、原样存进历史，只打日志。"""
    session, patch = _length_session([{"caller": "llm_l1_extract()", "response": {"updates": []}}], _KB_REPLY)
    with patch:
        reply = session.user_says("介绍一下这个课题的开题报告，50字以内")
    session.llm_client.assert_exhausted()
    assert reply == _KB_REPLY, reply
    assert session.store.get_all_turns(session.user_id)[-1]["text"] == reply


def test_within_tolerance_or_min_length_not_touched():
    """超出不到20%不压缩；"至少N字"写短了也不扩写——两种情况都不多调LLM。"""
    body = "甲" * 110
    for question, dispatched in [("说说看，100字以内", body), ("至少300字讲讲", _KB_REPLY)]:
        session, patch = _length_session([{"caller": "llm_l1_extract()", "response": {"updates": []}}], dispatched)
        with patch:
            reply = session.user_says(question)
        session.llm_client.assert_exhausted()
        assert reply == dispatched, (question, reply)


def test_generation_failure_explains_instead_of_dumping_raw():
    """真实LLM重试后仍然生成失败（2026-10-01）：本地知识库、翻聊天记录都给"如实说明
    +资料来源"，不把原始检索片段/原始聊天记录倒给用户（用户的核心要求：用户看到的
    应当是整理过的回答）。资料是对的，所以ok=True、不去联网兜底。"""
    import conversation_memory as cm_module

    # 本地知识库
    session = _build_session(script=[{"caller_prefix": "llm_generate_answer(", "response": None}])
    p_gate, p_retrieve, p_real = _local_kb_mocks(session, [True])
    with p_gate, p_retrieve, p_real:
        kb = session._try_answer_from_local_kb("多组学数据整合有哪些方法？")
    session.llm_client.assert_exhausted()
    assert kb.ok and "没能生成回答" in kb.display and "找到的资料：" in kb.display, kb.display
    assert "检索到" not in kb.display and "RRF=" not in kb.display, kb.display
    print("    [例子·本地知识库]\n" + "\n".join("      " + ln for ln in kb.display.splitlines()))

    # 翻聊天记录
    session = _build_session(script=[{"caller": "llm_generate_answer(history)", "response": None}])
    _seed_turns(session, [
        ("user", "我们组决定用Chroma做向量数据库，因为轻量、Python里直接能用"),
        ("assistant", "好的，记下了～"),
    ])
    with mock.patch.object(cm_module, "L2_BM25_FLOOR", 0.5):
        hist = session._try_answer_from_history("我们组之前决定用什么做向量数据库来着？")
    session.llm_client.assert_exhausted()
    assert hist is not None and hist.ok and "没能生成回答" in hist.display, hist and hist.display
    assert "衰减后分数" not in hist.display and "Chroma" not in hist.display, hist.display
    print("    [例子·翻聊天记录]\n" + "\n".join("      " + ln for ln in hist.display.splitlines()))


def test_compact_citations_lists_only_cited_sources_once():
    """引用来源只列正文实际引用到的、同一来源只列一次、编号重排（2026-10-01真机：
    正文只引用[1]，列表却是三条一样的"长安的课题开题报告"）。"""
    from llm_generate import compact_citations
    kb = "长安的课题开题报告"
    # 真机那条：三个片段同一份文件，正文只引用[1]
    assert compact_citations("五条不足……[1]", [kb, kb, kb]) == ("五条不足……[1]", [kb])
    # 引用了[2][3]（同一份文件）和[1]：同一来源只列一次，相邻重复编号合并
    text, listed = compact_citations("甲[2][3]。乙[1]。", ["文献A", kb, kb])
    assert listed == [kb, "文献A"] and text == "甲[1]。乙[2]。", (text, listed)
    # 没用到的不列；超出范围的编号删掉
    text, listed = compact_citations("只用了[3]，还有个[9]", ["A", "B", "C"])
    assert listed == ["C"] and text == "只用了[1]，还有个", (text, listed)
    # 正文一个编号都没有：列去重后的全部来源，保留出处
    assert compact_citations("没有编号", [kb, kb, "A"]) == ("没有编号", [kb, "A"])
    # 2026-10-02真机：模型自己在正文末尾写了来源列表，去掉（列表由代码加）
    text, listed = compact_citations("最近一次是9月12日[1]。\n\n引用来源：\n[1] JC笔记\n[1] JC笔记", ["JC笔记"])
    assert text == "最近一次是9月12日[1]。" and listed == ["JC笔记"], (text, listed)


def test_history_uses_recent_window_as_evidence_when_search_misses():
    """2026-10-01真机：刚说完"我不喜欢游泳，只喜欢骑行"，问"我不喜欢什么"答成"没翻到"
    （检索把那句排第一，但精排0.38没过0.5门槛）。最近窗口里的发言本身就是聊天记录，
    不经过检索门槛直接当证据交给生成；来源标签是"x分钟前你说的话"。"""
    script = [{"caller": "rewrite_for_history()", "response": {"rewrites": []}},  # 检索没过门槛先改写重查
              {"caller": "llm_generate_answer(history)", "response": "你说过不喜欢游泳[1]。"}]
    session = _build_session(script)
    _seed_turns(session, [("user", "我不喜欢游泳，只喜欢骑行"), ("assistant", "好的，更新一下"),
                          ("user", "我不喜欢什么")])  # 最后一条是当前这句，不能当证据
    with mock.patch.object(session, "_search_history", return_value=[]):
        sub = session._try_answer_from_history("我不喜欢什么")
    session.llm_client.assert_exhausted()
    assert sub is not None and sub.ok and "不喜欢游泳" in sub.display, sub and sub.display
    assert "前你说的话" in sub.display and "第0轮" not in sub.display, sub.display


def test_profile_value_none_marker_deletes_key():
    """画像抽取给value="（无）"表示删掉这个key（2026-10-01："不喜欢"里唯一一项被移走），
    回复末尾提示"已清空"。"""
    script = [{"caller": "llm_l1_extract()", "response": {"updates": [
        {"key": "兴趣爱好", "value": "骑行和游泳"}, {"key": "不喜欢", "value": "（无）"}]}}]
    session = _build_session(script)
    session.store.upsert_profile_fact(session.user_id, "兴趣爱好", "骑行")
    session.store.upsert_profile_fact(session.user_id, "不喜欢", "游泳")
    written = session._l1_write("最近又爱上游泳了")
    session.llm_client.assert_exhausted()
    assert session.store.get_profile(session.user_id) == {"兴趣爱好": "骑行和游泳"}
    note = session._memory_note(written)
    assert "不喜欢已清空" in note and "兴趣爱好改为骑行和游泳" in note, note


def test_history_rewrite_retry_uses_higher_floor():
    """翻聊天记录第一次没过门槛：改写成陈述句重查（rewrite_for_history），改写路径用
    更高的门槛L2_REWRITE_RERANK_FLOOR——过了的记录交给生成，没过的不用。"""
    import conversation_memory as cm_module
    from pipeline import RetrievedPassage
    from chunking import Chunk

    def passage(turn, text, rel):
        p = RetrievedPassage(chunk=Chunk(chunk_id=turn, source_file="3天前你说的话", text=text),
                             rrf_score=0.0, bm25_score=0.0, dense_score=0.0)
        p.relevance_score = p.rerank_score = rel
        return p

    for rel, should_use in [(0.8, True), (0.6, False)]:
        script = [{"caller": "rewrite_for_history()", "response": {"rewrites": ["我不喜欢……"]}}]
        if should_use:
            script.append({"caller": "llm_generate_answer(history)", "response": "你说过不喜欢香菜[1]。"})
        session = _build_session(script)
        _seed_turns(session, [("user", "我不吃香菜"), ("assistant", "好的")] + [("user", "闲聊"), ("assistant", "嗯")] * 4
                    + [("user", "我不喜欢什么")])
        hit = passage(0, "我不吃香菜", rel)
        search = lambda q: [hit] if q == "我不喜欢……" else []
        with mock.patch.object(session, "_search_history", side_effect=search),              mock.patch.object(session.kb_pipeline.reranker, "is_real", True),              mock.patch.object(session, "_recent_window_passages", return_value=[]),              mock.patch.object(session, "_try_answer_from_profile_fallback", return_value=None):
            sub = session._try_answer_from_history("我不喜欢什么")
        session.llm_client.assert_exhausted()
        assert (sub is not None and "香菜" in sub.display) == should_use, (rel, sub and sub.display)


def test_history_hit_on_question_brings_following_reply():
    """命中用户当时的提问时，带上紧跟着的那条助手回复（2026-10-01测试题F7："你上次推荐
    我看的电影"命中了"推荐个电影呗"，答案在下一条助手回复里没带进来）。"""
    import conversation_memory as cm_module
    from pipeline import RetrievedPassage
    from chunking import Chunk
    session = _build_session([])
    _seed_turns(session, [("user", "周末想看部电影，推荐一个呗"), ("assistant", "可以看看《星际穿越》"),
                          ("user", "今天天气不错"), ("assistant", "是呀"),
                          ("user", "你上次推荐我看的电影是哪部")])
    ask = RetrievedPassage(chunk=Chunk(chunk_id=0, source_file="3天前你说的话", text="周末想看部电影，推荐一个呗"),
                           rrf_score=0.0, bm25_score=0.0, dense_score=0.0)
    ask.relevance_score = ask.rerank_score = 0.9
    seen = {}

    def fake_generate(query, passages, client, **kw):
        seen["texts"] = [p.chunk.text for p in passages]
        return "我推荐的是《星际穿越》[2]。"
    with mock.patch.object(session, "_search_history", return_value=[ask]),          mock.patch.object(session, "_recent_window_passages", return_value=[]),          mock.patch.object(cm_module, "llm_generate_answer", side_effect=fake_generate):
        sub = session._try_answer_from_history("你上次推荐我看的电影是哪部")
    assert any("星际穿越" in t for t in seen["texts"]), seen
    assert "星际穿越" in sub.display and "我（助手）的回答" in sub.display, sub.display


TESTS = [
    test_search_history_filters_low_score_candidates,
    test_dispatch_router_general_skips_resolve_for_retrieval,
    test_dispatch_router_wiki_skips_resolve_for_retrieval,
    test_dispatch_router_history_uses_resolved_query_for_search,
    test_execute_sub_query_with_dependency_always_resolves_before_routing,
    test_execute_sub_query_without_dependency_skips_resolve,
    test_multi_hop_fanout_end_to_end,
    test_disambiguate_wiki_candidates_rejects_titles_outside_candidate_set,
    test_detect_answer_candidates_truncates_to_max_candidates,
    test_local_kb_gate_fail_triggers_rewrite_and_retrieves_with_it,
    test_local_kb_gate_pass_skips_rewrite,
    test_local_kb_no_rewrite_keeps_rejecting_without_generation,
    test_pipeline_retrieve_actually_uses_rewrites,
    test_clean_assistant_text,
    test_search_history_retrieves_assistant_reply_with_assistant_label,
    test_search_history_downweights_general_chat_instead_of_excluding,
    test_search_history_gate_uses_pre_decay_relevance,
    test_auto_intro_only_on_first_contact_and_not_stored,
    test_general_needs_search_goes_to_web_with_label,
    test_general_search_uses_router_search_query,
    test_general_subjective_does_not_search,
    test_search_failure_falls_back_to_own_knowledge_with_label,
    test_wiki_miss_goes_to_web_not_local_kb,
    test_local_kb_no_answer_falls_back_with_kb_miss_note,
    test_history_miss_never_searches_web,
    test_merge_labels_each_unverified_part_separately,
    test_catalog_question_answered_from_catalog_without_generation,
    test_catalog_scope_lists_only_what_was_asked,
    test_jc_session_resolved_from_catalog_by_date,
    test_paper_scope_searches_only_paper_text,
    test_merge_all_verified_keeps_citations,
    test_project_scope_passes_doc_filter_to_local_kb,
    test_pipeline_doc_types_filter_keeps_only_project_reports,
    test_group_chat_shares_conversation_but_not_profile,
    test_group_speaker_names_priority_and_owner,
    test_statement_plus_question_answers_both,
    test_pure_statement_writes_profile_and_notes_only_real_changes,
    test_profile_question_answered_by_router_with_profile_in_context,
    test_memory_extract_failure_falls_back_to_regex_and_still_answers,
    test_rewrite_that_answers_or_drifts_falls_back_to_original,
    test_length_request_parsing_and_body_split,
    test_over_limit_reply_is_not_condensed,
    test_within_tolerance_or_min_length_not_touched,
    test_generation_failure_explains_instead_of_dumping_raw,
    test_compact_citations_lists_only_cited_sources_once,
    test_history_uses_recent_window_as_evidence_when_search_misses,
    test_profile_value_none_marker_deletes_key,
    test_history_rewrite_retry_uses_higher_floor,
    test_history_hit_on_question_brings_following_reply,
]


def run():
    results = []
    for test_fn in TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            results.append({"name": name, "passed": True, "detail": ""})
        except AssertionError as e:
            results.append({"name": name, "passed": False, "detail": str(e)})
        except Exception as e:
            results.append({"name": name, "passed": False, "detail": f"{type(e).__name__}: {e}"})
    return results


def render_markdown(results):
    n_pass = sum(1 for r in results if r["passed"])
    lines = [
        "# 编排层回归测试结果（结构/接线验证，不含真实LLM语义验证）",
        "",
        f"通过 {n_pass}/{len(results)}。",
        "",
        "这套测试用 ScriptedLLMClient 代替真实LLM，验证的是"
        "代码收到LLM某个回复之后做什么"
        "（调用顺序、参数传递、分支/降级逻辑），不验证"
        "真实LLM会不会真的给出那个回复"
        "。decompose()/route_query()等prompt本身的语义质量，"
        "仍然只能靠 run_*_demo.py 这批脚本在你自己电脑上用真实模型验证，"
        "两者互补，详见本文件模块开头的说明。",
        "",
        "| 测试 | 结果 | 说明 |",
        "| --- | --- | --- |",
    ]
    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        detail = (r["detail"] or "-").replace("\n", " ").replace("|", "\\|")
        lines.append(f"| `{r['name']}` | {mark} | {detail} |")
    return "\n".join(lines)


def main():
    results = run()
    report = render_markdown(results)

    here = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(here, "orchestration_regression_test_results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    for r in results:
        mark = "PASS" if r["passed"] else "FAIL"
        print(f"[{mark}] {r['name']}")
        if not r["passed"]:
            print(f"       {r['detail']}")

    n_pass = sum(1 for r in results if r["passed"])
    print()
    print(f"{n_pass}/{len(results)} 通过。结果已写入 {out_path}")
    if n_pass < len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
