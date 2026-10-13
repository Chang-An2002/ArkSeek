# -*- coding: utf-8 -*-
"""
④"索引/聚合/动态更新"接入`ConversationSession`之后的端到端人工验证脚本——
验证四个探测脚本（`probe_homepage.py`/`probe_index_pages.py`/`probe_cargo.py`/
`probe_stage_boss.py`）证明可行的三条子路径，接进正式代码（`wiki_router.py`
新增的`wiki_query_kind`/`cargo_profession`/`cargo_rarity`字段+
`prts_cargo_client.py`+`conversation_memory.py`的
`_try_answer_from_cargo_filter()`）之后，是不是真的能在一次真实对话里被
正确识别和回答，而不是只能在独立脚本里单独测。

【这次要验证什么，对应HANDOFF.md⑰记录的三条子路径】
1. 首页预聚合类（"今天哪个干员过生日""最近新增了哪些干员"）——验证
   `route_query()`能不能正确把keyword解析成"首页"，而不是尝试去找一个
   叫"今天哪个干员"的干员页面（那样会查询失败）。这条路径的抓取/清洗
   代码本身完全没有改动，纯粹是路由判断这一层新加的能力。
2. 跨干员筛选类（"六星狙击干员有哪些"）——验证`route_query()`能不能正确
   识别出`wiki_query_kind="cargo_filter"`并抽出`cargo_profession`/
   `cargo_rarity`，以及`_try_answer_from_cargo_filter()`能不能查到正确
   的干员名单、且回答里明确标注"不是LLM生成的文字"（防止用户误以为这份
   名单也有编造风险）。
3. 关卡代号类（"5-10的boss是谁"）——验证这条本来就该零改动work的路径
   （`_locate_page()`直接支持代号查找）在接入真实路由之后确实还是零改动
   work，没有被新加的cargo_filter逻辑意外干扰。

【也要确认没有引入回归】
穿插了几个属于"普通entity查询"的旧用例（"能天使是什么职业"这种），确认
这次新增的三个字段和一段新的prompt说明，没有让原来运行良好的普通干员
查询意外被误判成cargo_filter或者keyword解析出错。

【必须在你自己电脑上跑】
需要真实访问prts.wiki（`action=cargoquery`跟`action=parse`一样会被云端
沙盒的出站白名单拦在403）+真实LLM（MiniMax）——这次要验证的正是
`route_query()`这次新增的判断能力，LLM不可用的话这条路径整体不会生效。

用法：
    cd src
    python run_index_aggregation_demo.py
"""

import os

from conversation_memory import ConversationSession
from llm_client import build_llm_client
from memory_store import MemoryStore
from pipeline import MemorySystemPipeline

_log_lines = []


def log(msg: str = ""):
    print(msg)
    _log_lines.append(msg)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    out_path = os.path.join(here, "..", "index_aggregation_demo_results.md")

    llm_client = build_llm_client()
    log(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    if not llm_client.is_real:
        log("警告：LLM不可用——这个脚本测的正是route_query()这次新增的判断"
            "能力，LLM不可用的话这几条路径整体不会生效，先确认.env里的"
            "MINIMAX_API_KEY配好了。")
    log("")

    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore(db_path=":memory:")
    session = ConversationSession(kb_pipeline, store, user_id="demo_user", llm_client=llm_client)

    # 每一项：(问题, 这一句想验证什么)
    script = [
        ("能天使是什么职业？",
         "回归检查：普通entity查询，行为应该跟这次改动之前完全一致——"
         "route=wiki、keyword=能天使，不应该被新加的字段干扰"),
        ("今天哪个干员过生日",
         "子路径1（首页预聚合）：应该route=wiki、keyword=首页，不应该"
         "尝试把'今天哪个干员'当成一个干员名去搜索（那样会查询失败/查到"
         "无关页面）。回答里应该出现今天生日的干员名字（如果当天恰好没有"
         "任何干员过生日，首页对应小节内容为空，回答里诚实说没有也是"
         "正确的，不代表bug）"),
        ("最近新增了哪些干员和关卡",
         "同样是子路径1，换一种问法，验证不是只认得'今天生日'这一种固定"
         "句式——回答里应该出现首页'近期新增'小节列出的干员/关卡名字"),
        ("六星狙击干员有哪些",
         "子路径2（跨干员筛选，Cargo查询）：应该route=wiki、"
         "wiki_query_kind=cargo_filter、cargo_profession=狙击、"
         "cargo_rarity=6，回答里应该是一份具体的干员名单（能天使、白金"
         "这类六星狙击干员），并且明确标注'不是LLM生成的文字'这类说明，"
         "不应该走到entity那条'查一个具体页面'的路径"),
        ("有哪些五星医疗干员",
         "同样是子路径2，换一个职业+稀有度组合，验证不是只认得'六星狙击'"
         "这一个特例"),
        ("5-10的boss是谁",
         "子路径3（关卡代号）：应该route=wiki、keyword=5-10（原样给出"
         "代号，不需要翻译成'长夜终尽'这个正式标题），回答里应该正确"
         "指出浮士德和梅菲斯特是这一关的领袖（boss）——这条路径的底层"
         "代码完全没有改动，这里主要是确认接入真实路由后依然正常工作"),
    ]

    for question, checkpoint in script:
        log("=" * 70)
        log(f"你说：{question}")
        log(f"【这句话要验证】{checkpoint}")
        log("-" * 70)
        reply = session.user_says(question)
        log(reply)
        log("")

    log("=" * 70)
    log("整体重点检查：")
    log("0. '能天使是什么职业'这条回归检查，行为应该跟改动前完全一致，")
    log("   没有被新字段干扰。")
    log("1. '今天哪个干员过生日'/'最近新增了哪些干员和关卡'两条，有没有")
    log("   正确查到首页内容，而不是查询失败/查到无关页面。")
    log("2. '六星狙击干员有哪些'/'有哪些五星医疗干员'两条，有没有给出具体")
    log("   的干员名单（不是笼统地说'我不确定'），名单是否看起来合理（六星")
    log("   狙击应该包含能天使、白金这类知名干员）。")
    log("3. '5-10的boss是谁'这条，有没有正确说出浮士德/梅菲斯特是领袖，")
    log("   没有因为新代码的加入而意外退化。")
    log("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ④索引/聚合/动态更新——端到端验证结果（人工核对）\n\n```\n")
        f.write("\n".join(_log_lines))
        f.write("\n```\n")
    print(f"\n完整输出已写到：{os.path.abspath(out_path)}")
