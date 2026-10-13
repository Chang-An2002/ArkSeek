# -*- coding: utf-8 -*-
"""候选列表消歧机制接入`ConversationSession`之后的端到端人工验证脚本——
验证"关键词短语没有对应任何真实页面标题"这类问题（README.md"接下来"
第15条里如实记录为"仍未解决"的那一半，跟④"索引/聚合/动态更新"解决的
三条子路径是相关但不同的问题）现在能不能被正确处理。

【这次要验证什么】
`prts_wiki_client.py`新增`search_page_candidates()`（模糊搜索取多条候选，
不再无条件只取第一条）+`_locate_page()`新增`disambiguator`可选回调；
`wiki_router.py`新增`disambiguate_wiki_candidates()`（真正做判断的LLM
调用：读用户原问题+候选标题+搜索摘要，选出确定对应的那一条，拿不准就
诚实返回None，不勉强选一个）；`conversation_memory.py`的
`_try_answer_from_wiki_page()`把这个函数包成闭包传给
`fetch_page_for_answer()`。完整设计动机见`prts_wiki_client.py`模块开头
"坑A遗留的一个更根本的问题"一节+`wiki_router.py`模块开头"新增
disambiguate_wiki_candidates()"一节。

【这次要验证的具体行为，不是"必须查到正确答案"】
"主线第五章的最终boss是谁"这类问题，wiki上到底有没有一个候选能对应上，
完全取决于prts.wiki真实的页面组织方式（这次没有像④那样先跑探测脚本
确认过），所以这次真机验证要看的不是"必须答对"，而是这几件事有没有
如实发生：
1. 精确标题匹配失败之后，有没有真的走到"模糊搜索多条候选+消歧判断"这条
   新路径（看终端打印："模糊搜索有N个候选（标题列表）"这行日志）。
2. 如果候选里确实有一条对应正确页面，消歧判断能不能选中它、并且正确
   查到、答对（看有没有"消歧判断选中《xxx》"这行日志+最终回答是否正确）。
3. 如果候选里没有一条真的对应（大概率情况——"主线第五章"未必是任何
   页面标题的一部分），有没有诚实报告"没有找到"（看有没有"消歧判断不出
   确定的一个，诚实报告没有找到对应页面"这行日志+最终回答诚实说没查到，
   而不是编一个boss名字出来）。
   **这一种结果不代表bug**，只代表wiki上确实没有一个候选能精确回答这个
   问题，这本身就是这次要如实验证的边界，不是这次改动承诺解决的范围。
4. "能天使是什么职业"这条回归检查——精确标题命中，完全不会走到新增的
   候选/消歧逻辑，行为应该跟改动前完全一致。

【必须在你自己电脑上跑】
需要真实访问prts.wiki（云端沙盒的403限制见`prts_wiki_client.py`模块
开头）+真实LLM（MiniMax）——消歧判断本身就是一次新增的LLM调用，LLM不
可用的话这条路径不会生效。

用法：
    cd src
    python run_disambiguation_demo.py
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
    out_path = os.path.join(here, "..", "disambiguation_demo_results.md")

    llm_client = build_llm_client()
    log(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    if not llm_client.is_real:
        log("警告：LLM不可用——消歧判断本身就是一次新增的LLM调用，LLM不可用"
            "的话这条路径整体不会生效，先确认.env里的MINIMAX_API_KEY配好了。")
    log("")

    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore(db_path=":memory:")
    session = ConversationSession(kb_pipeline, store, user_id="demo_user", llm_client=llm_client)

    # 每一项：(问题, 这一句想验证什么)
    script = [
        ("能天使是什么职业？",
         "回归检查：精确标题直接命中，完全不会走到新增的候选/消歧逻辑，"
         "终端不应该出现'模糊搜索有N个候选'这行日志，行为应该跟改动前"
         "完全一致"),
        ("明日方舟主线第五章的最终boss是谁",
         "这次要验证的主要目标（README.md'接下来'第15条里如实记录的"
         "仍未解决问题）。重点看终端日志：有没有走到'模糊搜索有N个候选'"
         "这一步；如果有候选，消歧判断选中了哪一个、对不对；如果诚实"
         "报告'没有找到对应页面'，回答里应该也如实说明查不到、不应该"
         "编一个boss名字出来——这种结果本身不代表bug，代表wiki上确实"
         "没有一个候选能精确对应这个问题"),
        ("明日方舟第五章剧情讲的是什么",
         "换一种问法，同样是'章节'这类short phrase，看路由抽出的关键词"
         "是什么、消歧判断的候选和结果是否和上一条一致或呈现出新的情况"),
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
    log("0. '能天使是什么职业'这条回归检查，没有出现'模糊搜索有N个候选'")
    log("   这行日志（说明精确标题命中，新逻辑完全没被触碰到）。")
    log("1. '主线第五章'相关的两条，终端有没有出现'模糊搜索有N个候选")
    log("   （标题列表）'这行日志——这行日志出现，就说明新的候选列表机制")
    log("   真的被触发了，不管最终有没有选中一个候选。")
    log("2. 如果消歧判断选中了某个候选，回答内容对不对；如果诚实报告没")
    log("   找到，回答有没有如实说明、而不是编一个答案出来。")
    log("3. 把这几条日志和最终回答完整发回来，一起判断这次要不要往前")
    log("   继续投入（比如把search_page_candidates的limit调大、或者给")
    log("   disambiguate_wiki_candidates的prompt补充更多例子），还是就此")
    log("   记录为'诚实的已知局限，暂不继续投入'。")
    log("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 候选列表消歧机制——端到端验证结果（人工核对）\n\n```\n")
        f.write("\n".join(_log_lines))
        f.write("\n```\n")
    print(f"\n完整输出已写到：{os.path.abspath(out_path)}")
