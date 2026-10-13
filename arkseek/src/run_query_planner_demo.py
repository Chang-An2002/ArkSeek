# -*- coding: utf-8 -*-
"""
检索架构升级（多跳/复合问题编排层）接入`ConversationSession`之后的端到端
人工验证脚本——完整设计动机见`query_planner.py`模块开头，这个脚本只负责
在真实LLM+真实prts.wiki网络下跑一遍真机验证，云端沙盒只做过结构性验证
（用假LLM client+假wiki抓取，验证接线逻辑对不对，不能验证真实LLM能不能
正确分解/解析代词/合并成一句话）。

【这次要验证什么】
1. 依赖式多跳（HANDOFF"接下来"第21条的具体案例）："玛恩纳的侄女的职业是
   什么"——应该被`decompose()`拆成两个有依赖关系的子查询（先查"侄女是谁"，
   再用查到的实体查"职业"），最终答案里两跳的内容都要对（人物+职业），
   不能只答出其中一步就停。这次是真的走检索验证这层关系，不是像"这次能
   答对不代表系统已经具备多跳能力"那次一样，靠LLM训练数据里记得的知识
   蒙对——如果这次也蒙对了，不能证明编排本身生效，需要重点核对回复里
   有没有明确体现出"先查到了侄女是谁，再查了这个人的职业"这个两步过程
   （比如引用来源里出现两个不同的wiki页面）。
2. 独立并列复合问题（HANDOFF"接下来"第15条）："之前说的向量数据库"+
   "能天使的种族"——应该被拆成两个互不依赖的子查询，一个走history一个走
   wiki，最终答案里两部分都要有，不能像改动前那样路由只能选一个、只答
   一半。
3. 单查询场景完全不受影响（回归测试）：随便一个阶段一/二反复验证过的
   老问题，答案和格式应该跟这次升级之前逐字一致——如果不一致，说明"完全
   退化到原有行为"这条设计承诺没有兑现，是需要立刻处理的回归。
4. 普通闲聊/陈述句依然正常：验证`decompose()`没有对不该拆的内容画蛇添足
   （比如把一句陈述硬拆成好几个子查询）。

【必须在你自己电脑上跑】
跟`run_conversation_wiki_demo.py`一样，需要真实访问prts.wiki（依赖式多跳
案例要查两次不同的wiki页面）、也需要真实LLM（MiniMax）key配置好——分解、
每一跳的路由判断、改写、最后的合并，全部是这次新增的LLM调用，云端沙盒的
假client测不出真实效果。

用法：
    cd src
    python run_query_planner_demo.py
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
    out_path = os.path.join(here, "..", "query_planner_demo_results.md")

    llm_client = build_llm_client()
    log(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    if not llm_client.is_real:
        log("警告：LLM不可用——这个脚本测的正是检索编排层这条2.2专属路径，"
            "LLM不可用的话decompose()会直接退化成不拆分，测不出这次改动的"
            "效果，先确认.env里的MINIMAX_API_KEY配好了。")
    log("")

    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore(db_path=":memory:")
    session = ConversationSession(kb_pipeline, store, user_id="demo_user", llm_client=llm_client)

    # 先铺垫一轮真实的历史陈述，供后面"独立并列复合问题"里的history部分翻找
    log("=" * 70)
    log("铺垫：先陈述一件事，供后面的history部分翻旧账")
    log("-" * 70)
    log("你说：我们组决定用Chroma做向量数据库")
    log(session.user_says("我们组决定用Chroma做向量数据库"))
    log("")

    script = [
        ("能天使是什么职业，天赋是什么？",
         "回归测试：单查询场景，应该完全退化到原有的_dispatch_with_router()"
         "行为，答案格式（【L3知识库——明日方舟wiki《能天使》...】这套包装）"
         "应该跟这次升级之前逐字一致，不应该出现'拆成几部分'这类新增的"
         "编排痕迹"),
        ("你好，今天天气不错",
         "回归测试：普通闲聊，decompose()不应该画蛇添足地把它拆成多个子"
         "查询，应该判定'不拆分'，正常走route=general"),
        ("我们组之前决定用什么做向量数据库来着，另外能天使是什么种族？",
         "重点用例①独立并列复合问题：应该被拆成两个互不依赖的子查询"
         "（一个翻旧账找Chroma、一个查wiki找能天使种族），最终回答要同时"
         "包含两部分内容，不能像改动前那样只答出其中一半"),
        ("玛恩纳的侄女的职业是什么？",
         "重点用例②依赖式多跳：应该被拆成'玛恩纳的侄女是谁'->'她的职业是"
         "什么'两个有依赖关系的子查询，第二步要能借助第一步查到的人物名字"
         "正确检索到职业信息，回答里两跳内容都要对（人物+职业），引用来源"
         "里应该看到两个不同的wiki页面（说明真的分两次查证了，不是蒙对的）"),
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
    log("1. '能天使是什么职业，天赋是什么'的回答格式是不是跟升级之前逐字")
    log("   一致（【L3知识库——明日方舟wiki《能天使》...】这套包装），没有")
    log("   出现'拆成几部分'这类新增的编排痕迹——这是'完全退化到原有行为'")
    log("   这条设计承诺的直接验证。")
    log("2. '你好，今天天气不错'有没有被误拆成多个子查询（正常应该是一次")
    log("   性判定不拆分，走route=general）。")
    log("3. 独立并列那条用例，回答里是不是同时出现了Chroma/向量数据库相关")
    log("   内容和能天使种族相关内容，不是只答了其中一半。")
    log("4. 依赖式多跳那条用例，回答里是不是同时出现了'侄女是谁'（人物名）")
    log("   和'职业'两部分内容，引用来源里是不是能看到两个不同的wiki页面")
    log("   （证明真的分两步查证了）——如果只有一个页面或者答案里缺了某一")
    log("   跳的内容，说明依赖注入没有生效，需要回头排查。")
    log("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 检索架构升级（多跳/复合问题编排层）——端到端验证结果（人工核对）\n\n```\n")
        f.write("\n".join(_log_lines))
        f.write("\n```\n")
    print(f"\n完整输出已写到：{os.path.abspath(out_path)}")
