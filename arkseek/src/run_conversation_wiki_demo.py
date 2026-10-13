# -*- coding: utf-8 -*-
"""
"明日方舟万事通"接入`ConversationSession`之后的端到端人工验证脚本——
验证wiki查询是不是真的能在一次真实的多轮对话里被自然调用，而不是只能靠
独立脚本单独测。

【这次要验证什么】
`conversation_memory.py`新增了统一分流入口（现在是`_dispatch_with_router()`，
最初是`_try_answer_from_router()`，后来因为要修一个L2抢答导致标记漏打的
真实bug而重新设计，见该文件模块开头说明），`wiki_router.py`负责判断"这句
话该查prts.wiki、该查本地知识库、该去翻L2历史对话、还是纯粹一句寒暄不用
查任何东西"。这几块之前都只在沙盒里
用假LLM响应做过接线测试（验证"调用顺序对不对、失败了会不会正确退回本地
知识库"），没有真实验证过：①真实LLM能不能正确区分"问明日方舟"和"问本地
知识库"这两类问题；②路由判断出的关键词对不对（尤其是问题里有多个专有
名词、或者没有明确指向某个干员的时候）；③L1/L2的判断是不是仍然优先于
wiki路由生效，不会因为新增了这条路径就抢了L1/L2本该处理的问题。

【测试脚本的设计意图】
这版是第三次迭代，前两次真机测试各暴露了一个真实问题，都已经修复，这次
一并验证：①"你好"被误判成检索性问题拒答→修复成route="general"直接生成
问候；②"你好，还记得我是谁吗"这类复合句式只答问题不接寒暄→修复成
llm_l1_turn()顺带识别寒暄成分（greeting_reply字段）；③"今天吃什么，推荐
几道菜"这类不涉及两个知识库、但也不是纯寒暄的请求被当成"问题答不上来"
拒答→跟①合并成同一个修复，route="general"现在覆盖寒暄+闲聊+请求建议+
观点性问题+其他通用问题，统一自然回应，不再区分"chitchat"和"none"。

【本轮新增：材料图标名称丢失 + 代词指代解析不出来这两个修复的端到端验证】
又加了两轮对话（见脚本末尾第6/7条）：先问"能天使的三技能专三的材料是什么"，
验证`prts_wiki_client.py`"坑E"那次修复（材料图标名字不再丢失）；紧接着换成
"她精二的材料是什么"这种用代词指代的追问，验证`wiki_router.py`"真机测试
暴露的第6个问题"那次修复（`route_query()`能借助上一轮对话解析出"她"指的
是能天使，正确判给route=wiki，而不是误判成route=history）。这两个修复
都只能在真实网络+真实LLM下端到端验证，云端沙盒做不到。

下面这组对话穿插了这几类问题：
1. 本地知识库能答的问题（跟之前阶段一/二的测试用例一致）——应该走
   route="local_kb"，wiki路由完全不介入。
2. 明日方舟wiki问题，包含之前反复验证过的"能天使"（职业+天赋复合问题，
   方便直接对比这次端到端的效果跟`run_prts_wiki_demo.py`独立测试时是否
   一致）和一个全新的干员（换一个名字，验证路由的关键词提取不是只认
   "能天使"这一个特例）。
3. 一句纯寒暄（"你好"）——应该route="general"，直接给自然的问候回复。
4. 一句"寒暄+问自己信息"的复合句（"你好，还记得我是谁吗"）——应该命中
   L1，且回复里既有寒暄又有称呼信息，不能只答一半。
5. 一句先陈述再追问自己信息的对话——验证L1仍然优先于L3生效，不会因为
   新增路由路径就意外触发route_query()（这个多余的LLM调用本身也是一种
   浪费，即使最后结果没错也不该发生）。
6. 一句真正跟两个知识库都无关、但也不是在提问验证性事实的请求（"今天吃
   什么，推荐几道菜"）——应该route="general"，得到实际的推荐内容，而不是
   被当成"检索不到证据"拒答（这是第二轮真机测试之后新增的用例，验证跟
   第3类用例区分开的same-bucket合并逻辑）。

【必须在你自己电脑上跑】
跟`prts_wiki_client.py`一样，需要真实访问prts.wiki；同时因为这次接入了
`ConversationSession`的完整L1/L2/L3流程，也需要真实LLM（MiniMax）key
配置好——这次不是mock测试，是真刀真枪的端到端验证。

用法：
    cd src
    python run_conversation_wiki_demo.py
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
    out_path = os.path.join(here, "..", "conversation_wiki_demo_results.md")

    llm_client = build_llm_client()
    log(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    if not llm_client.is_real:
        log("警告：LLM不可用——这个脚本测的正是wiki路由+生成这条2.2专属路径，"
            "LLM不可用的话wiki路径整体不会生效，测不出这次改动的效果，"
            "先确认.env里的MINIMAX_API_KEY配好了。")
    log("")

    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore(db_path=":memory:")
    session = ConversationSession(kb_pipeline, store, user_id="demo_user", llm_client=llm_client)

    # 每一项：(问题, 这一句想验证什么)
    script = [
        ("你好",
         "应该route=general，直接给一句自然的问候回复，不应该落到本地知识库"
         "gate走拒答流程（第一轮真机测试发现的真实体验问题）"),
        ("多组学数据整合有哪些方法？",
         "应该route=local_kb，走本地知识库正常生成回答"),
        ("能天使是什么职业，天赋是什么？",
         "应该route=wiki、keyword=能天使，职业+天赋都要答对、回答有条理"
         "（跟run_prts_wiki_demo.py独立测试时的效果对比一下）"),
        ("推进之王的技能是什么？",
         "换一个干员名，验证路由提取关键词不是只认得'能天使'这一个特例"),
        ("我叫长安，我的研究方向是单细胞谱系追踪",
         "这是L1该处理的自我介绍，不应该触发路由判断"),
        ("你好，还记得我是谁吗",
         "复合句式：既是寒暄又是问自己信息。应该命中L1，且回复里既接住了"
         "'你好'的寒暄、又答出了'称呼'，不能只答一半（第二轮真机测试发现的"
         "真实体验问题）"),
        ("今天吃什么，推荐几道菜吧",
         "跟两个知识库都无关，但也是个正常的闲聊/建议请求，不是在验证某个"
         "具体事实。应该route=general，得到实际的推荐内容，而不是被当成"
         "'检索不到证据'拒答（第二轮真机测试发现的真实体验问题：不应该只是"
         "个查证据的工具，首先得是个正常的聊天助手）"),
        ("能天使的三技能专三的材料是什么",
         "应该route=wiki、keyword=能天使，回答里材料名字要正确显示（比如"
         "'技巧概要·卷3''D32钢''聚酸酯块'这类名字），不能只有一串孤立的数字——"
         "这是验证'材料图标名称在HTML清洗中丢失'这个bug的修复（见"
         "prts_wiki_client.py模块开头'坑E'）"),
        ("她精二的材料是什么",
         "这句话本身用的是代词'她'，真机测试踩过这里被误判成route=history、"
         "回复固定的'没翻到相关记录'——应该借助上一句的对话上下文，正确"
         "解析出'她'指的是能天使，判给route=wiki、keyword=能天使，正常答出"
         "精二材料，而不是落到history的兜底文案（见wiki_router.py模块开头"
         "'真机测试暴露的第6个问题'）"),
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
    log("0. '你好'有没有得到一句自然的问候回复，而不是被误判成检索性问题、")
    log("   落到本地知识库gate拒答。")
    log("1. 明日方舟问题是不是都正确路由到了wiki（回复里带'明日方舟wiki'标注），")
    log("   本地知识库问题有没有被误路由到wiki或general。")
    log("2. '能天使'那条职业+天赋是不是都答对了、回答有条理——这个跟")
    log("   run_prts_wiki_demo.py独立测试的效果理论上应该一致，因为底层用的")
    log("   是同一个fetch_page_for_answer()+llm_generate_answer()。")
    log("3. '我叫长安...'有没有被L1正确拦下，没有意外触发路由判断（如果")
    log("   回复里出现了wiki/general相关内容就是问题）。")
    log("4. '你好，还记得我是谁吗'这句复合句，回复里是不是既有寒暄又有'称呼'，")
    log("   不能只答其中一半（这是这次新增的重点检查项）。")
    log("5. '今天吃什么，推荐几道菜吧'有没有得到实际的推荐内容，而不是被误判")
    log("   成'检索不到证据'拒答（这也是这次新增的重点检查项）。")
    log("6. '能天使的三技能专三的材料是什么'这轮，材料名字（技巧概要·卷3/")
    log("   D32钢/聚酸酯块这类）有没有正确出现在回答里，而不是只有一串孤立")
    log("   数字——这是本轮新增的重点检查项，验证'材料图标名称丢失'的修复。")
    log("7. '她精二的材料是什么'这轮，有没有借助上一句的上下文正确解析出")
    log("   '她'=能天使、正常答出精二材料，而不是落到'这个我这边没翻到相关的")
    log("   记录'这句history兜底文案——这也是本轮新增的重点检查项，验证")
    log("   '代词指代解析不出来被误判成history'这个bug的修复。")
    log("=" * 70)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# ConversationSession接入明日方舟wiki——端到端验证结果（人工核对）\n\n```\n")
        f.write("\n".join(_log_lines))
        f.write("\n```\n")
    print(f"\n完整输出已写到：{os.path.abspath(out_path)}")
