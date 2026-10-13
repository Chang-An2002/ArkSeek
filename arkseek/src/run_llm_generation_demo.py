# -*- coding: utf-8 -*-
"""
2.2 L0滚动摘要 + L2/L3生成式回答的端到端验证脚本 —— 跟 run_llm_l1_demo.py
是同一个习惯：人工核对用，不是自动判分的测试套件，走完整 ConversationSession
流程，跑完把输出写成 llm_generation_demo_results.md，不用再手动复制粘贴。

跑三组场景：

1. **L3生成式回答（gate通过）**：问一个知识库能答上的问题，检查回复是不是
   变成了一段自然语言（带[编号]引用），而不是2.1那种"检索到N条相关片段"的
   证据罗列格式。

2. **L3拒答（gate不通过，不应该调用LLM）**：问一个知识库里没有的问题，检查
   回复是不是还是2.1原本那句拒答文案——即使真实LLM可用，gate没过也不应该
   生成任何内容，这是防幻觉的硬性防线，不是LLM生成的开关。

3. **L0滚动摘要**：故意把window_size调小（2），让对话很快就溢出窗口，跑几轮
   之后用 session.get_session_summary() 看摘要有没有被正确滚动更新。

（L2生成式回答复用的是跟L3同一个 llm_generate_answer() 函数，逻辑已经在L3
这组验证过了，这里不重复跑一遍L2专属场景，重点是想再确认一次L0摘要这个
之前完全没有真实调用验证过的新功能。）
"""

import os

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession
from llm_client import build_llm_client

_log_lines = []


def log(msg: str = ""):
    print(msg)
    _log_lines.append(msg)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    out_path = os.path.join(here, "..", "llm_generation_demo_results.md")

    kb_pipeline = MemorySystemPipeline(kb_dir)
    llm_client = build_llm_client()
    log(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    log(f"当前向量检索后端：{kb_pipeline.dense.name}")
    log(f"当前精排后端：{kb_pipeline.reranker.name}")
    if not llm_client.is_real:
        log("警告：LLM不可用，下面跑出来的会是2.1的规则版行为，看不出2.2的区别。")
    log("")

    # ---- 场景1：L3生成式回答 ----
    log("=" * 70)
    log("场景1：L3生成式回答（gate应该通过）")
    log("=" * 70)
    store1 = MemoryStore(db_path=":memory:")
    session1 = ConversationSession(kb_pipeline, store1, user_id="demo_gen_l3_answer",
                                    llm_client=llm_client)
    q = "多组学数据整合有哪些方法？"
    log(f"\n你说：{q}")
    log(session1.user_says(q))
    log("")

    # ---- 场景2：L3拒答（不应该调用LLM） ----
    log("=" * 70)
    log("场景2：L3拒答（gate不通过，不应该调用LLM，回复应该还是2.1原文案）")
    log("=" * 70)
    store2 = MemoryStore(db_path=":memory:")
    session2 = ConversationSession(kb_pipeline, store2, user_id="demo_gen_l3_reject",
                                    llm_client=llm_client)
    q = "今天晚饭吃什么？"
    log(f"\n你说：{q}")
    log(session2.user_says(q))
    log("")

    # ---- 场景3：L0滚动摘要 ----
    log("=" * 70)
    log("场景3：L0滚动摘要（window_size故意调小成2，方便快速触发窗口溢出）")
    log("=" * 70)
    store3 = MemoryStore(db_path=":memory:")
    session3 = ConversationSession(kb_pipeline, store3, user_id="demo_gen_l0_summary",
                                    window_size=2, llm_client=llm_client)
    script = [
        "我叫长安，是生命科学专业的研究生",
        "我在做单细胞谱系追踪方向的研究",
        "我们组决定用Chroma做向量数据库先跑通逻辑",
        "今天晚饭还没想好吃什么",
    ]
    for turn in script:
        log(f"\n你说：{turn}")
        log(session3.user_says(turn))
        log(f"[调试] 当前L0摘要：{session3.get_session_summary()}")
    log("")

    log("看完above三组结果，重点检查：")
    log("1. 场景1的回复是不是一段自然语言（带[1][2]这种引用标注），"
        "而不是「检索到N条相关片段」这种证据罗列格式。")
    log("2. 场景2的回复是不是还是【未找到可靠信息】这句话，没有被生成模型"
        "强行答出一个内容——这条最关键，验证的是防幻觉gate没有被绕过。")
    log("3. 场景3每一轮打印的[调试]摘要行，是不是从None开始，在窗口溢出"
        "（超过2轮）之后开始出现内容，并且随着对话推进不断滚动更新、"
        "覆盖掉旧值而不是越攒越长。")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 2.2 L0摘要+L2/L3生成式回答端到端验证结果（非自动化测试，人工核对）\n\n```\n")
        f.write("\n".join(_log_lines))
        f.write("\n```\n")
    print(f"\n完整输出已写到：{os.path.abspath(out_path)}")
