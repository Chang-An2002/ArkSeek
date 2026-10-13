# -*- coding: utf-8 -*-
"""
QQ群压力测试准备（二）的真机验证：查不到时 联网搜索 -> 大模型自身知识，分级标注

真实LLM + 真实联网搜索（MiniMax Token Plan，会消耗少量搜索次数）+ 真实BGE模型，
内存数据库，不碰data/memory.db。每道题一个新会话，互不影响。

逐项检查：
  - 客观问题走联网搜索，结尾是"【以上内容来自网络搜索，未经核实，请以实际为准】"
  - 主观闲聊直接回答，不带任何标注
  - wiki/本地知识库查得到的照常附引用，不带兜底标注
  - wiki查不到、本地知识库查不到（后者前面有"长安的资料里没有找到这部分内容"）
    都转去联网搜索
  - 复合问题各部分按来源分段标注
  - 之前遗留的B10（把"研究院"抄成"学院"）是否不再出现
  - 翻旧账查不到维持"没翻到"，不联网

用法：
    cd src
    python run_fallback_demo.py
"""

import os
from datetime import datetime

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession
from llm_client import build_llm_client

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "fallback_demo_results.md")

CASES = [
    ("推荐个讲明日方舟攻略的up主", "客观问题（推荐具体资源）：应走联网搜索，带网络搜索标注"),
    ("明日方舟哪个干员最强", "客观问题（社区共识）：应走联网搜索，带网络搜索标注"),
    ("你觉得明日方舟哪个干员最帅", "主观闲聊：直接回答，不带标注"),
    ("能天使是什么职业？", "wiki能查到：照常附引用，不带兜底标注"),
    ("主线第五章的最终boss是谁", "wiki查不到页面（之前实测0个候选）：应转去联网搜索"),
    ("变异检测常用什么工具？", "本地知识库拒答（run_tests第1题）：应转去联网搜索，前面说明作者课题资料里没有"),
    ("这个课题是和哪个课题组合作的？", "本地知识库能查到：照常附引用；检查B10——应写\"研究院\"不是\"学院\""),
    ("能天使是什么职业？另外推荐个明日方舟攻略up主", "复合问题：wiki部分和联网搜索部分分段，后者单独带标注"),
    ("我上次说的那个实验室代号叫什么来着？", "翻旧账查不到（新会话没有历史）：维持\"没翻到\"，不联网"),
]


def main():
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能跑这个验证。")
        return

    lines = ["# 查不到时的兜底链与分级标注：真机验证", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　LLM：{client.name}", ""]
    for i, (q, purpose) in enumerate(CASES, start=1):
        session = ConversationSession(kb, MemoryStore(":memory:"), user_id=f"fallback_{i}", llm_client=client)
        reply = session.user_says(q)
        searches = getattr(session.web_search, "calls", 0)
        print(f"\n=== {i}. {q}（联网搜索{searches}次）\n{reply}")
        lines += [f"## {i}. {q}", f"- 目的：{purpose}", f"- 联网搜索次数：{searches}", "- 回复：", "",
                  "  > " + reply.replace("\n", "\n  > "), ""]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已写到：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
