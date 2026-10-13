# -*- coding: utf-8 -*-
"""
L2检索范围放开（用户+助手全部发言、闲聊降权不排除）的真机验证脚本

跑一段脚本化多轮对话（真实LLM + 真实BGE模型，内存数据库，不碰data/memory.db），
逐项验证当初排除助手发言/闲聊的两个理由现在已经处理好，以及放开之后新增的
两个要求：
  1. 闲聊自我延续（当年的故障）不复发：连续几次"吃了吗"都应该走general路由，
     不会被L2当成"翻旧账"反复引用
  2. 动机场景：请求建议（被标成闲聊）-> 助手推荐 -> 之后问"你上次推荐我
     吃什么来着"，现在应该能翻到助手那条推荐
  3. 归属：翻到助手说过的话时，回答要表述成"我之前推荐/回答过"，不能说成
     "你说过"
  4. 回归：用户自己陈述过的事实（Chroma那句）照旧能被翻出来，表述成"你说过"

用法：
    cd src
    python run_l2_scope_demo.py
"""

import os
from datetime import datetime

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession
from llm_client import build_llm_client


HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "l2_scope_demo_results.md")

SCRIPT = [
    ("吃了吗您", "闲聊，应走general"),
    ("吃了吗", "重复闲聊：当年的故障是这里被L2当成翻旧账、引用上一句闲聊，应仍走general"),
    ("今天晚饭吃什么？", "请求建议，应走general，助手给出推荐（这一轮会被标成闲聊）"),
    ("我们组决定用Chroma做向量数据库", "用户陈述事实，应记下（is_chitchat=False）"),
    ("中午好", "闲聊，把前面的内容往后推"),
    ("你上次推荐我晚饭吃什么来着？", "动机场景：应走history，翻到助手那条推荐，表述成\"我之前推荐过\""),
    ("我们组打算用什么做向量数据库来着？", "回归：应走history，翻到用户自己那句，表述成\"你说过\""),
]


def main():
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能跑这个验证。")
        return
    store = MemoryStore(":memory:")
    session = ConversationSession(kb, store, user_id="l2_scope_demo", llm_client=client)

    lines = ["# L2检索范围放开验证报告", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"向量检索：**{kb.dense.name}**　精排：**{kb.reranker.name}**　LLM：**{client.name}**", ""]
    for i, (say, purpose) in enumerate(SCRIPT, start=1):
        reply = session.user_says(say)
        print(f"\n=== 第{i}轮：{say}\n{reply}")
        lines += [f"## 第{i}轮：{say}", f"- 目的：{purpose}", "- 回复：", "",
                  "  > " + reply.replace("\n", "\n  > "), ""]

    rows = store.get_all_turns("l2_scope_demo")
    marked = [r["text"] for r in rows if r["role"] == "user" and r["is_general_chat"]]
    lines += ["## 闲聊标记", "", f"被标成闲聊（降权、不排除）的用户发言：{marked}", ""]
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n被标成闲聊的用户发言：{marked}\n报告已写到：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
