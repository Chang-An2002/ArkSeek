# -*- coding: utf-8 -*-
"""
目录类问题按范围作答的真机验证（2026-09-28，用户看测试题总览时发现的问题）

原来路由只要判成"列清单"（kb_query_kind="catalog"），目录那一步不管问什么都把
13篇全部列出来："长安做过几次JC、都讲了哪些"也列了没讲过的5篇，"作者读过CoSpar
那篇吗"这种只问一篇的也列了全部13篇。修复：路由多给catalog_scope（all/jc/not_jc），
只问某一篇的问题判成content走检索（检索到这篇的目录条目片段来回答）。

两部分：
  1. 路由判断：每题问3遍，看kb_query_kind/catalog_scope判得对不对、稳不稳
  2. 完整对话：每题一个新会话，看最终回答有没有多列、有没有答到点上

用法：
    cd src
    python run_catalog_scope_demo.py
"""

import os
from datetime import datetime

from llm_client import build_llm_client
from wiki_router import route_query

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "catalog_scope_demo_results.md")
RUNS = 3

# (问题, 期望(kb_query_kind, catalog_scope), 回答里必须有的, 回答里不能有的, 说明)
CASES = [
    ("作者都读过哪些文献？", ("catalog", "all"), ["CoSpar", "LoxCode"], [], "问全部：列全13篇"),
    ("长安做过几次JC？都讲了哪些文章？", ("catalog", "jc"), ["8次", "ArchR", "SPACE-seq"], ["CoSpar"],
     "只问JC：不该列没讲过的5篇（CoSpar没讲过）"),
    ("长安有哪些读过但没在JC上讲过的文献？", ("catalog", "not_jc"), ["CoSpar"], ["ArchR"],
     "只问没讲过的：不该列讲过的（ArchR讲过）"),
    ("作者读过CoSpar那篇文献吗？", ("content", "all"), ["CoSpar"], ["ArchR", "LoxCode"],
     "只问一篇：走检索，回答读过、没在JC上讲过，不该把目录列一遍"),
    ("LoxCode是哪次JC讲的？", ("content", "all"), ["LoxCode", "7月5日"], ["CoSpar", "ArchR"],
     "只问一篇：回答2025年7月5日"),
]


def main():
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能跑这个验证。")
        return
    lines = ["# 目录类问题按范围作答：真机验证", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             "## 1. 路由判断（每题3遍）", "", "| 问题 | 期望 | 各遍结果 | 结论 |", "|---|---|---|---|"]
    route_ok = 0
    for q, exp, *_ in CASES:
        got = []
        for _ in range(RUNS):
            r = route_query(q, client) or {}
            got.append((r.get("kb_query_kind", "?") if r.get("route") == "local_kb" else f"route={r.get('route')}",
                        r.get("catalog_scope", "?")))
        n = sum(g == exp for g in got)
        route_ok += n
        mark = "✅" if n == RUNS else ("❌" if n == 0 else "⚠️")
        print(f"{mark} {q} 期望{exp} 实际{got}")
        lines.append(f"| {q} | {exp[0]}/{exp[1]} | {' / '.join(f'{a}/{b}' for a, b in got)} | {mark} |")
    lines.insert(4, f"**路由 {route_ok}/{RUNS * len(CASES)} 次判对；完整对话见第2部分**\n")

    # 完整对话要加载知识库，放在路由判断之后
    from pipeline import MemorySystemPipeline
    from memory_store import MemoryStore
    from conversation_memory import ConversationSession
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    lines += ["", "## 2. 完整对话", ""]
    ans_ok = 0
    for i, (q, _, must, must_not, note) in enumerate(CASES, start=1):
        session = ConversationSession(kb, MemoryStore(":memory:"), user_id=f"cat_{i}", llm_client=client)
        reply = session.user_says(q)
        body = reply.split("引用来源：")[0]
        missing = [k for k in must if k not in body]
        extra = [k for k in must_not if k in body]
        ok = not missing and not extra
        ans_ok += ok
        print(f"\n=== {q} {'✅' if ok else '❌'}\n{reply}")
        lines += [f"### {i}. {q} {'✅' if ok else '❌'}", f"- 说明：{note}",
                  f"- 必须有：{must}（缺：{missing or '无'}）；不能有：{must_not}（多出：{extra or '无'}）",
                  "- 回复：", "", "  > " + reply.replace("\n", "\n  > "), ""]
    lines.insert(5, f"**完整对话 {ans_ok}/{len(CASES)} 符合预期**\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n路由 {route_ok}/{RUNS * len(CASES)}，完整对话 {ans_ok}/{len(CASES)}。报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
