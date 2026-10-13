# -*- coding: utf-8 -*-
"""
路由新字段 kb_scope（"project"=问作者自己的课题，只查课题报告 / "all"=全库）的
边界稳定性检查。跟 run_search_boundary_demo.py（needs_search的主客观边界）是同一
个思路：同一批问题各问几遍，看判断是否正确、是否前后一致。

为什么要查：kb_scope="project"会把检索范围收窄到两份课题报告——误判成project
的话，问文献/JC的问题会在报告里查不到而转去联网搜索；误判成all则回到知识库扩充
后"文献把'这个课题'的问题冲散"的老问题（B9答成文献里的Bowtie1）。

用法：
    cd src
    python run_kb_scope_boundary_demo.py
"""

import os
from collections import Counter
from datetime import datetime

from llm_client import build_llm_client
from wiki_router import route_query

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "kb_scope_boundary_demo_results.md")
RUNS = 3

# (问题, 期望的kb_scope, 说明)
CASES = [
    ("这个课题的scRNA-seq数据用什么软件做alignment？", "project", "B9，明确说'这个课题'"),
    ("这个project和哪个lab合作的？", "project", "口语+英文"),
    ("作者的课题用的是什么小鼠模型？", "project", "明确说'作者的课题'（群友可能这么叫）"),
    ("长安的课题用的是什么小鼠模型？", "project", "用昵称"),
    ("你们的实验用的是什么肝损伤模型？", "project", "'你们的实验'指作者这边"),
    ("作者的课题现在进展到哪一步了？", "project", "问进展"),
    ("lineage tracing课题使用的是什么动物的什么组织？", "project", "run_tests第6题，'课题'"),
    ("LoxCode是怎么产生高多样性条形码的？", "all", "问文献"),
    ("DARLIN小鼠和CARLIN相比改进了什么？", "all", "问文献对比"),
    ("作者在JC上是怎么介绍ArchR的？", "all", "问JC，不是课题"),
    ("25.11.29那次JC讲了哪篇文章？", "all", "问JC"),
    ("What problem does CoSpar solve?", "all", "英文问文献"),
    ("谱系追踪一般有哪些技术路线？", "all", "泛泛的领域问题"),
    ("LoxCode那篇文章的作者是谁？", "all", "'作者'指文献作者，不是长安的课题"),
]


def main():
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能跑这个验证。")
        return
    lines = ["# kb_scope 边界稳定性", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"每题问{RUNS}遍。✅={RUNS}遍都判对；⚠️=有判对也有判错（不稳定）；❌={RUNS}遍都判错。", "",
             "| 问题 | 期望 | 各遍结果 | 结论 | 说明 |", "|---|---|---|---|---|"]
    correct = total = 0
    for q, expected, note in CASES:
        got = []
        for _ in range(RUNS):
            r = route_query(q, client)
            if r is None:
                got.append("(失败)")
            elif r["route"] != "local_kb":
                got.append(f"route={r['route']}")
            else:
                got.append(r["kb_scope"])
        n_ok = sum(g == expected for g in got)
        correct += n_ok
        total += RUNS
        mark = "✅" if n_ok == RUNS else ("❌" if n_ok == 0 else "⚠️")
        print(f"{mark} {q} 期望{expected} 实际{got}")
        lines.append(f"| {q} | {expected} | {' / '.join(got)} | {mark} | {note} |")
    lines.insert(6, f"**{correct}/{total} 次判对**\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{correct}/{total} 次判对。报告已写到：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
