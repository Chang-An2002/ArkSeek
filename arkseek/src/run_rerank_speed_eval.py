# -*- coding: utf-8 -*-
"""
精排提速的准确率对比（2026-09-30，回复慢）

分步计时（run_latency_profile.py）发现本地检索里最慢的是精排：bge-reranker-v2-m3 在CPU上
每条候选约0.5秒，一次15~27条候选要8~17秒；命中同义说法表时原问题和扩展问法各精排一遍，
翻倍。这里比较几种做法，同一批问题、只测检索（不调LLM）：

  改动前     ：原问题和每个扩展问法各精排一遍，候选池 20 条/问法
  B          ：扩展问法只参与召回，精排合成一个问法打一遍分，候选池 20 条
  B+池12     ：再把交给精排的候选砍到 RRF 融合排名前 12 条
  B+每路6条  ：每一路召回（BM25/向量）只取前 6 条（唯一候选最多 12 条）

问题：用不同说法问课题的35题（带"正确章节"检查点）+ 查询改写那套15题（含3道
不该过门槛的诱导题和2道无关题）+ 知识库扩充里走本地检索的文献/JC题。
每题记：第一次是否过门槛、最高精排分、前3条是否找对（有检查点的题）、前3条跟
"改动前"是否一样、精排耗时。

用法：cd src && python run_rerank_speed_eval.py
报告写到 ../rerank_speed_eval_results.md
"""

import os
import time
from datetime import datetime

from pipeline import MemorySystemPipeline
from timing import Trace
import run_project_phrasing_demo as phrasing

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "rerank_speed_eval_results.md")
PROJECT = ["课题报告"]

# (组, 问题, 检索范围, 检查点短语 或 None, 期望过门槛 True/False/None)
QUESTIONS = []
for q, _, markers in phrasing.CASES:
    QUESTIONS.append(("课题·原15题", q, PROJECT, markers, True))
for q, _, markers in phrasing.HELDOUT:
    QUESTIONS.append(("课题·留出集1", q, None, markers, None))
for q, _, markers in phrasing.HELDOUT2:
    QUESTIONS.append(("课题·留出集2", q, None, markers, True))
for q, note in phrasing.NEGATIVE:
    QUESTIONS.append(("反例", q, None, None, "过不了" not in note))
for q, scope, expect in [
    ("变异检测常用什么工具？", None, None), ("多组学数据整合有哪些方法？", None, None),
    ("今天晚饭吃什么？", None, False), ("米哈游的游戏运营岗位需要什么能力？", None, False),
    ("这个课题是和哪个课题组合作的？", PROJECT, True),
    ("lineage tracing课题使用的是什么动物的什么组织？", PROJECT, None),
    ("这项研究用CRISPR条形码系统进行谱系标记的动物模型是什么？", None, None),
    ("这项研究的CRISPR条形码系统对什么组织的细胞进行谱系标记？", None, None),
    ("这个课题的scRNA-seq数据用什么软件做alignment？", PROJECT, None),
    ("这个project和哪个lab合作的？", PROJECT, None),
    ("这个课题的lineage barcode是用什么系统做的？", PROJECT, None),
    ("CRISPR screening的hit一般怎么筛选？", None, False),
    ("肝癌的targeted therapy有哪些药物？", None, False),
    ("What GPU is best for deep learning?", None, False),
    ("spatial transcriptomics和scRNA-seq哪个分辨率更高？", None, False),
]:
    QUESTIONS.append(("查询改写那套", q, scope, None, expect))
for q in ["长安2025年7月5日那次JC讲的是什么？", "25.11.29那次JC讲了哪篇文章，主要结论是什么？",
          "LoxCode是怎么产生高多样性条形码的？", "What problem does CoSpar solve?",
          "DARLIN小鼠和CARLIN相比改进了什么？", "SPACE-seq结合了哪两类技术？",
          "基于DNA甲基化的谱系追踪用的是什么测序方法？", "作者在JC上是怎么给大家介绍ArchR的？",
          "Why does DARLIN fuse TdT to Cas9?", "LoxCode那篇文章的作者是谁？"]:
    QUESTIONS.append(("文献/JC", q, None, None, True))

CONFIGS = [  # (名称, 合并扩展问法, 每路召回条数, 精排候选池/问法)
    ("改动前", False, 10, 20),
    ("B", True, 10, 20),
    ("B+池12", True, 10, 12),
    ("B+每路6条", True, 6, 12),
]


def run_config(kb, merge, per_channel, pool):
    kb.MERGE_EXPANSIONS_FOR_RERANK = merge
    kb.top_k_per_channel, kb.rerank_pool_size = per_channel, pool
    out = []
    for grp, q, scope, markers, _ in QUESTIONS:
        trace = Trace(q)
        t0 = time.perf_counter()
        passages, passed = trace.run(kb.retrieve_with_gate, q, 3, None, scope)
        total = time.perf_counter() - t0
        rr = sum(s[1] for s in trace.spans if s[2] == "精排")
        n_cand = sum(int(s[3].split("条")[0]) for s in trace.spans if s[2] == "精排")
        top = max((p.rerank_score or 0) for p in passages) if passages else 0.0
        hit = None if markers is None else any(
            m in p.chunk.text.replace("\n", "") for p in passages for m in markers)
        out.append(dict(passed=passed, top=top, hit=hit, ids=[p.chunk.chunk_id for p in passages],
                        rr=rr, n=n_cand, total=total))
    return out


def main():
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    kb.retrieve("预热", top_n=3)
    results = {}
    for name, merge, per_channel, pool in CONFIGS:
        print(f"== {name} ==")
        results[name] = run_config(kb, merge, per_channel, pool)
    base = results["改动前"]

    lines = ["# 精排提速的准确率对比", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"{len(QUESTIONS)}道题，只测检索（不调LLM）。门槛：精排分≥{kb.RERANK_FLOOR}。"
             "“过门槛符合期望”只统计有明确期望的题；“前3条跟改动前一样”比的是片段集合。", "",
             "## 汇总", "",
             "| 做法 | 过门槛（全部） | 过门槛符合期望 | 前3条找对（有检查点的） | 前3条跟改动前一样 | 精排总耗时 | 精排候选总条数 | 检索总耗时 |",
             "|---|---|---|---|---|---|---|---|"]
    expect_idx = [i for i, x in enumerate(QUESTIONS) if x[4] is not None]
    hit_idx = [i for i, x in enumerate(QUESTIONS) if x[3] is not None]
    for name, *_ in CONFIGS:
        r = results[name]
        lines.append(
            f"| {name} | {sum(x['passed'] for x in r)}/{len(r)} | "
            f"{sum(r[i]['passed'] == QUESTIONS[i][4] for i in expect_idx)}/{len(expect_idx)} | "
            f"{sum(bool(r[i]['hit']) for i in hit_idx)}/{len(hit_idx)} | "
            f"{sum(set(r[i]['ids']) == set(base[i]['ids']) for i in range(len(r)))}/{len(r)} | "
            f"{sum(x['rr'] for x in r):.0f}s | {sum(x['n'] for x in r)} | {sum(x['total'] for x in r):.0f}s |")
    lines += ["", "## 逐题（过门槛 / 最高分 / 找对 / 前3条同改动前 / 精排秒数）", "",
              "| 组 | 问题 | 期望过门槛 | " + " | ".join(n for n, *_ in CONFIGS) + " |",
              "|---|---|---|" + "---|" * len(CONFIGS)]
    for i, (grp, q, _, markers, expect) in enumerate(QUESTIONS):
        cells = []
        for name, *_ in CONFIGS:
            x = results[name][i]
            hit = "" if x["hit"] is None else ("找对" if x["hit"] else "没找对")
            same = "" if name == "改动前" else ("同" if set(x["ids"]) == set(base[i]["ids"]) else "变")
            cells.append(f"{'✅' if x['passed'] else '❌'} {x['top']:.3f} {hit} {same} {x['rr']:.1f}s".strip())
        exp = "" if expect is None else ("应过" if expect else "不该过")
        lines.append(f"| {grp} | {q} | {exp} | " + " | ".join(cells) + " |")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[7:7 + len(CONFIGS) + 2]))
    print(f"报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
