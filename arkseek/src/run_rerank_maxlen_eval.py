# -*- coding: utf-8 -*-
"""
精排截断长度实验：默认长度（bge-reranker-v2-m3 的上限 8192 token）vs 截到 512（2026-09-30，回复慢）

起因：分步计时发现精排是本地检索最慢的一步。当时的猜测是"没设 max_length，按模型上限
8192 个 token 算，CPU 上很慢"，BGE 官方工具包的精排默认 max_length=512。这个脚本验证
这个猜测：同一批候选，分别用两种截断长度打分，比较耗时和分数/排序。

结论（第一次在临时脚本里跑，结果记在 README"回复慢：先量再改"的对比表里）：猜测不成立——
知识库片段本来就短（token 长度中位 362），超过 512 的只有约 5%，截断几乎不省时间，只有
候选里恰好有长片段的题快一些；前 3 条完全不变。慢的真正原因是模型本身的计算量
（5.6 亿参数在 CPU 上每条候选约 0.5 秒），所以没有采用，后来改成 OpenVINO 跑核显。

只测精排这一步，两种长度都用 PyTorch 跑在 CPU 上（跟当时的正式配置一样）。
用法：cd src && ..\\.venv\\Scripts\\python.exe run_rerank_maxlen_eval.py
报告写到 ../rerank_maxlen_eval_results.md
"""

import os
import statistics
import time
from datetime import datetime

from pipeline import MemorySystemPipeline
from reranker import CrossEncoderReranker

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "rerank_maxlen_eval_results.md")
PROJECT = ["课题报告"]
QUERIES = [("这个课题的实验设计是怎样的？", PROJECT), ("这个课题是和哪个课题组合作的？", PROJECT),
           ("LoxCode是怎么产生高多样性条形码的？", None), ("What problem does CoSpar solve?", None),
           ("DARLIN小鼠和CARLIN相比改进了什么？", None), ("这些困难打算怎么解决？", PROJECT)]


def main():
    os.environ["ARKSEEK_RERANKER"] = "torch"  # 这个实验比较的是 PyTorch 下的两种截断长度
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    default = CrossEncoderReranker()
    short = CrossEncoderReranker()
    from sentence_transformers import CrossEncoder
    short.model = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    tok = short.model.tokenizer
    lens = [len(tok(c.text)["input_ids"]) for c in kb.chunks]
    for rr in (default, short):  # 预热
        rr.rerank("预热", kb.retrieve("预热", top_n=5), top_n=5)

    lines = ["# 精排截断长度：默认（8192）vs 512", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"默认 max_length：{default.model.max_length}；知识库片段 token 长度：中位 {statistics.median(lens):.0f}、"
             f"最长 {max(lens)}、超过 512 的 {sum(l > 512 for l in lens)}/{len(lens)}。", "",
             "同一批候选（整个候选池）分别用两种长度打分，PyTorch、CPU。", "",
             "| 问题 | 候选条数 | 默认耗时 | 512耗时 | 最高分（默认→512） | 前3条相同 | 单条最大分差 |",
             "|---|---|---|---|---|---|---|"]
    for q, scope in QUERIES:
        kb.reranker = default
        pool = kb.retrieve(q, top_n=100, doc_types=scope)
        res = {}
        for name, rr in (("默认", default), ("512", short)):
            t0 = time.perf_counter()
            rr.rerank(q, pool, top_n=len(pool))
            res[name] = (time.perf_counter() - t0, {p.chunk.chunk_id: p.rerank_score for p in pool})
        a, b = res["默认"][1], res["512"][1]
        top = lambda s: sorted(s, key=s.get, reverse=True)[:3]
        lines.append(f"| {q} | {len(pool)} | {res['默认'][0]:.1f}s | {res['512'][0]:.1f}s | "
                     f"{max(a.values()):.3f}→{max(b.values()):.3f} | {'是' if top(a) == top(b) else '否'} | "
                     f"{max(abs(a[k] - b[k]) for k in a):.3f} |")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
