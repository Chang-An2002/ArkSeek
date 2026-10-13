# -*- coding: utf-8 -*-
"""
L3查询改写（query_planner.rewrite_for_kb()）的真机验证脚本

对每条用例分别跑两遍 retrieve_with_gate()：
  - 基线：不传 query_rewriter（run_tests.py / 2.1 的行为）
  - 改写：传入 rewrite_for_kb()（2.2 下 _try_answer_from_local_kb() 的行为）

用例分三组：
  A. run_tests.py 原有的8条（test_cases.json）——回归，第6条是这次要救的已知局限
  B. 泛化：换几种中英混杂的问法问知识库里确实有答案的内容，检验方法本身
     有没有泛化能力，而不是只对第6条那一句话有效（prompt的few-shot例子
     刻意没用测试集原句）
  C. 误放行探针：跟课题领域相近、但知识库里并没有答案的问题——改写+取最高分
     天然会放宽gate，这组专门盯住这个风险

需要真实BGE embedding+精排模型和真实LLM（.env 里配好 key）。

用法：
    cd src
    python run_kb_rewrite_demo.py
"""

import json
import os
from datetime import datetime

from pipeline import MemorySystemPipeline
from llm_client import build_llm_client
from query_planner import rewrite_for_kb
from llm_generate import is_no_answer, llm_generate_answer
from wiki_router import route_query


HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "kb_rewrite_demo_results.md")

EXTRA_CASES = [
    # B. 泛化：知识库里有答案，问法中英混杂
    {"group": "B", "query": "这个课题的scRNA-seq数据用什么软件做alignment？", "expect": "should_answer",
     "expect_keywords": ["CellRanger"],
     "note": "英文缩写+英文术语混在中文句子里；原文是\"CellRanger软件比对至小鼠参考基因组mm10\"，要看改写能不能把alignment对上\"比对\""},
    {"group": "B", "query": "这个project和哪个lab合作的？", "expect": "should_answer",
     "expect_keywords": ["林盛达"],
     "note": "口语化中英混杂；原文是\"与生命科学研究院林盛达实验室合作\"，要看project/lab能不能对上\"课题/实验室\""},
    {"group": "B", "query": "这个课题的lineage barcode是用什么系统做的？", "expect": "should_answer",
     "expect_keywords": ["CRISPR"],
     "note": "对照：原文里本身就有\"lineage barcode\"这个英文词，预期基线直接就能过，不需要改写。"
             "2026-09-25按用户意见补上\"这个课题\"：原问法没有主语，知识库扩充后有歧义（文献里有"
             "scGESTALT/CARLIN/DARLIN等好几套系统）；长安的课题用的是基于CRISPR-Cas9的条形码系统"
             "（LSL-Cas9-tdTomato小鼠+AAV递送sgRNA），不是DARLIN"},
    # C. 误放行探针：领域相近，知识库里没有答案
    {"group": "C", "query": "CRISPR screening的hit一般怎么筛选？", "expect": "should_reject",
     "expect_keywords": [],
     "note": "原文有很多CRISPR和\"筛选\"字样，但讲的是条形码过滤，不是CRISPR筛选实验的hit判定——字面最容易撞车的一道。"
             "2026-09-25用户确认拒答正确：这本身是个不太标准的问法（学界有HIT-Cas9技术，但跟这个问法不沾边），适合当'应该拒绝'的测试"},
    {"group": "C", "query": "肝癌的targeted therapy有哪些药物？", "expect": "should_reject",
     "expect_keywords": [],
     "note": "原文大量出现\"肝脏\"，但课题是谱系追踪，不涉及肝癌用药"},
    {"group": "C", "query": "What GPU is best for deep learning?", "expect": "should_reject",
     "expect_keywords": [],
     "note": "纯英文、完全无关——检验改写会不会把无关问题\"翻译\"得跟知识库沾边"},
    {"group": "C", "query": "spatial transcriptomics和scRNA-seq哪个分辨率更高？", "expect": "should_answer",
     "expect_keywords": ["resolution"],
     "note": "原来预期拒答：两份课题报告用了scRNA-seq但没有跟空间转录组做比较。2026-09-25用户确认改成应回答——"
             "知识库扩充后SPACE-seq文献里确实讨论了Slide-seqV2约10μm的近细胞分辨率，回答有依据、有引用"},
]


def _missing(passages, keywords):
    text = "".join(p.chunk.text for p in passages)
    return [kw for kw in keywords if kw not in text]


def _judge(passed, passages, case):
    actual = "should_answer" if passed else "should_reject"
    ok = actual == case["expect"]
    if ok and passed and case.get("expect_keywords"):
        ok = not _missing(passages, case["expect_keywords"])
    return actual, ok


def main():
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    client = build_llm_client()
    if not client.is_real or not kb.reranker.is_real:
        print("需要真实LLM和真实精排模型才能跑这个验证。")
        return

    with open(os.path.join(HERE, "test_cases.json"), encoding="utf-8") as f:
        cases = [dict(c, group="A") for c in json.load(f)] + EXTRA_CASES

    rows = []
    for case in cases:
        q = case["query"]
        # 跟真实对话流程一样先问路由：问的是作者自己的课题（kb_scope="project"）
        # 时只在课题报告里检索（知识库扩充后加的元数据过滤，见pipeline.retrieve()）
        route = route_query(q, client) or {}
        scope = route.get("kb_scope", "all")
        doc_types = ["课题报告"] if scope == "project" else None
        base_p, base_pass = kb.retrieve_with_gate(q, doc_types=doc_types)
        calls = []

        def rewriter(text):
            out = rewrite_for_kb(text, client)
            calls.append(out)
            return out

        rw_p, rw_pass = kb.retrieve_with_gate(q, query_rewriter=rewriter, doc_types=doc_types)
        # 跟2.2的 _try_answer_from_local_kb() 一样：gate过了才生成，没过就是固定拒答文案。
        # 生成步骤输出NO_ANSWER（可答性判断，见llm_generate.py）时，真实系统按拒答处理、
        # 转去联网搜索——所以"改写路径"的最终判定看 gate通过 且 生成没说NO_ANSWER，
        # 不再只看gate（只看gate会把被可答性判断正确拦下的题误判成放行）。
        answer = ((llm_generate_answer(q, rw_p, client, caller_label="local_kb") or "（生成调用失败）")
                  if rw_pass else kb.REJECT_MESSAGE)
        answered = rw_pass and not is_no_answer(answer)
        b_actual, b_ok = _judge(base_pass, base_p, case)
        r_actual, r_ok = _judge(answered, rw_p, case)
        rows.append({
            **case, "b_actual": b_actual, "b_ok": b_ok,
            "b_top": max((p.rerank_score for p in base_p), default=0.0),
            "r_actual": r_actual, "r_ok": r_ok, "gate_pass": rw_pass,
            "r_top": max((p.rerank_score for p in rw_p), default=0.0),
            "rewrites": calls[0] if calls else None,
            "answer": answer, "scope": scope,
            "top_source": rw_p[0].chunk.text.replace("\n", "") if rw_p else "",
        })

    names = {"A": "A. run_tests.py原有用例（回归）", "B": "B. 泛化（中英混杂问法，知识库有答案）",
             "C": "C. 误放行探针（领域相近，知识库没有答案）"}
    lines = [
        "# L3查询改写验证报告", "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
        f"向量检索：**{kb.dense.name}**　精排：**{kb.reranker.name}**　LLM：**{client.name}**", "",
        "基线=不改写（run_tests.py/2.1行为）；改写=gate第一次没过时调用rewrite_for_kb()重新检索（2.2行为）。"
        "应答用例还要求返回的片段里包含全部预期关键词。", "",
    ]
    for g in "ABC":
        group = [r for r in rows if r["group"] == g]
        lines += [f"## {names[g]}", "",
                  "| 问题 | 预期 | 基线 | 基线最高分 | 改写 | 改写后最高分 | LLM改写出的问法 |",
                  "|---|---|---|---|---|---|---|"]
        for r in group:
            if r["rewrites"] is None:
                rw = "（gate一次就过，未调用）"
            elif not r["rewrites"]:
                rw = "（判断不需要改写）"
            else:
                rw = "<br>".join(r["rewrites"])
            lines.append(
                f"| {r['query']} | {r['expect']} | {'✅' if r['b_ok'] else '❌'} {r['b_actual']} | {r['b_top']:.3f} | "
                f"{'✅' if r['r_ok'] else '❌'} {r['r_actual']} | {r['r_top']:.3f} | {rw} |")
        lines += ["", f"基线 {sum(r['b_ok'] for r in group)}/{len(group)}，"
                      f"改写 {sum(r['r_ok'] for r in group)}/{len(group)}", ""]

    lines += ["## 逐条详情（测试目的 + 改写路径下的最终回答）", ""]
    for i, r in enumerate(rows, start=1):
        lines += [
            f"### {r['group']}{i}. {r['query']}",
            f"- 目的：{r.get('note') or '-'}",
            f"- 预期：{r['expect']}；改写路径实际：{r['r_actual']}（最高精排分{r['r_top']:.3f}）；检索范围：{'只查课题报告' if r['scope'] == 'project' else '全库'}"
            f"{' ✅' if r['r_ok'] else ' ❌'}",
            f"- 排第1的片段（开头）：{r['top_source'][:120]}…",
            "- 最终回答：",
            "",
            "  > " + r["answer"].replace("\n", "\n  > "),
            "",
        ]

    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n报告已写到：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
