# -*- coding: utf-8 -*-
"""
批量测试脚本 —— 解决"每次测试都要手动复制终端输出发给Claude"的效率问题

用法：
    cd src
    python run_tests.py

会读取同目录下的 test_cases.json（可以自己加/改测试问题，不用改代码），
把每个问题的检索结果（含 BM25/向量/RRF 原始分数、置信度判断结果）写成一份
Markdown 报告，存到项目根目录的 test_results.md。

跑完之后不用再截图/复制粘贴终端输出——Claude 可以直接读取 test_results.md
这个文件（通过设备连接），报告里已经包含诊断需要的全部原始分数和片段内容。

【关于 expect_keywords 的说明】
一开始这个脚本的"通过/不通过"只检查一件事：`answer()` 到底是"给出了结果"还是
"拒答了"（对应 test_cases.json 里的 expect 字段）。后来发现这个检查太粗——有两条
真实案例，gate 判定成"可靠、给出结果"，但返回的候选片段里其实根本没有回答问题
真正需要的关键信息（比如问"和哪个课题组合作"，判定通过了，但候选片段里没有提到
真正的合作实验室名字）。也就是说，"格式上通过"不等于"内容真的答对了"。

所以加了 expect_keywords 字段：给一条测试用例指定"回答里必须出现哪些关键词"，
脚本会去检查返回给用户的候选片段里是不是真的包含这些词。不是每条用例都需要填
这个字段——只有能明确说出"标准答案应该提到什么"的问题才填，留空表示不做这项检查
（比如"应该拒答"的问题、或者没有唯一标准答案的开放性问题）。
"""

import json
import os
from datetime import datetime

from pipeline import MemorySystemPipeline


def load_test_cases(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def locate_answer_in_pool(pipeline: MemorySystemPipeline, query: str, keywords):
    """内容检查没通过时用的诊断：标准答案到底是'压根没被召回进候选池'，
    还是'召回进来了、但精排/排序把它排到了后面'——这两种情况的排查方向完全
    不同（前者要查BM25/向量召回，后者要查精排打分/阈值），之前只打印
    top_n_detail=5条时区分不出来，这里直接把整个候选池（rerank_pool_size条）
    都扫一遍，找第一条同时包含所有关键词的片段，报告它在池子里排第几、
    分数是多少；找不到就直接说明"不在候选池中"。
    """
    full_pool = pipeline.retrieve(query, top_n=pipeline.rerank_pool_size)
    for rank, p in enumerate(full_pool, start=1):
        if all(kw in p.chunk.text for kw in keywords):
            rerank_display = f"精排={p.rerank_score:.3f}" if p.rerank_score is not None else "无精排分数"
            # matched_query 跟原始query不一样，说明这个分数是靠改写问法打出来
            # 的（见pipeline.retrieve()的rewrites参数）；run_tests.py本身不传
            # 改写，这里正常不会出现，保留是为了以后复用这个诊断函数
            query_note = ""
            if p.matched_query and p.matched_query != query:
                query_note = f"，命中改写问法《{p.matched_query}》"
            return f"在候选池中排第{rank}名（共{len(full_pool)}条候选，{rerank_display}，BM25={p.bm25_score:.2f}，向量={p.dense_score:.3f}{query_note}）"
    return f"不在候选池的{len(full_pool)}条候选中——说明BM25+向量召回阶段就没选中它，精排根本看不到，问题在召回而不是排序"


def run(pipeline: MemorySystemPipeline, cases, top_n_detail: int = 5, top_n_answer: int = 3):
    results = []
    for case in cases:
        query = case["query"]
        expect = case.get("expect", "")
        expect_keywords = case.get("expect_keywords", [])
        note = case.get("note", "")

        # top_n_detail 故意比 answer() 实际使用的 top_n_answer 大一些，
        # 这样即使某条正确片段被置信度截断卡在门外，报告里也能看到它排第几名、
        # 分数是多少，方便诊断"是没召回到"还是"召回到了但被截断/判定不可靠"。
        passages = pipeline.retrieve(query, top_n=top_n_detail)
        answer_text = pipeline.answer(query, top_n=top_n_answer)

        rejected = answer_text.startswith("【未找到可靠信息】")
        actual = "should_reject" if rejected else "should_answer"
        gate_passed = (not expect) or (actual == expect)

        # 内容检查：只看真正返回给用户的那几条片段（跟 answer() 用的是同一个
        # top_n_answer，不是诊断用的 top_n_detail），检查 expect_keywords
        # 是不是都出现在里面。没填 expect_keywords 就跳过这项检查（记 None，
        # 而不是 True/False，方便报告里区分"没检查"和"检查过了"）。
        shown_text = "".join(p.chunk.text for p in passages[:top_n_answer])
        if expect_keywords:
            missing = [kw for kw in expect_keywords if kw not in shown_text]
            content_passed = (len(missing) == 0)
        else:
            missing = []
            content_passed = None  # 未设置检查项

        passed = gate_passed and (content_passed is not False)

        # 内容检查没通过时，自动去整个候选池里定位标准答案，省得每次都要手动
        # 排查"到底是召回问题还是排序/阈值问题"。
        pool_diagnosis = None
        if content_passed is False:
            pool_diagnosis = locate_answer_in_pool(pipeline, query, expect_keywords)

        results.append({
            "query": query,
            "expect": expect,
            "actual": actual,
            "gate_passed": gate_passed,
            "expect_keywords": expect_keywords,
            "missing_keywords": missing,
            "content_passed": content_passed,
            "pool_diagnosis": pool_diagnosis,
            "passed": passed,
            "note": note,
            "passages": passages,
            "answer_text": answer_text,
        })
    return results


def render_markdown(results, dense_backend_name: str, reranker_name: str,
                     reranker_is_real: bool, kb_size: int) -> str:
    lines = []
    lines.append("# 记忆系统检索测试报告\n")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"当前向量检索后端：**{dense_backend_name}**\n")
    lines.append(f"当前精排后端：**{reranker_name}**\n")
    gate_desc = "精排分数>=0.5" if reranker_is_real else "精排不可用，退回BM25>=5兜底"
    lines.append(f"当前置信度判断依据：**{gate_desc}**\n")
    lines.append(f"知识库片段数：{kb_size}\n")

    n_pass = sum(1 for r in results if r["passed"])
    lines.append(f"\n## 汇总：{n_pass}/{len(results)} 条通过预期\n")
    lines.append("（"
                  "结果=可靠性判断+内容关键词检查都通过才算✅；"
                  "内容检查列里 \"-\" 表示这条用例没设 expect_keywords，不检查内容"
                  "）\n")
    lines.append("| 问题 | 预期 | 实际 | 内容检查 | 结果 |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        if r["content_passed"] is None:
            content_col = "-"
        elif r["content_passed"]:
            content_col = "✅ 关键词齐全"
        else:
            content_col = f"❌ 缺少：{'、'.join(r['missing_keywords'])}"
        mark = "✅" if r["passed"] else "❌"
        lines.append(f"| {r['query']} | {r['expect'] or '(未设预期)'} | {r['actual']} | {content_col} | {mark} |")

    lines.append("\n## 逐条详情\n")
    for i, r in enumerate(results, start=1):
        lines.append(f"### {i}. {r['query']}")
        lines.append(f"- 备注：{r['note']}")
        lines.append(f"- 预期：{r['expect'] or '(未设预期)'}　实际：{r['actual']}　可靠性判断：{'通过' if r['gate_passed'] else '不通过'}")
        if r["expect_keywords"]:
            kw_status = "全部命中" if r["content_passed"] else f"缺少 {'、'.join(r['missing_keywords'])}"
            lines.append(f"- 内容关键词检查（应包含：{'、'.join(r['expect_keywords'])}）：{kw_status}")
            if r["pool_diagnosis"]:
                lines.append(f"- 标准答案定位诊断：{r['pool_diagnosis']}")
        lines.append(f"- 综合结果：{'通过' if r['passed'] else '不通过'}")
        lines.append(f"- `answer()` 的输出：")
        lines.append("  ```")
        for line in r["answer_text"].splitlines():
            lines.append(f"  {line}")
        lines.append("  ```")
        lines.append(f"- 完整候选片段（`retrieve()` 返回前{len(r['passages'])}条——精排可用时是精排后的顺序，"
                      f"不可用时是RRF融合排序，用来诊断'没召回到'还是'召回到了但被截断/判定不可靠'）：")
        for j, p in enumerate(r["passages"], start=1):
            snippet = p.chunk.text.replace("\n", " ")[:80]
            rerank_display = f"　精排={p.rerank_score:.3f}" if p.rerank_score is not None else ""
            # matched_query 跟原始问题不一样时，说明查询拆解在这条候选上起了作用
            query_display = ""
            if p.matched_query and p.matched_query != r["query"]:
                query_display = f"　[命中子问题：{p.matched_query}]"
            lines.append(
                f"  {j}. 来源={p.chunk.source_file}　"
                f"RRF={p.rrf_score:.4f}　BM25={p.bm25_score:.2f}　向量={p.dense_score:.3f}{rerank_display}{query_display}"
            )
            lines.append(f"     > {snippet}...")
        lines.append("")

    return "\n".join(lines)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    cases_path = os.path.join(here, "test_cases.json")
    out_path = os.path.join(here, "..", "test_results.md")

    print("正在加载知识库和检索模型……")
    pipeline = MemorySystemPipeline(kb_dir)

    cases = load_test_cases(cases_path)
    print(f"共 {len(cases)} 条测试问题，开始跑……")
    results = run(pipeline, cases)

    report = render_markdown(
        results, pipeline.dense.name, pipeline.reranker.name,
        pipeline.reranker.is_real, len(pipeline.chunks),
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    n_pass = sum(1 for r in results if r["passed"])
    print(f"\n完成：{n_pass}/{len(results)} 条通过预期。")
    print(f"完整报告已写到：{os.path.abspath(out_path)}")
    print("不用再复制粘贴终端输出了，把这个文件同步到坚果云之后告诉Claude看这个文件就行。")


if __name__ == "__main__":
    main()
