# -*- coding: utf-8 -*-
"""
"用不同说法问课题"的检索验证（2026-09-29，同义说法检索改进）

起因：问"这个课题的实验设计"查不到长安的课题报告——报告里写的是"技术路线/试验方案/
研究方法"，没有"实验设计"这个词（词汇不匹配，vocabulary mismatch）。

这里只测检索这一步，不调用LLM（结果确定、不花额度）：每题看两件事——
①第一次检索能不能过置信度门槛（过不了就不会用长安的资料回答）；
②正确的章节在不在前3条（按该章节正文里的几个"检查点"短语判断）。
分两种范围各跑一遍：只查课题报告（路由判成"这个课题"时）、查整个知识库。

用法：cd src && python run_project_phrasing_demo.py [--tag 名字]
报告写到 ../project_phrasing_demo_results[_名字].md，改动前后各跑一次对比。
"""

import os
import sys
from datetime import datetime

from pipeline import MemorySystemPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
REPORT = os.path.join(HERE, "..", f"project_phrasing_demo_results{'_' + TAG if TAG else ''}.md")

# (问题, 对应章节, 检查点短语（前3条里出现任一即算找对）)
DESIGN = ["数据来源与预处理", "质控与条形码比对", "多组学数据整合与亚集提取", "肝胆系细胞亚群分析", "研究目的 研究方法"]
CASES = [
    ("这个课题的实验设计是怎样的？", "开题 2.2 研究方法/试验方案", DESIGN),
    ("这个课题的技术方案是什么？", "开题 2.2 研究方法/试验方案", DESIGN),
    ("这个课题具体是怎么做的？", "开题 2.2 研究方法/试验方案", DESIGN),
    ("这个课题用了哪些分析方法和软件？", "开题 2.2.1 研究方法", ["研究目的 研究方法", "CellRanger", "Monocle2"]),
    ("这个课题的研究思路是什么？", "开题 2.1 研究目标/内容", ["本研究旨在", "细胞图谱的构建与验证", "克隆谱系动态的基础分析"]),
    ("这个课题想解决什么问题？", "开题 2.1.3 关键问题", ["技术分析层面", "科学探究层面"]),
    ("这个课题有什么创新的地方？", "开题 2.3 创新点", ["首次使用创新型", "转分化的争论"]),
    ("这个课题预计能产出什么？", "开题 2.5 预期成果", ["预计发表高水平学术论文"]),
    ("这个课题的时间安排是怎样的？", "开题 2.4 研究计划", ["研究时间阶段划分", "2025.01-2025.04"]),
    ("这个课题可行吗？", "开题 2.2.4 可行性分析", ["理论可行性", "技术可行性"]),
    ("做这个课题还缺什么条件？", "开题 3.2.2 尚缺少的条件", ["尚显不足", "缺乏对当前CRISPR-Cas9编辑形式"]),
    ("这个课题目前做到哪一步了？", "中期 一、已完成的任务", ["完成课题方案细化", "联合分析框架"]),
    ("这个课题接下来还要做什么？", "中期 二、尚须完成的任务", ["需进一步扩展并均衡bulk scar参考库", "需完善肝胆系细胞状态转换的验证"]),
    ("这个课题遇到了什么困难？", "中期 三、存在的问题", ["单细胞通量有限", "参考库数量有限"]),
    ("这些困难打算怎么解决？", "中期 四、拟采取的办法", ["针对单细胞通量有限", "针对 bulk参考库"]),
]


# 留出集：同义说法表是看着上面15题写的，下面这些是之后另写的说法（不参与写表，用来看
# 表能不能泛化；比如"麻烦""啥""交出"都不在表里）
HELDOUT = [
    # 用户2026-09-30：这句本身有歧义（实验怎么设计 / 时间怎么安排），答不对是问法的问题
    ("长安这个项目的实验是怎么安排的？", "开题 2.2 试验方案（问法有歧义）", DESIGN),
    ("课题里的数据具体是按什么流程处理的？", "开题 2.2.3 试验方案", DESIGN),
    ("这个课题的目标是啥？", "开题 2.1.1 研究目标", ["本研究旨在"]),
    ("这个项目的亮点在哪？", "开题 2.3 创新点", ["首次使用创新型", "转分化的争论"]),
    ("课题最后打算交出什么？", "开题 2.5 预期成果", ["预计发表高水平学术论文"]),
    ("开题里的进度规划是什么样的？", "开题 2.4 研究计划", ["研究时间阶段划分", "2025.01-2025.04"]),
    ("中期的时候课题碰到了哪些麻烦？", "中期 三、存在的问题", ["单细胞通量有限", "参考库数量有限"]),
    ("这个课题为什么值得做？", "开题 1.2 研究意义", ["功能-谱系-状态"]),
    ("中期报告里说已经完成了哪些工作？", "中期 一、已完成的任务", ["完成课题方案细化", "联合分析框架"]),
]
# 留出集2（2026-09-30）：在"给片段生成可能被问的问题"之前写好、写完不再改，
# 写的时候没看过生成的问题；换了一批章节（数据来源、质控、克隆定义、统计软件、课题背景、
# 已有条件、中期的问题和办法）。检查点都用正文里的原话，不用章节标题
HELDOUT2 = [
    ("长安课题里用的小鼠是从哪来的？", "开题 2.2.3（1）数据来源", ["小鼠品系由"]),
    ("这个课题筛细胞的质控标准是多少？", "开题 2.2.3（2）质控", ["线粒体基因比例>15%"]),
    ("课题里的克隆是怎么定义的？", "开题 2.2.3（4）亚群分析", ["共享相同突变谱的条形码定义克隆"]),
    ("这个课题打算用什么做统计？", "开题 2.2.3（5）统计与可视化", ["所有统计分析均使用R"]),
    ("这个课题为什么选肝脏来研究？", "开题 1.1 课题来源", ["肝脏是人体胎儿时期主要的造血器官"]),
    ("做这个课题手里已经有什么基础了？", "开题 3.2.1 已具备的条件", ["已产出初步的单细胞文库数据"]),
    ("中期的时候课题的参考库有什么不足？", "中期 三、存在的问题", ["参考库数量有限"]),
    ("课题后面打算怎么增加细胞数量？", "中期 四、拟采取的办法", ["增加可分析细胞数量"]),
]
# 反例：不是在问课题。同义说法表不该触发、也不该把它们拉到课题报告上
NEGATIVE = [
    ("DARLIN有什么创新？", "文献（不是课题）"),
    ("CoSpar解决了什么问题？", "文献（不是课题）"),
    ("AlphaFold3的原理是什么？", "知识库里没有，应该过不了门槛"),
]


def _run_set(kb, cases, title, lines):
    import time
    passed_n = hit_n = 0
    lines += [f"## {title}（查整个知识库）", "", "| 问题 | 对应章节 | 第一次过门槛 | 最高精排分 | 前3条找对 | 耗时 |",
              "|---|---|---|---|---|---|"]
    for q, section, markers in cases:
        t0 = time.time()
        passages, passed = kb.retrieve_with_gate(q, top_n=3, query_rewriter=None, doc_types=None)
        dt = time.time() - t0
        top = max((p.rerank_score or 0) for p in passages) if passages else 0
        hit = any(m in p.chunk.text.replace(chr(10), "") for p in passages for m in markers)
        passed_n += passed
        hit_n += hit
        lines.append(f"| {q} | {section} | {'✅' if passed else '❌'} | {top:.3f} | {'✅' if hit else '❌'} | {dt:.1f}s |")
    lines += ["", f"**{title}：过门槛 {passed_n}/{len(cases)}，前3条找对 {hit_n}/{len(cases)}**", ""]
    return f"{title}：过门槛 {passed_n}/{len(cases)}，找对 {hit_n}/{len(cases)}"


def _run_negative(kb, lines):
    lines += ["## 反例（不是在问课题）", "", "| 问题 | 说明 | 过门槛 | 最高精排分 | 前3条来自课题报告的条数 |", "|---|---|---|---|---|"]
    for q, note in NEGATIVE:
        passages, passed = kb.retrieve_with_gate(q, top_n=3, query_rewriter=None, doc_types=None)
        top = max((p.rerank_score or 0) for p in passages) if passages else 0
        n_proj = sum(p.chunk.meta.get("doc_type") == "课题报告" for p in passages)
        lines.append(f"| {q} | {note} | {'✅' if passed else '❌'} | {top:.3f} | {n_proj} |")
    lines.append("")


def _apply_experiment_switches():
    """对比实验用的开关（只在这个脚本里生效，不改正式代码）：
    --headers long|leaf|none  课题报告片段标题：完整章节路径（正式做法）/只写最末一级/不按章节切
    --no-synonyms             关掉同义说法表"""
    import chunking
    import kb_sources
    import query_synonyms
    mode = sys.argv[sys.argv.index("--headers") + 1] if "--headers" in sys.argv else "long"
    if mode == "none":
        def old_style(text, label, meta=None, start_chunk_id=0):
            out = chunking.chunk_document(text, label, meta, start_chunk_id=start_chunk_id)
            for c in out:
                c.text = f"[{label}]\n{c.text}"
            return out
        kb_sources.chunk_by_sections = old_style
    elif mode == "leaf":
        orig = chunking.chunk_by_sections

        def leaf_only(text, label, meta=None, start_chunk_id=0):
            out = orig(text, label, meta, start_chunk_id=start_chunk_id)
            for c in out:
                head, _, body = c.text.partition("\n")
                c.text = f"【{label} › {head.strip('【】').split(' › ')[-1]}】\n{body}"
            return out
        kb_sources.chunk_by_sections = leaf_only
    if "--no-synonyms" in sys.argv:
        query_synonyms.expand_for_project = lambda query, doc_types=None: []
    return f"标题={mode}，同义词={'关' if '--no-synonyms' in sys.argv else '开'}"


def main():
    switches = _apply_experiment_switches()
    print(f"[实验开关] {switches}")
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    lines = [f"# 用不同说法问课题：检索验证{'（' + TAG + '）' if TAG else ''}", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"只测检索（不调LLM）。门槛：精排分≥{kb.RERANK_FLOOR}。“找对”=对应章节的检查点短语出现在前3条里。", ""]
    summary = []
    for scope_name, doc_types in (("只查课题报告", ["课题报告"]), ("查整个知识库", None)):
        passed_n = hit_n = 0
        lines += [f"## {scope_name}", "", "| 问题 | 对应章节 | 第一次过门槛 | 最高精排分 | 前3条找对 |", "|---|---|---|---|---|"]
        for q, section, markers in CASES:
            passages, passed = kb.retrieve_with_gate(q, top_n=3, query_rewriter=None, doc_types=doc_types)
            top = max((p.rerank_score or 0) for p in passages) if passages else 0
            hit = any(m in p.chunk.text.replace(chr(10), "") for p in passages for m in markers)
            passed_n += passed
            hit_n += hit
            lines.append(f"| {q} | {section} | {'✅' if passed else '❌'} | {top:.3f} | {'✅' if hit else '❌'} |")
        lines += ["", f"**{scope_name}：过门槛 {passed_n}/{len(CASES)}，前3条找对 {hit_n}/{len(CASES)}**", ""]
        summary.append(f"{scope_name}：过门槛 {passed_n}/{len(CASES)}，找对 {hit_n}/{len(CASES)}")
    summary.append(_run_set(kb, HELDOUT, "留出集（写同义表时没看过的说法）", lines))
    summary.append(_run_set(kb, HELDOUT2, "留出集2（生成问题之前写好）", lines))
    _run_negative(kb, lines)
    lines.insert(5, "**" + "；".join(summary) + "**\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(summary))
    print(f"报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
