# -*- coding: utf-8 -*-
"""
课题报告的同义说法表（2026-09-29，同义说法检索改进）

问"这个课题的实验设计"查不到：报告里写的是"研究方法/技术路线/试验方案"。用户问课题时
用的是口语（"怎么做的""遇到了什么困难""时间怎么安排"），报告用的是开题/中期报告的
固定栏目名（"拟采取的研究方法""存在的问题""研究计划及预测进展"）。这是搜索里最常见的
词汇不匹配，最老、最可靠的办法是同义词表（查询扩展）：问法里出现某个说法，就另外
生成一个带上报告栏目名的问法一起检索（multi-query，精排取各问法里的最高分）。

只在问的是课题时用：检索范围限定在课题报告，或者问题里提到课题/开题/中期/项目。
免得"这篇文献有什么创新"这类问题也被扩成课题报告的"创新点"。
表是人工维护的、结果确定；以后发现新的说法对不上，往里加一行就行。
"""

import re
from typing import List, Optional, Sequence

# (问法里可能出现的说法, 报告里对应的栏目名)
SYNONYM_GROUPS = [
    (("实验设计", "实验方案", "试验设计", "技术方案", "研究设计", "怎么做", "如何做", "具体做法", "实验流程", "分析流程"),
     "拟采取的研究方法 技术路线 试验方案"),
    (("研究思路", "思路", "研究目的", "目的是"), "研究目标 研究内容"),
    (("想解决什么", "要解决什么", "解决什么问题", "科学问题"), "拟解决的关键问题"),
    (("创新", "新意", "亮点", "特色"), "研究的创新点"),
    (("产出", "成果", "发表"), "预期研究成果"),
    (("时间安排", "进度安排", "时间规划", "计划"), "研究计划及预测进展 时间阶段"),
    (("可行", "靠谱", "做得成"), "可行性分析"),
    (("缺什么", "条件", "不足"), "已具备的实验、资料等条件 尚缺少的实验、资料条件"),
    (("做到哪", "进展", "完成了什么", "做了什么", "进度如何"), "已完成的任务及取得的成果"),
    (("接下来", "下一步", "后续", "还要做", "还需要做"), "尚须完成的任务"),
    (("困难", "难点", "挑战", "遇到了什么问题"), "存在的问题"),
    (("怎么解决", "解决办法", "如何解决", "对策", "应对"), "拟采取的办法"),
    (("意义", "价值", "为什么做"), "课题的研究意义 理论意义 实践意义"),
    (("背景", "现状", "别人做过"), "国内外研究现状分析"),
]

_ABOUT_PROJECT = re.compile(r"课题|开题|中期|项目|project", re.I)
MAX_VARIANTS = 2


def merge_variants(query: str, variants: Sequence[str]) -> str:
    """几个扩展问法合成一个（"原问题（栏目A 栏目B）"），精排只用它打一遍分（2026-09-30，
    回复慢：原来原问题和每个扩展问法各精排一遍，命中同义表时精排时间翻倍）。"""
    if len(variants) == 1:
        return variants[0]
    prefix = f"{query}（"
    sections = [v[len(prefix):-1] if v.startswith(prefix) and v.endswith("）") else v for v in variants]
    return f"{query}（{' '.join(sections)}）"


def expand_for_project(query: str, doc_types: Optional[Sequence[str]] = None) -> List[str]:
    """返回需要额外检索的问法（最多MAX_VARIANTS个）；不是在问课题、或者没有命中任何说法时返回空列表。"""
    if not (doc_types and "课题报告" in doc_types) and not _ABOUT_PROJECT.search(query):
        return []
    variants = []
    for triggers, sections in SYNONYM_GROUPS:
        if any(t in query for t in triggers):
            variants.append(f"{query}（{sections}）")
        if len(variants) >= MAX_VARIANTS:
            break
    return variants
