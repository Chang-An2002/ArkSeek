# -*- coding: utf-8 -*-
"""
L1画像"合并还是覆盖"的边界稳定性检查（2026-09-28，用户要求）

背景：2.2的L1由LLM判断新说法跟画像里已有信息的关系——补充细节就合并（"准确地说，
我在做肝脏纤维化"→"通过单细胞技术研究肝脏纤维化"），明确否定才覆盖（"不是单细胞，
我做的其实是肝脏纤维化"→"肝脏纤维化"）。这两种prompt里有例子；用户想知道说不清是
换方向还是补充的说法（"我现在做肝脏纤维化了"）会怎么判、稳不稳。

做法：画像固定为 {"研究方向": "单细胞"}，每句话调 llm_l1_extract()（2026-09-29改版前是llm_l1_turn()）问3遍，看"研究方向"
更新成什么：同时含"单细胞"和"纤维化"=合并，只含"纤维化"=覆盖。期望是"-"的是
本身就有歧义的说法，不判对错，只看分布和是否前后一致。

用法：
    cd src
    python run_l1_merge_boundary_demo.py
"""

import os
from collections import Counter
from datetime import datetime

from llm_client import build_llm_client
from llm_l1 import llm_l1_extract

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "l1_merge_boundary_demo_results.md")
RUNS = 3
PROFILE = {"研究方向": "单细胞"}

# (消息, 期望：合并/覆盖/-（本身有歧义）, 说明)
CASES = [
    ("准确地说，我在做肝脏纤维化方向的研究", "合并", "对照：prompt里的补充例子"),
    ("不是单细胞，我做的其实是肝脏纤维化方向", "覆盖", "对照：prompt里的否定例子"),
    ("具体是用单细胞测序研究肝脏纤维化", "合并", "两个信息都在句子里"),
    ("我也在做肝脏纤维化", "合并", "'也'=在原有基础上增加"),
    ("单细胞只是手段，我关注的是肝脏纤维化", "合并", "单细胞没被否定，只是降为手段"),
    ("我最近转去做肝脏纤维化了", "覆盖", "'转去'=换方向"),
    ("我换方向了，现在做肝脏纤维化", "覆盖", "明说换方向"),
    ("我现在做肝脏纤维化了", "-", "有歧义：'现在…了'可能是换方向，也可能是细化"),
    ("我的研究方向是肝脏纤维化", "-", "有歧义：重新陈述，没说跟单细胞什么关系"),
    ("我主要做肝脏纤维化", "-", "有歧义：'主要'可能是细化重点，也可能是纠正"),
]


def classify(result):
    if result is None:
        return "调用失败"
    values = [v for k, v in result if k == "研究方向"]
    if not values:
        return f"没写研究方向：{result}"
    v = values[-1]
    if "纤维化" in v and "单细胞" in v:
        return "合并"
    if "纤维化" in v:
        return "覆盖"
    return f"其它：{v}"


def main():
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能跑这个验证。")
        return
    lines = ["# L1画像：合并还是覆盖的边界稳定性", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"画像固定为 `{PROFILE}`，每句话问{RUNS}遍。✅={RUNS}遍都符合期望；⚠️=不稳定；❌=都不符合；"
             "期望为\"-\"的是本身有歧义的说法，只看分布。", "",
             "| 消息 | 期望 | 各遍结果（研究方向更新成） | 结论 | 说明 |", "|---|---|---|---|---|"]
    ok_runs = total = 0
    for msg, exp, note in CASES:
        got, values = [], []
        for _ in range(RUNS):
            r = llm_l1_extract(msg, dict(PROFILE), client)
            got.append(classify(r))
            values.append("；".join(f"{k}={v}" for k, v in (r or [])) or "-")
        if exp == "-":
            dist = Counter(got)
            mark = "一致" if len(dist) == 1 else "不一致"
        else:
            n = sum(g == exp for g in got)
            ok_runs += n
            total += RUNS
            mark = "✅" if n == RUNS else ("❌" if n == 0 else "⚠️")
        cells = " / ".join(f"{g}（{v}）" for g, v in zip(got, values))
        print(f"{mark} {msg} 期望{exp} -> {cells}")
        lines.append(f"| {msg} | {exp} | {cells} | {mark} | {note} |")
    lines.insert(6, f"**有明确期望的 {ok_runs}/{total} 次符合**\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n有明确期望的 {ok_runs}/{total} 次符合。报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
