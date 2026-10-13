# -*- coding: utf-8 -*-
"""
路由"主观/客观"边界（route="general" 时的 needs_search 字段）稳定性验证

用户的决策：问题按主客观区分——客观的、需要事实依据或时效性信息的走联网搜索
（答案标注"来自网络搜索"），纯主观闲聊直接回答。用户同时要求"记得验证这个
区分边界是否稳定"，这个脚本就是做这件事：每道题跑 RUNS 遍 route_query()，
看判断对不对、同一道题几遍之间是否一致。只调用路由判断，不真的去搜索。

题目分三组：
  客观（预期needs_search=True）、主观（预期False）、边界（最容易摇摆的说法，
  预期值是按"答案是事实还是口味"这条标准人工判定的，报告里单独统计）。

用法：
    cd src
    python run_search_boundary_demo.py
"""

import os
from datetime import datetime

from llm_client import build_llm_client
from wiki_router import route_query

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "search_boundary_demo_results.md")
RUNS = 3

CASES = [
    # (问题, 预期needs_search, 分组)
    ("推荐个讲明日方舟攻略的up主", True, "客观"),
    ("明日方舟哪个干员最强", True, "客观"),
    ("明日方舟最近有什么新活动", "route=wiki", "客观"),  # 正确路由是wiki首页（近期新增），不走needs_search
    ("空间转录组和单细胞测序哪个分辨率更高", True, "客观"),
    ("今天北京天气怎么样", True, "客观"),
    ("明日方舟是哪家公司做的", True, "客观"),
    ("Python里怎么读取一个json文件", True, "客观"),
    ("诺贝尔生理学或医学奖2025年颁给了谁", True, "客观"),
    ("你觉得明日方舟哪个干员最帅", False, "主观"),
    ("今天晚饭吃什么", False, "主观"),
    ("你好呀", False, "主观"),
    ("明日方舟好玩吗", False, "主观"),
    ("你是谁", False, "主观"),
    ("我今天有点累", False, "主观"),
    ("给我讲个笑话", False, "主观"),
    ("你喜欢什么颜色", False, "主观"),
    ("新手玩明日方舟应该先练哪些干员", True, "边界"),
    ("明日方舟和原神哪个更好玩", False, "边界"),
    ("周末去哪玩比较好", False, "边界"),
    ("有什么好看的科幻电影推荐", True, "边界"),
    ("学生物信息学需要先学什么", True, "边界"),
    ("熬夜对身体有什么危害", True, "边界"),
    ("你觉得我该不该读博", False, "边界"),
    ("明日方舟的剧情讲的是什么", True, "边界"),
]


def main():
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能跑这个验证。")
        return

    rows = []
    for q, expect, group in CASES:
        outcomes = []
        for _ in range(RUNS):
            r = route_query(q, client)
            if r is None:
                outcomes.append("解析失败")
            elif r["route"] != "general":
                outcomes.append(f"route={r['route']}")
            else:
                outcomes.append(f"search={r['needs_search']}")
        want = expect if isinstance(expect, str) else f"search={expect}"
        correct = sum(o == want for o in outcomes)
        stable = len(set(outcomes)) == 1
        rows.append((q, group, expect, outcomes, correct, stable))
        print(f"[{group}] {q} -> {outcomes}（预期search={expect}）")

    lines = ["# 主观/客观边界（needs_search）稳定性验证", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}　LLM：{client.name}　每题{RUNS}遍", "",
             "\"route=wiki\"之类表示这道题根本没被判成general（不走needs_search这条判断），单独看。", "",
             "| 分组 | 问题 | 预期 | 各遍结果 | 正确次数 | 稳定 |", "|---|---|---|---|---|---|"]
    for q, group, expect, outcomes, correct, stable in rows:
        lines.append(f"| {group} | {q} | search={expect} | {'、'.join(outcomes)} | {correct}/{RUNS} | "
                     f"{'✅' if stable else '⚠️摇摆'} |")
    lines.append("")
    for g in ("客观", "主观", "边界"):
        grp = [r for r in rows if r[1] == g]
        lines.append(f"- {g}：{sum(r[4] for r in grp)}/{len(grp) * RUNS} 次判断符合预期，"
                     f"{sum(r[5] for r in grp)}/{len(grp)} 道题{RUNS}遍结果一致")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    print(f"报告已写到：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
