# -*- coding: utf-8 -*-
"""
回复慢：分步计时（2026-09-30，先量再改）

用固定的几类问题各跑一遍真实流程，记下每一步花了多少时间（timing.py），
汇总成一张表，看时间到底花在哪里：LLM调用（几次、每次多久、有没有排队）、
本地检索的召回和精排、wiki请求、联网搜索、字数压缩……

每题一个新会话（:memory: 数据库，不碰 data/memory.db），需要历史的先预置几轮。
联网搜索默认换成不联网的替身（不消耗搜索额度，客观问题会退到大模型自身知识）；
加 --real-search 用真实搜索接口，把搜索本身的耗时也量进来。
正式计时前先空跑一次检索，把模型第一次推理的预热时间排除掉。

用法：cd src && python run_latency_profile.py [题号 ...] [--real-search] [--tag 名字]
报告写到 ../latency_profile_results[_名字].md
"""

import os
import sys
import time
from datetime import datetime

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession
from llm_client import build_llm_client
from timing import Trace

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
REPORT = os.path.join(HERE, "..", f"latency_profile_results{'_' + TAG if TAG else ''}.md")


class _NoWeb:
    is_real = True
    name = "no-web"

    def search(self, query, max_results=5):
        return None


# (类别, 消息, 预置历史[(role, text)])
CASES = [
    ("闲聊", "你好呀，今天有点累", []),
    ("wiki", "能天使是什么职业？", []),
    ("课题（命中同义表）", "这个课题的实验设计是怎样的？", []),
    ("课题（不命中同义表）", "这个课题是和哪个课题组合作的？", []),
    ("文献", "LoxCode是怎么产生高多样性条形码的？", []),
    ("翻聊天记录", "我们组之前决定用什么做向量数据库来着？",
     [("user", "我们组决定用Chroma做向量数据库，因为轻量、Python里直接能用"), ("assistant", "好的，记下了～")]),
    ("拆成子问题", "能天使和德克萨斯的天赋分别是什么？", []),
    ("知识库外的客观问题", "AlphaFold3的原理是什么？", []),
    ("字数要求", "介绍一下这个课题的开题报告，100字以内", []),
]


def _sum(trace, kind):
    spans = [s for s in trace.spans if s[2] == kind]
    return sum(s[1] for s in spans), len(spans)


def main():
    llm = build_llm_client()
    if not llm.is_real:
        print("需要真实LLM才能跑这个计时。")
        return
    real_search = "--real-search" in sys.argv
    only = {int(a) for a in sys.argv[1:] if a.isdigit()}
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    kb.retrieve("预热", top_n=3)  # 模型第一次推理有额外开销，不算进正式计时

    lines = ["# 回复慢：分步计时", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"每题一个新会话、跑一遍真实流程；联网搜索：{'真实接口' if real_search else '不联网的替身'}。"
             "“开始”是相对这条消息开始过了多少秒；并行的步骤时间段会重叠，所以各步相加会超过总耗时。", ""]
    summary = ["| # | 类别 | 问题 | 总耗时 | LLM（次数） | LLM排队 | 召回 | 精排（次数） | wiki请求 | 联网搜索 | 聊天记录检索+写索引 |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
    details = []
    for i, (kind, msg, history) in enumerate(CASES, start=1):
        if only and i not in only:
            continue
        session = ConversationSession(kb, MemoryStore(":memory:"), user_id=f"latency_{i}", llm_client=llm)
        if not real_search:
            session.web_search = _NoWeb()
        for role, text in history:
            session.store.append_turn(session.user_id, role, text)
        trace = Trace(f"#{i} [{kind}] {msg}")
        t0 = time.perf_counter()
        reply = trace.run(session.user_says, msg, True)  # 跟qq_bot一样，摘要推迟到回复之后、不计入
        total = time.perf_counter() - t0
        report = trace.emit()
        llm_t, llm_n = _sum(trace, "LLM")
        q_t, _ = _sum(trace, "LLM排队")
        rec_t, _ = _sum(trace, "召回")
        rr_t, rr_n = _sum(trace, "精排")
        wiki_t, _ = _sum(trace, "wiki请求")
        web_t, _ = _sum(trace, "联网搜索")
        hist_t = _sum(trace, "聊天记录检索")[0] + _sum(trace, "聊天记录写索引")[0]
        summary.append(f"| {i} | {kind} | {msg} | **{total:.1f}s** | {llm_t:.1f}s（{llm_n}） | {q_t:.1f}s | "
                       f"{rec_t:.1f}s | {rr_t:.1f}s（{rr_n}） | {wiki_t:.1f}s | {web_t:.1f}s | {hist_t:.1f}s |")
        body = [ln.replace("[timing] ", "", 1) for ln in report.splitlines()[1:]]
        details += [f"## {i}. [{kind}] {msg}", "", "```", *body, "```", "", "回复：", ""]
        details += ["> " + ln for ln in reply.splitlines()] + [""]
    lines += ["## 汇总", ""] + summary + [""] + details
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(summary))
    print(f"报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
