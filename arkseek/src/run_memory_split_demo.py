# -*- coding: utf-8 -*-
"""
"记忆写入和回答分开"的真机验证（2026-09-29，用户选的方案B）

改版前L1是三选一分类器，陈述句/画像问题会当场截住整条消息；真机发现"说自己的
信息+顺带一个问题"3/3只回了"已记住"、问题没人回答。改版后L1只抽取要记的信息，
每条消息都走回答流程，画像作为背景交给路由（"我是谁"由路由读着画像直接回答），
写入成功时由代码在末尾加"（已记下：…）"。详见llm_l1.py模块开头。

验收标准是"答对"，不看走了哪条路（用户2026-09-29）。每题一个新会话（:memory:
数据库，不碰data/memory.db），需要历史的先预置几轮对话。联网搜索换成不联网的
替身（不消耗搜索额度；客观问题会退到大模型自身知识并标注）。

用法：cd src && python run_memory_split_demo.py        （全部）
      python run_memory_split_demo.py 12 3    （只重跑第12、3题，报告写到 *_rerun.md）
"""

import os
import sys
import time
from datetime import datetime

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession, _HISTORY_MISS_REPLY
from llm_client import build_llm_client

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "memory_split_demo_results.md")


class _NoWeb:
    is_real = True
    name = "no-web"

    def search(self, query, max_results=5):
        return None


def _body(reply):
    """去掉末尾的"（已记下：…）"，剩下的是回答本身。"""
    return reply.split("\n\n（已记下：")[0].strip()


def _honest_miss(reply):
    body = _body(reply)
    return _HISTORY_MISS_REPLY in body or any(w in body for w in ("没翻到", "没有找到", "没找到", "没能找到", "没有记录", "没提到", "没有提到", "不太清楚"))


# (分组, 说明, 画像, 预置历史[(role, text)], 消息, 检查函数(reply, profile_after) -> bool, 期望说明)
CASES = [
    ("混合句", "真机发现的原句①：换方向+提问", {"研究方向": "单细胞"}, [],
     "我现在改做肝纤维化了，这个方向有什么常用的分析方法？",
     lambda r, p: "纤维化" in p.get("研究方向", "") and "单细胞" not in p.get("研究方向", "")
     and "已记下" in r and len(_body(r)) > 40,
     "研究方向改为肝纤维化（覆盖）；末尾有已记下；前面有实际回答"),
    ("混合句", "真机发现的原句②：自我介绍+wiki问题", {}, [],
     "我叫小周，能天使是什么职业？",
     lambda r, p: p.get("称呼") == "小周" and "狙击" in _body(r) and "已记下" in r,
     "记下称呼小周；回答里有能天使的职业（狙击）"),
    ("混合句", "真机发现的原句③：身份+查资料请求（非主人）", {}, [],
     "我是做单细胞的，你能帮我查查开题报告里用了什么测序平台吗",
     lambda r, p: any("单细胞" in v for v in p.values()) and len(_body(r)) > 10 and "已记下" in r
     and "长安" in _body(r).split("\n引用来源：")[0],
     "记下单细胞；请求部分有回应（不是只回已记下）；正文点明是长安的开题报告（2026-09-29用户要求，不能只写在引用来源里）"),
    ("纯陈述", "自我介绍", {}, [],
     "我叫小周",
     lambda r, p: p.get("称呼") == "小周" and r.endswith("（已记下：称呼：小周）"),
     "自然回应+（已记下：称呼：小周）"),
    ("纯陈述", "团队决定不算个人信息", {}, [],
     "我们组决定用Chroma做向量数据库",
     lambda r, p: not p and "已记下" not in r,
     "不写画像、不加已记下（聊天记录里照样存着）"),
    ("画像问题", "我是谁", {"称呼": "长安", "研究方向": "单细胞谱系追踪"}, [],
     "我是谁",
     lambda r, p: "长安" in r and "已记下" not in r,
     "用画像回答出长安"),
    ("画像问题", "我的研究方向", {"称呼": "长安", "研究方向": "单细胞谱系追踪"}, [],
     "我的研究方向是什么",
     lambda r, p: "谱系追踪" in r,
     "用画像回答出单细胞谱系追踪"),
    ("画像问题", "带问候", {"称呼": "长安"}, [],
     "你好，还记得我是谁吗",
     lambda r, p: "长安" in r,
     "回应问候+答出长安"),
    ("翻聊天记录", "之前说过今晚的计划（画像里只有无关的兴趣爱好）", {"兴趣爱好": "游泳"},
     [("user", "今晚我打算去吃KFC"), ("assistant", "好呀，炸鸡配可乐～")],
     "我今晚打算吃KFC吗",
     lambda r, p: "KFC" in _body(r) and "游泳" not in _body(r) and not _honest_miss(r),
     "从聊天记录答出打算吃KFC，不拿游泳硬套"),
    ("翻聊天记录", "没说过今晚的计划", {"兴趣爱好": "游泳"}, [],
     "我今晚打算吃KFC吗",
     lambda r, p: _honest_miss(r) and "游泳" not in _body(r),
     "老实说没翻到，不编"),
    ("翻聊天记录", "之前聊过导师推荐的文献和阅读进度", {},
     [("user", "导师推荐我先读那篇谱系追踪的综述，叫《Lineage tracing: a history》"),
      ("assistant", "好的，这篇是经典综述～"),
      ("user", "那篇综述我已经读完一半了"), ("assistant", "不错，继续加油！")],
     "我导师推荐我读的是哪篇文献，我读完了吗",
     lambda r, p: "Lineage tracing" in _body(r) and "一半" in _body(r),
     "答出文献名，并说明读了一半"),
    ("翻聊天记录", "没聊过导师推荐的文献", {}, [],
     "我导师推荐我读的是哪篇文献，我读完了吗",
     lambda r, p: _honest_miss(r) or "没有" in _body(r),
     "老实说没翻到，不编"),
]


def main():
    llm = build_llm_client()
    if not llm.is_real:
        print("需要真实LLM才能跑这个验证。")
        return
    calls = {"n": 0, "who": []}
    orig = llm.chat

    def counting(messages, **kw):
        calls["n"] += 1
        calls["who"].append(kw.get("caller", ""))
        return orig(messages, **kw)
    llm.chat = counting

    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    lines = ["# 记忆写入和回答分开：真机验证", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             "验收标准是答对（不看走了哪条路）。每题一个新会话，需要历史的先预置几轮对话；"
             "联网搜索用不联网的替身。", ""]
    only = {int(a) for a in sys.argv[1:]}
    report = REPORT.replace(".md", "_rerun.md") if only else REPORT
    ok_n = n_run = 0
    for i, (group, note, profile, history, msg, check, expect) in enumerate(CASES, start=1):
        if only and i not in only:
            continue
        n_run += 1
        session = ConversationSession(kb, MemoryStore(":memory:"), user_id=f"split_demo_{i}", llm_client=llm)
        session.web_search = _NoWeb()
        for k, v in profile.items():
            session.store.upsert_profile_fact(session.user_id, k, v)
        for role, text in history:
            session.store.append_turn(session.user_id, role, text)
        calls["n"], calls["who"] = 0, []
        t0 = time.time()
        reply = session.user_says(msg)
        dt = time.time() - t0
        after = session.store.get_profile(session.user_id)
        ok = bool(check(reply, after))
        ok_n += ok
        mark = "✅" if ok else "❌"
        print(f"{mark} [{group}] {msg}  ({dt:.1f}秒, LLM{calls['n']}次)\n{reply}\n")
        lines += [f"## {i}. [{group}] {msg} {mark}", f"- 说明：{note}", f"- 期望：{expect}",
                  f"- 画像：{profile or '空'} → {after or '空'}；预置历史{len(history)}条",
                  f"- 耗时 {dt:.1f} 秒，LLM调用 {calls['n']} 次：{'、'.join(calls['who'])}", "- 回复：", ""]
        lines += ["> " + ln for ln in reply.splitlines()] + [""]
    lines.insert(6, f"**{ok_n}/{n_run} 通过**" + (f"（只重跑了第{sorted(only)}题）" if only else "") + "\n")
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{ok_n}/{n_run} 通过。报告：{os.path.abspath(report)}")


if __name__ == "__main__":
    main()
