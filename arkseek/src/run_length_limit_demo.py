# -*- coding: utf-8 -*-
"""
用户字数要求的真机验证（2026-09-29，见length_limit.py）

每题都带字数要求，覆盖各条回答路径：本地资料、wiki、翻聊天记录、大模型自身知识、
闲聊、拆成多个子问题；外加"至少N字"（只要求、不扩写）和英文。检查两件事：
①正文长度（上限：不超过上限的120%；一句话：只有一句；下限：只记录，不判）；
②关键事实还在（压缩不能把答案压没了）。

  python run_length_limit_demo.py             改动后的行为，报告写 length_limit_demo_results.md
  python run_length_limit_demo.py --baseline  改动前的行为（运行时去掉prompt里的字数规则、
                                              关掉生成后检查），报告写 *_baseline.md，用来对比
每题一个新会话（:memory:数据库，不碰data/memory.db）；联网搜索用不联网的替身。
"""

import os
import sys
import time
from datetime import datetime

import conversation_memory
import llm_generate
import query_planner
from conversation_memory import ConversationSession
from length_limit import count_sentences, count_units, parse_length_request, split_reply, TOLERANCE
from llm_client import build_llm_client
from memory_store import MemoryStore
from pipeline import MemorySystemPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = "--baseline" in sys.argv
ONLY = {int(a) for a in sys.argv[1:] if a.isdigit()}  # 只重跑这几题：python run_length_limit_demo.py 3 6
REPORT = os.path.join(HERE, "..", "length_limit_demo_results" + ("_baseline" if BASELINE else "")
                      + ("_rerun" if ONLY else "") + ".md")


class _NoWeb:
    is_real = True
    name = "no-web"

    def search(self, query, max_results=5):
        return None


# (说明, 预置历史, 问题, 答案里应该有（全部；元组=其中任一即可）)
CASES = [
    ("本地资料", [], "介绍一下这个课题的开题报告，100字以内", ["长安"]),
    ("本地资料（文献）", [], "DARLIN小鼠和CARLIN相比改进了什么？不超过80字", ["TdT"]),
    ("wiki", [], "用一百字介绍一下能天使", [("狙击", "速射手")]),
    ("翻聊天记录", [("user", "我们组决定用Chroma做向量数据库，主要是因为轻量、Python里直接能用，部署也简单"),
                   ("assistant", "好的，记下了～")],
     "我们组之前决定用什么做向量数据库来着？为什么？50字以内", ["Chroma"]),
    ("大模型自身知识（一句话）", [], "一句话说说什么是CRISPR", ["CRISPR"]),
    ("闲聊", [], "用50字左右介绍一下你自己", []),
    ("拆成多个子问题", [], "能天使和德克萨斯的天赋分别是什么？100字以内", ["能天使", "德克萨斯"]),
    ("下限（只要求、不扩写）", [], "至少300字讲讲这个课题的实验设计", ["长安"]),
    ("英文", [], "Explain what CoSpar does in 40 words", ["CoSpar"]),
]


def _apply_baseline():
    """运行时恢复改动前的行为：prompt里去掉字数规则，关掉生成后检查。"""
    rule = llm_generate.LENGTH_RULE
    for name in ("_SYSTEM_PROMPT", "_WEB_SYSTEM_PROMPT", "_OWN_KNOWLEDGE_SYSTEM_PROMPT"):
        setattr(llm_generate, name, getattr(llm_generate, name).replace(rule, ""))
    query_planner.LENGTH_RULE = ""
    conversation_memory.ConversationSession._log_length_check = lambda self, text, reply: None
    import length_limit  # 检索前去掉字数要求也是这次新加的，改动前没有
    length_limit.strip_length_request = lambda text: text
    conversation_memory.strip_length_request = lambda text: text


def _judge(question, reply, must_have):
    req = parse_length_request(question)
    _, body, _ = split_reply(reply)
    n = count_units(body, req.unit)
    has = lambda k: any(x.lower() in body.lower() for x in (k if isinstance(k, tuple) else (k,)))
    facts_ok = all(has(k) for k in must_have)
    if req.one_sentence:
        s = count_sentences(body)
        return facts_ok and s <= 1, f"{s}句（要求一句话）"
    if req.max_units:
        limit = req.max_units * (1 + TOLERANCE)
        return facts_ok and n <= limit, f"{n}{req.unit}（上限{req.max_units}，容忍到{limit:.0f}）"
    return facts_ok, f"{n}{req.unit}（要求至少{req.min_units}，只记录不判）"


def main():
    if BASELINE:
        _apply_baseline()
    llm = build_llm_client()
    if not llm.is_real:
        print("需要真实LLM才能跑这个验证。")
        return
    calls = {"all": 0}
    orig = llm.chat

    def counting(messages, **kw):
        calls["all"] += 1
        return orig(messages, **kw)
    llm.chat = counting

    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    mode = "改动前（baseline）" if BASELINE else "改动后"
    lines = [f"# 字数要求：真机验证（{mode}）", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             "上限按不超过120%判；一句话按句子数判；下限只记录不判。同时检查关键事实还在。", ""]
    ok_n = 0
    n_run = 0
    for i, (note, history, q, must) in enumerate(CASES, start=1):
        if ONLY and i not in ONLY:
            continue
        n_run += 1
        s = ConversationSession(kb, MemoryStore(":memory:"), user_id=f"len_demo_{i}", llm_client=llm)
        s.web_search = _NoWeb()
        for role, text in history:
            s.store.append_turn(s.user_id, role, text)
        calls["all"] = 0
        t0 = time.time()
        reply = s.user_says(q)
        dt = time.time() - t0
        ok, detail = _judge(q, reply, must)
        ok_n += ok
        mark = "✅" if ok else "❌"
        print(f"{mark} [{note}] {q}  正文{detail}，{dt:.1f}秒，LLM{calls['all']}次")
        lines += [f"## {i}. [{note}] {q} {mark}", f"- 正文：{detail}；应含：{must or '-'}",
                  f"- 耗时 {dt:.1f} 秒，LLM调用 {calls['all']} 次", "- 回复：", ""]
        lines += ["> " + ln for ln in reply.splitlines()] + [""]
    lines.insert(6, f"**{ok_n}/{n_run} 符合**" + (f"（只重跑了第{sorted(ONLY)}题）" if ONLY else "") + "\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{ok_n}/{n_run} 符合。报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
