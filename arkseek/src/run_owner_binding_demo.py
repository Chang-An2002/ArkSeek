# -*- coding: utf-8 -*-
"""
主人绑定 + 群昵称的真机验证（2026-09-28，第一次QQ群测试之后）

群测时长安问"那你介绍一下我的开题报告吧"，机器人不知道提问的就是长安，本地知识库
没过线、转去联网搜索，答了一段不相干的内容；普通群友问自己的事时也会这样。修复：
①按账号ID绑定主人（.env 的 ARKSEEK_OWNER_IDS，这里用环境变量模拟），主人说"我的…"
就查长安的资料；②非主人问"我的开题报告/我是做什么研究的"判给history，从对话里找，
找不到就说没记录，不去查长安的资料冒充、也不联网搜索；③群里用QQ昵称称呼别人。

同一个群里依次：主人问 -> 群友问同样的问题 -> 群友问"刚才谁问了开题报告"。
联网搜索换成假的，只记录有没有被调用。

用法：
    cd src
    python run_owner_binding_demo.py
"""

import os
from datetime import datetime

os.environ["ARKSEEK_OWNER_IDS"] = "OWNERID"

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession
from llm_client import build_llm_client

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "owner_binding_demo_results.md")


class _NoWeb:
    is_real = True
    name = "fake"

    def __init__(self):
        self.queries = []

    def search(self, q, max_results=5):
        self.queries.append(q)
        return None


GID = "qq_group:demo"
OWNER, FRIEND = f"{GID}:OWNERID", f"{GID}:FRIEND"
# (谁问, 问题, 回答里应该有（任一）, 回答里不能有, 能不能联网搜索, 说明, 写明是长安的就可以提到)
# 最后一项（用户2026-09-29定）：群友问自己的东西时，回答里可以提到长安那份的内容，但必须写明
# 是长安的——"不能有"的词出现时，只要正文里也写了"长安"就不算冒充；没写就是冒充。
CASES = [
    (OWNER, "那你介绍一下我的开题报告吧，100字以内", ["谱系", "条形码", "CRISPR"], ["网络搜索"], False,
     "主人本人问'我的开题报告'：应查长安的课题报告作答", False),
    (FRIEND, "那你介绍一下我的开题报告吧，100字以内", ["没", "不知道", "不清楚"], ["CRISPR", "谱系追踪"], False,
     "群友问'我的开题报告'：应说没有TA的记录，不能拿长安的报告冒充（可以提长安的，但要写明是长安的），也不联网", True),
    (FRIEND, "我是做什么研究的呀", ["没", "不知道", "不清楚"], ["谱系", "单细胞"], False,
     "群友问自己：同上", True),
    (FRIEND, "刚才谁问了开题报告", ["长安"], ["群友"], True,
     "群里翻旧账：应说出是长安（主人）问的，不暴露'群友A'这类内部代号", False),
]


def main():
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能跑这个验证。")
        return
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    store = MemoryStore(":memory:")
    store.set_display_name(FRIEND, "测试昵称")
    sessions = {}
    lines = ["# 主人绑定 + 群昵称：真机验证", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    ok_n = 0
    for i, (who, q, any_of, none_of, web_ok, note, attributed_ok) in enumerate(CASES, start=1):
        s = sessions.get(who)
        if s is None:
            s = sessions[who] = ConversationSession(kb, store, user_id=who, conversation_id=GID, llm_client=client)
            s.web_search = _NoWeb()
        before = len(s.web_search.queries)
        reply = s.user_says(q)
        searched = len(s.web_search.queries) > before
        body = reply.split("引用来源：")[0]
        forbidden_hit = any(k in body for k in none_of) and not (attributed_ok and "长安" in body)
        ok = any(k in body for k in any_of) and not forbidden_hit and (web_ok or not searched)
        ok_n += ok
        who_label = "主人" if who == OWNER else "群友"
        print(f"\n=== [{who_label}] {q} {'✅' if ok else '❌'}（联网搜索：{'调用了' if searched else '没调用'}）\n{reply}")
        lines += [f"## {i}. [{who_label}] {q} {'✅' if ok else '❌'}", f"- 说明：{note}",
                  f"- 应有其一：{any_of}；不能有：{none_of}；联网搜索：{'调用了' if searched else '没调用'}",
                  "- 回复：", "", "  > " + reply.replace("\n", "\n  > "), ""]
    lines.insert(4, f"**{ok_n}/{len(CASES)} 符合预期**\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n{ok_n}/{len(CASES)} 符合预期。报告：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
