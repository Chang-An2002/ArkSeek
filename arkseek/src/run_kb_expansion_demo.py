# -*- coding: utf-8 -*-
"""
知识库扩充（QQ群压力测试准备（三））的真机验证

知识库从"两份课题报告"扩充到：Zotero「本地知识库」13篇文献原文 + 8次JC的讲稿/
笔记/PPT + 文献目录，向量模型换成中英通用的bge-m3，切片改成按token数。这个脚本
用真实LLM走完整对话流程（路由 -> 检索/目录 -> 生成 -> 必要时兜底），每题一个新
会话，检查：去哪里查的、答得对不对（预期关键词）、引用的是不是对的资料。

题目分组：
  目录：列清单类问题，应该直接查文献目录、列全
  JC：问某一次JC，包括只有PPT、没写讲稿的25.11.29
  文献内容：中文问、英文问、中英混杂问文献里的具体内容
  讲稿视角：问"作者是怎么讲的"，应该引用讲稿/笔记
  回归：原来两份课题报告的问题照样能答
  范围外：知识库里没有的，应转联网搜索并标注（不能硬从文献里凑）

用法：
    cd src
    python run_kb_expansion_demo.py
"""

import os
import re
from datetime import datetime

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession
from llm_client import build_llm_client

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, "..", "kb_expansion_demo_results.md")

# (分组, 问题, 预期回答里应该出现的关键词（任一组全部命中即可）, 预期来源关键词, 说明)
CASES = [
    ("目录", "作者都读过哪些文献？", [["13篇", "LoxCode", "CoSpar", "没在JC上讲过"]], "文献目录",
     "应直接查目录列全13篇，讲过/没讲过分组"),
    ("目录", "长安做过几次JC？都讲了哪些文章？", [["ArchR", "SPACE-seq"]], "文献目录",
     "列清单类，JC讲过8篇；用昵称提问"),
    ("JC", "长安2025年7月5日那次JC讲的是什么？", [["LoxCode"]], "JC",
     "日期只出现在文件名里，靠片段开头的文档标签才能匹配"),
    ("JC", "25.11.29那次JC讲了哪篇文章，主要结论是什么？", [["肝细胞癌", "分区"], ["hepatocellular", "zonation"],
                                                    ["肝癌", "分区"]], "",
     "这次只有PPT没写讲稿，要从PPT和文献原文里答"),
    ("文献内容", "LoxCode是怎么产生高多样性条形码的？", [["Cre"], ["loxP"], ["重组"]], "LoxCode",
     "中文问英文文献"),
    ("文献内容", "What problem does CoSpar solve?", [["CoSpar"]], "CoSpar",
     "纯英文问英文文献"),
    ("文献内容", "DARLIN小鼠和CARLIN相比改进了什么？", [["多样性"], ["diversity"]], "",
     "需要对比两篇文献（Li 2023 DARLIN / Bowling 2020 CARLIN）"),
    ("文献内容", "SPACE-seq结合了哪两类技术？", [["空间"], ["spatial"]], "SPACE-seq",
     "SPACE-seq（最新一次JC）"),
    ("文献内容", "基于DNA甲基化的谱系追踪用的是什么测序方法？", [["甲基化"], ["methylation"]], "",
     "Chen 2025 Nature Methods"),
    ("讲稿视角", "作者在JC上是怎么给大家介绍ArchR的？", [["ArchR"]], "JC",
     "应引用作者的讲稿/PPT，而不只是文献原文"),
    ("回归", "这个课题是和哪个课题组合作的？", [["林盛达"]], "课题",
     "原来两份课题报告的问题，扩充后照样要答对"),
    ("回归", "lineage tracing课题使用的是什么动物的什么组织？", [["小鼠", "肝"]], "",
     "run_tests第6题（英文术语问中文课题）"),
    ("范围外", "AlphaFold3的原理是什么？", [["【以上内容来自网络搜索"],
                                       ["【以上由大模型根据自身知识回答"]], "",
     "知识库里没有，应转联网搜索并标注，不能硬从谱系追踪文献里凑。路由直接判成一般知识问题"
     "（不是先查本地再落空）也对，所以不要求出现'作者的课题资料里没有'"),
    ("文献内容", "Why does DARLIN fuse TdT to Cas9?", [["TdT", "insertion"], ["TdT", "diversity"]], "",
     "英文问题应该用英文回答（第6题曾用中文回答英文问题）"),
    ("讲稿视角", "长安的课题用的是什么小鼠模型？", [["长安"]], "课题",
     "应该称'长安'，不能说成'你的课题'，也不能用他/她指代长安"),
    ("称呼", "LoxCode那篇文章的作者是谁？", [["Weber"]], "",
     "'作者'在这里指文献作者（Weber等人），不能答成长安；目录条目里标了'文献作者：'"),
    ("称呼", "作者读过CoSpar那篇文献吗？", [["长安", "CoSpar"]], "",
     "群友说'作者'又没指明哪篇文献时指长安：应回答长安读过、但没在JC上讲过"),
]


_CJK = re.compile(r"[一-鿿]")
# 第一轮真机验证里出现过的错误称呼（"你的lineage tracing课题""你们与…合作"
# "他在其中负责…"）
_FORBIDDEN_ADDRESS = ["你的课题", "你的lineage", "你们与", "你们的课题", "他在", "他负责", "他还", "他没有",
                      "她在", "她负责", "她还", "他的", "她的"]


_REAL_NAMES = __import__("kb_sources").load_config().get("owner_real_names", [])


def main():
    kb = MemorySystemPipeline(os.path.join(HERE, "..", "data", "knowledge_base"))
    client = build_llm_client()
    if not client.is_real:
        print("需要真实LLM才能跑这个验证。")
        return

    lines = ["# 知识库扩充：真机验证", "",
             f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "",
             f"向量检索：**{kb.dense.name}**　精排：**{kb.reranker.name}**　LLM：**{client.name}**　"
             f"知识库片段数：{len(kb.chunks)}", ""]
    passed = 0
    for i, (group, q, kw_sets, src_kw, note) in enumerate(CASES, start=1):
        session = ConversationSession(kb, MemoryStore(":memory:"), user_id=f"kbx_{i}", llm_client=client)
        reply = session.user_says(q)
        kw_ok = any(all(k.lower() in reply.lower() for k in ks) for ks in kw_sets)
        src_ok = (not src_kw) or (src_kw in reply)
        # 通用检查（每题都查）：英文提问要用英文回答；提到长安的资料时称"长安"，
        # 不能说成"你的课题"，也不能用他/她指代长安，也不能出现长安的真名
        body = reply.split("引用来源：")[0].split("参考链接：")[0]
        lang_ok = True
        if not _CJK.search(q):
            lang_ok = len(_CJK.findall(body)) < 0.1 * max(len(body), 1)
        # (?<!其)：排除"其他在…"这类正常用法（第二轮验证里误报过）
        bad_words = ([w for w in _FORBIDDEN_ADDRESS if re.search("(?<!其)" + re.escape(w), body)]
                     if ("长安" in reply or "作者" in reply) else [])
        # 真名不能出现在回答里（入库时已替换成昵称，见kb_sources.redact_owner_names()）
        bad_words += [n for n in _REAL_NAMES if n.lower() in reply.lower()]
        ok = kw_ok and src_ok and lang_ok and not bad_words
        passed += ok
        print(f"\n=== [{group}] {q}  {'✅' if ok else '❌'}\n{reply}")
        lines += [f"## {i}. [{group}] {q} {'✅' if ok else '❌'}", f"- 说明：{note}",
                  f"- 预期关键词：{' 或 '.join('+'.join(ks) for ks in kw_sets)}（{'命中' if kw_ok else '未命中'}）"
                  + (f"；预期来源含「{src_kw}」（{'是' if src_ok else '否'}）" if src_kw else "")
                  + ("" if lang_ok else "；❗英文提问却用中文回答")
                  + (f"；❗称呼不对（或出现了真名）：{bad_words}" if bad_words else ""),
                  "- 回复：", "", "  > " + reply.replace("\n", "\n  > "), ""]
    lines.insert(6, f"**{passed}/{len(CASES)} 符合预期**\n")
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n{passed}/{len(CASES)} 符合预期。报告已写到：{os.path.abspath(REPORT)}")


if __name__ == "__main__":
    main()
