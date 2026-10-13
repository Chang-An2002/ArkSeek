# -*- coding: utf-8 -*-
"""
给课题报告的片段补上"这段能回答的问题"（2026-09-30，同义说法检索改进第二步）

【为什么】用户问课题时用口语（"交出什么""进度规划""怎么做的"），课题报告写的是书面
正文。上一步（章节路径标题+同义说法表）对写表时看过的问法15/15，对新说法只有4/9——
表是一个词一个词补的，覆盖不了口语的各种说法；而且留出集在改动前就能把正确章节召回到
前3条，卡的是精排给这些片段的分数太低（问题和正文字面差太远）。

【做法：业界的doc2query / "假设问题"索引】建索引时让LLM替每个片段写几个"用户可能怎么问"
的问题，写进片段里（放在章节标题下面一行），BM25、向量、精排都能用上——用户的问法往往
跟这些问题很接近。只在建索引时算一次、结果缓存；以后每次提问不多花任何调用。

【离线生成，不在启动时调LLM】生成是单独的一步：`python build_kb_questions.py`，结果写进
data/kb_question_cache.json（本机缓存，不提交）。启动机器人、跑测试时只读缓存，不调LLM——
缓存缺了照常能用（只是那些片段没有补充问题），启动时提示去跑一下生成脚本。这样测试不会
意外调LLM，换一台电脑也不会一启动就先调几十次。

缓存按片段原文（章节标题+正文，补问题之前的）的哈希存：报告改了、切法改了，对应片段的
哈希就变了，重新生成；没变的直接复用。

只做课题报告（有清楚的章节结构、问法最口语）；JC讲稿、文献原文暂不做，看效果再说。
"""

import hashlib
import json
import os
import re
from typing import Dict, List

from llm_json import strip_json_fence, try_parse_json

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "..", "data", "kb_question_cache.json")
# 改了生成prompt就加1，旧缓存全部作废。v2（2026-09-30）：表格/时间表也算内容（v1把
# "2.4 研究计划"的时间表判成没内容、没生成问题）；不许用他/她/他们指代长安
QUESTIONS_VERSION = 2
GENERATE_CONCURRENCY = 2  # 批量生成时同时几个请求：4个持续打满时撞过MiniMax Token Plan的速率限制（429）
N_QUESTIONS = 4
_TARGET_DOC_TYPES = ("课题报告",)

_PROMPT = """下面是{owner}的课题报告里的一个片段（第一行是它所在的文档和章节）。这个聊天机器人会被
别人问起{owner}的课题，请列出{n}个这段内容能直接回答的问题，用来帮检索找到这段。

要求：
- 像真人在群里随口问的那样，口语化，换着不同的说法和角度问；可以有一两个正式说法；
- 称呼课题时用"这个课题""{owner}的课题""这个项目""开题/中期报告"这类说法；提到{owner}
  就直接写"{owner}"，不要用"他""她""他们"；
- 只写这段内容真能回答的问题，不要写答案，不要编片段里没有的东西；
- 表格、时间安排表、软件清单也是实际内容，照样出问题；只有片段仅仅是参考文献列表、
  或者只有一个图的标题时，questions才给空数组。

只输出JSON对象，比如{{"questions": ["问题1", "问题2"]}}，不要输出别的文字。

片段：
{chunk}"""


_QUESTION_LINE = re.compile(r"\n（这段能回答的问题：[^\n]*）(?=\n|$)")


def strip_questions(text: str) -> str:
    """补进片段里的问题只给检索用（BM25/向量/精排），交给生成模型和展示证据时去掉——
    业界做法里这类为检索补充的文字不给回答用，免得模型把"可能被问的问题"当成资料内容。"""
    return _QUESTION_LINE.sub("", text)


def _key(base_text: str) -> str:
    return hashlib.sha256(f"v{QUESTIONS_VERSION}\n{base_text}".encode("utf-8")).hexdigest()


def _load_cache() -> Dict[str, List[str]]:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _targets(chunks) -> list:
    return [c for c in chunks if c.meta.get("doc_type") in _TARGET_DOC_TYPES]


def attach_questions(chunks) -> None:
    """把缓存里的问题写进对应片段（章节标题下面一行）。缓存缺的片段原样不动，打一行提示。"""
    cache = _load_cache()
    missing = 0
    for c in _targets(chunks):
        qs = cache.get(_key(c.text))
        if qs is None:
            missing += 1
            continue
        if qs:
            head, sep, body = c.text.partition("\n")
            c.text = f"{head}\n（这段能回答的问题：{'｜'.join(qs)}）{sep}{body}"
    if missing:
        print(f"[kb_questions] {missing}个课题报告片段还没有生成\"能回答的问题\"（不影响使用，只是这些片段"
              f"对口语问法没那么敏感）——运行 python build_kb_questions.py 生成。")


def generate_missing(chunks, client, owner: str) -> int:
    """给缓存里还没有的课题报告片段生成问题并写回缓存（chunks必须是补问题之前的）。返回生成了几个。"""
    from concurrency import submit
    cache = _load_cache()
    todo = [c for c in _targets(chunks) if _key(c.text) not in cache]

    def one(c):
        raw = client.chat([{"role": "user", "content": _PROMPT.format(owner=owner, n=N_QUESTIONS, chunk=c.text)}],
                          caller="kb_questions()")
        # try_parse_json只认JSON对象（数组会返回None），所以让模型输出{"questions": [...]}
        # （2026-09-30第一版要的是裸数组，全部被当成解析失败，白调了一轮）
        data = try_parse_json(strip_json_fence(raw)) if raw else None
        qs = data.get("questions") if data else None
        if not isinstance(qs, list):
            return None
        return [q.strip() for q in qs if isinstance(q, str) and q.strip()][:N_QUESTIONS]

    # 分批、每批GENERATE_CONCURRENCY个同时发，别持续打满接口的速率限制
    results = []
    for start in range(0, len(todo), GENERATE_CONCURRENCY):
        batch = [(c, submit(one, c, tag=f"片段{start + j + 1}/{len(todo)}"))
                 for j, c in enumerate(todo[start:start + GENERATE_CONCURRENCY])]
        results += [(c, fut.result()) for c, fut in batch]
    done = failed = 0
    for c, qs in results:
        if qs is None:
            failed += 1
            continue
        cache[_key(c.text)] = qs
        done += 1
        if done <= 3:  # 打几条样例，确认生成的东西像样
            print(f"[kb_questions] 样例：{c.text.split(chr(10), 1)[0][-40:]} -> {qs}")
    if failed:
        print(f"[kb_questions] {failed}/{len(todo)}个片段生成失败（调用失败或格式不对），下次运行会再试")
    # 只留当前还存在的片段，报告改过之后旧片段的问题不留
    alive = {_key(c.text) for c in _targets(chunks)}
    cache = {k: v for k, v in cache.items() if k in alive}
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1)
    return done
