# -*- coding: utf-8 -*-
"""
用户在问题里提的字数要求（2026-09-29，用户定的方案）

【问题】"100字以内"只是问题原文的一部分，没有任何一条规则要求模型遵守，真机里
回答经常明显超出。

【做法：业界处理用户格式/长度要求的常规两步】
1. 生成时明确要求：各生成prompt里加一条"问题里写了字数要求就遵守、字数只算正文"
   （llm_generate.py、query_planner.synthesize()）。
2. 生成后用代码数一遍正文，只打日志（超出上限20%以上、"至少N字"写短了、"一句话"
   写了不止一句）。
   【2026-10-01去掉了"超出就调LLM压缩"】原来超出上限20%以上会再调一次LLM压缩。
   推理模型压缩时会在思考里反复逐字数字数，同一个压缩任务4~99秒不等、经常超时，
   成了回复慢和超时的主要来源。用户决定字数靠大模型读问题自己控制——目的是"别长篇
   大论"，99字和101字没区别。"至少N字"写短了也不扩写（资料只有这么多，硬扩写只能靠编）。
   生成prompt里的字数规则（2026-10-01）也从"硬性要求"放宽成"大致在这个范围，
   不用逐字数"。

字数只算正文：来源标题（【L3知识库——…】）、开头的"（长安的资料里没有找到…）"、
"引用来源"列表、【以上内容来自…】这类标注、[1]引用编号、markdown加粗都不算。

只处理明确给了数字或"一句话"的说法；"简短点"这种没有数字的不处理（用户2026-09-29
同意：模型本来会自己缩短，也不知道"简短"该是多少字）。
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple

TOLERANCE = 0.2  # 超出上限20%以内算达标（用户2026-09-29同意）

_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUM = r"(\d+|[零一二两三四五六七八九十百千]+)"


def _to_int(s: str) -> Optional[int]:
    if s.isdigit():
        return int(s)
    total, cur = 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            cur = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (cur or 1) * _CN_UNITS[ch]
            cur = 0
        else:
            return None
    total += cur
    return total or None


@dataclass
class LengthRequest:
    max_units: Optional[int] = None   # 上限（"100字以内"），"左右"也按上限检查
    min_units: Optional[int] = None   # 下限（"至少300字"），只在生成时要求，不自动扩写
    unit: str = "字"                  # "字" 或 "words"
    one_sentence: bool = False
    raw: str = ""                     # 识别到的原文片段，写日志用


_RANGE_ZH = re.compile(_NUM + r"\s*(?:到|至|-|~|～)\s*" + _NUM + r"\s*个?字")
_MAX_ZH_AFTER = re.compile(_NUM + r"\s*个?字\s*(?:以内|之内|内|以下)")
# 少写了"字"的"用100内介绍……"（2026-10-01真机）。前面必须有"用/写/控制在"，避免把
# "费用10以内的干员""使用10以内"这类数值条件当成字数要求
_MAX_ZH_NO_UNIT = re.compile(r"(?:(?<![费使作])用|写|控制在)\s*" + _NUM + r"\s*(?:以内|之内|内)")
_MAX_ZH_BEFORE = re.compile(r"(?:不超过|不要超过|别超过|不多于|最多|少于|控制在|限)\s*" + _NUM + r"\s*个?字")
_MIN_ZH = re.compile(r"(?:至少|不少于|不低于|起码|最少)\s*" + _NUM + r"\s*个?字|" + _NUM + r"\s*个?字\s*以上")
_APPROX_ZH = re.compile(_NUM + r"\s*个?字\s*(?:左右|上下)|(?:大约|大概|约)\s*" + _NUM + r"\s*个?字|(?:用|写)\s*" + _NUM + r"\s*个?字")
_ONE_SENTENCE = re.compile(r"一句话")
_MAX_EN = re.compile(r"\b(?:in|within|under|at most|no more than|less than|fewer than)\s+(\d+)\s+words?\b", re.I)
_MIN_EN = re.compile(r"\b(?:at least|no less than|more than|over)\s+(\d+)\s+words?\b", re.I)
# 不带修饰词的"3 words"不算（"what do these 3 words mean"不是字数要求）
_APPROX_EN = re.compile(r"\b(?:about|around|roughly|approximately)\s+(\d+)\s+words?\b", re.I)


def _first_num(m) -> Optional[int]:
    for g in m.groups():
        if g:
            return _to_int(g)
    return None


def parse_length_request(text: str) -> Optional[LengthRequest]:
    """从用户的话里识别字数要求；没有明确的数字/"一句话"就返回None（行为跟以前一样）。"""
    m = _RANGE_ZH.search(text)
    if m:
        a, b = _to_int(m.group(1)), _to_int(m.group(2))
        if a and b:
            return LengthRequest(max_units=max(a, b), min_units=min(a, b), raw=m.group(0))
    for rx, kind in ((_MAX_ZH_AFTER, "max"), (_MAX_ZH_NO_UNIT, "max"), (_MAX_ZH_BEFORE, "max"), (_MIN_ZH, "min"), (_APPROX_ZH, "max")):
        m = rx.search(text)
        if m and _first_num(m):
            n = _first_num(m)
            return LengthRequest(max_units=n if kind == "max" else None, min_units=n if kind == "min" else None,
                                 raw=m.group(0))
    for rx, kind in ((_MAX_EN, "max"), (_MIN_EN, "min"), (_APPROX_EN, "max")):
        m = rx.search(text)
        if m:
            n = int(m.group(1))
            return LengthRequest(max_units=n if kind == "max" else None, min_units=n if kind == "min" else None,
                                 unit="words", raw=m.group(0).strip())
    if _ONE_SENTENCE.search(text):
        return LengthRequest(one_sentence=True, raw="一句话")
    return None


_ALL_PATTERNS = (_RANGE_ZH, _MAX_ZH_AFTER, _MAX_ZH_NO_UNIT, _MAX_ZH_BEFORE, _MIN_ZH, _APPROX_ZH, _MAX_EN, _MIN_EN, _APPROX_EN)


def strip_length_request(text: str) -> str:
    """检索用：去掉字数要求这类"对回答形式的要求"，只留问题本身。

    实测（2026-09-29）字数要求原样进检索会拉低相关性：精排分"讲讲这个课题的实验设计"
    0.434 → 加上"至少300字"0.298；"介绍一下这个课题的开题报告"0.924 → 加上"，100字
    以内"0.833（置信度门槛0.5）。"至少300字讲讲这个课题的实验设计"因此没查到长安的
    课题报告，退到了大模型自身知识。业界做法：检索用的查询和对回答的格式要求分开——
    检索用去掉要求的问题，生成时仍然看完整原句（字数规则照样生效）。"""
    out = text
    for rx in _ALL_PATTERNS:
        out = rx.sub("", out)
    out = re.sub(r"用?一句话", "", out)
    out = re.sub(r"^[\s，,、；;。]+|[\s，,、；;]+(?=[？?。！!]*$)", "", out)
    out = re.sub(r"([，,、；;])\s*[，,、；;]+", r"\1", out).strip()
    return out or text


# ---------- 拆出正文 ----------

_HEAD_LINE = re.compile(r"^(?:【[^】\n]*】|（[^）\n]*(?:没有找到|以下来自)[^）\n]*）)[ \t]*$")
_TAIL_START = re.compile(r"\n\s*(?:(?:引用来源|参考链接)\s*[:：]|【以上)")


def split_reply(reply: str) -> Tuple[str, str, str]:
    """拆成 (开头的来源标题行, 正文, 结尾的引用列表/来源标注)。三段拼回去等于原文。"""
    lines = reply.split("\n")
    n_head = 0
    while n_head < len(lines) and _HEAD_LINE.match(lines[n_head].strip() or "x"):
        n_head += 1
    head = "\n".join(lines[:n_head])
    rest = "\n".join(lines[n_head:])
    if head:
        head += "\n"
    m = _TAIL_START.search(rest)
    if m:
        return head, rest[:m.start()], rest[m.start():]
    return head, rest, ""


_MARKUP = re.compile(r"\[(?:\d+|画像)\]|\*\*|^[ \t]*[-*•][ \t]+", re.M)


def count_units(body: str, unit: str = "字") -> int:
    clean = _MARKUP.sub("", body)
    if unit == "words":
        return len(re.findall(r"[A-Za-z0-9'’-]+", clean)) + len(re.findall(r"[一-鿿]", clean))
    return len(re.sub(r"\s", "", clean))


def count_sentences(body: str) -> int:
    clean = _MARKUP.sub("", body).strip()
    return len([s for s in re.split(r"[。！？!?]+|\n+", clean) if s.strip()])


def exceeds_limit(req: LengthRequest, body: str) -> bool:
    """正文超出上限20%以上（"一句话"按句子数）。只用来打日志，见模块开头。"""
    if req.one_sentence:
        return count_sentences(body) > 1
    if req.max_units:
        return count_units(body, req.unit) > req.max_units * (1 + TOLERANCE)
    return False
