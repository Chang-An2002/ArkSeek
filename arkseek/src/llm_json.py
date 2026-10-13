# -*- coding: utf-8 -*-
"""
LLM返回的JSON解析共用工具

背景：`llm_l1.py`/`llm_l0_summary.py`/`wiki_router.py`三个文件，各自都独立
实现了一遍"LLM有时会加markdown代码块包裹，先剥掉；然后`json.loads()`解析，
解析失败就返回None，不让调用方拿到一个半成品结果"这套逻辑，写法和变量名都
高度相似，纯粹是复制粘贴出来的重复——用户真机测试反馈时问过"这几处route/
判断分类的写法是不是有堆屎山代码的嫌疑"，这三处重复就是当时指出的一个具体
证据，这个文件是那次讨论之后做的第一处清理。

【为什么只抽这两步，不做一个大而全的"通用LLM JSON解析框架"】
三个文件里`_parse_response()`真正各自不同的部分——校验JSON里到底该有哪些
字段、每个字段该是什么类型、字段之间有没有互相依赖的合法性要求（比如
wiki_router.py里"route=wiki却没给keyword就当解析失败"这类业务规则）——是
各自领域特定的，没法共用，也不该抽到一起（抽出来只会变成一个每个字段都要
if-else特判"当前是哪个调用方"的怪东西，比维持现状更难维护）。同样，"解析
失败之后要不要有一个正则兜底、兜底具体怎么写"也是领域特定的：
`llm_l0_summary.py`/`wiki_router.py`能用"抓最后一个字符串字段"这个技巧兜底
（见各自模块开头"真机测试发现的真实问题"一节），是因为它们的JSON形状足够
简单；`llm_l1.py`的`updates`是数组套对象，形状更复杂，没有对应的简单兜底可
写，目前就是解析失败直接返回None，不硬凑一个兜底。

真正在三个文件里完全长得一样、纯粹是复制粘贴的，只有"剥掉markdown代码块
外壳"和"`json.loads()`包一层try/except、失败或者解析结果不是dict就返回
None"这两步——这个文件只抽出这两步，把"抽象出来的东西比原来的重复代码更
复杂"这种为了复用而复用的反模式挡在外面。
"""

import json
import re
from typing import Optional

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)


def strip_json_fence(raw: str) -> str:
    """去掉LLM有时会加的markdown代码块包裹（```json ... ```或``` ... ```），
    效果跟三个文件原来各自的写法（`wiki_router.py`/`llm_l1.py`是同一个正则，
    `llm_l0_summary.py`是手写的字符串前缀判断）完全一致，只是不用再抄三遍。
    """
    return _JSON_FENCE_RE.sub("", raw.strip()).strip()


def try_parse_json(cleaned: str) -> Optional[dict]:
    """`json.loads()`包一层try/except：解析失败（`JSONDecodeError`）或者
    解析成功但结果不是一个dict（比如LLM返回了一个JSON数组），统一返回
    None，不抛异常、也不区分这两种失败——调用方看到None，要么去试自己的
    正则兜底，要么直接放弃这一轮，两种处理方式都不需要知道失败的具体原因。
    """
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
