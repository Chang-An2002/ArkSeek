# -*- coding: utf-8 -*-
"""
"最近对话"上下文格式化 —— 从 wiki_router.py / llm_l1.py 两处逐字相同的
`_format_recent_context()`私有实现里抽出来的公共版本。

【为什么现在才抽，之前为什么没抽】
wiki_router.py模块开头"给以后再加route分类定的一条纪律"和`_format_recent_
context()`注释里都提过同一个判断标准：两个文件各自维护一份完全相同的小
函数时，先不抽象——"抽出来比维持重复更复杂"的反面案例；等真的出现第三个
消费者，再合并成公共模块，跟`llm_json.py`当初"三处重复了才抽"是同一个门槛。

这次多跳检索架构升级（见`query_planner.py`模块开头）一次性把消费者从2个
变成4个：`llm_generate.py`要把这段"最近对话"喂给L2/L3的生成步骤（让它在
证据本身不够消歧的情况下也能正确处理"她"这类指代，完整背景见该文件里
`extra_context`参数的说明）；`query_planner.py`的多跳编排器要用同一种格式
把"前面子查询已经查到的结果"当成一段类似"最近对话"的背景，注入给依赖它的
下一个子查询（复用的正是route_query()解析代词/指代这同一套机制，不是另外
发明一套占位符替换逻辑，完整背景见`query_planner.py`模块开头"依赖怎么解析"
一节）。凑够了三个新增消费者，过了当初定的门槛，所以这次把重复的实现合并
到这里，wiki_router.py/llm_l1.py原有的私有实现同步删除、改成从这里导入，
逐字行为不变。

不需要更复杂的抽象：所有调用方对这段"背景文本"的用法（放在prompt最前面，
当成帮助理解代词/省略指代、或者帮助理解"已经查到的前置信息"的背景块，
不是让LLM把背景内容当成当前要处理的对象）完全一致，没有各自定制的余地，
一个纯函数就够，不需要类或者配置项。
"""

from typing import Iterable


def format_recent_context(recent_context: Iterable) -> str:
    """把一组形如{"role":..., "text":...}的行（兼容sqlite3.Row，以及
    query_planner.py编排器临时拼的普通dict）格式化成一小段"最近对话："
    前缀文本。

    recent_context为空/None时返回空字符串——这是所有调用方都依赖的向后
    兼容保证：不传这个参数、或者传空列表（比如对话刚开始、还没有任何历史）
    时，prompt要跟"从来没有这段逻辑"时逐字一致，不能引入任何多余的文本。
    """
    if not recent_context:
        return ""
    lines = [f"{row['role']}：{row['text']}" for row in recent_context]
    return "最近对话：\n" + "\n".join(lines) + "\n"


CURRENT_LABEL = "当前这句话"


def with_current_message(recent_context: Iterable, text: str) -> str:
    """背景+当前这句话，拼成user消息的内容。

    【2026-10-01真机：隔了三天发"中午好"，被路由成"查课题报告"】原来是
    `f"{format_recent_context(ctx)}{text}"`，当前这句直接接在最近对话最后
    一行后面，没有任何标记——模型分不清哪句是这一轮要处理的，"最近对话"里
    上一轮的请求（"请介绍我的开题和中期报告"）被当成了当前问题（复现10次
    4次判成local_kb，1次把更早的"再次显示摘要"又答了一遍）。跟常见做法一样
    把背景和当前输入显式分开、加标签。

    没有背景时原样返回text，prompt跟以前逐字一致。
    """
    block = format_recent_context(recent_context)
    if not block:
        return text
    return f"{block}\n{CURRENT_LABEL}：{text}"


def current_time_line() -> str:
    """每次请求时的当前时间，接在会写回复的system prompt末尾。

    【2026-10-01真机】问"今天是什么节日"，回答成了4月的某个节日——prompt里从来
    没有告诉过模型现在是哪天，它只能凭训练数据猜。跟ChatGPT/Claude等产品的常见
    做法一样，每次请求都把当前日期、星期、时间写进system prompt。放在末尾：前面
    固定的部分不变，不影响接口的前缀缓存。用固定的东八区，不依赖系统时区和tzdata。
    """
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone(timedelta(hours=8)))
    weekday = "一二三四五六日"[now.weekday()]
    return (f"\n当前时间：{now.year}年{now.month}月{now.day}日（星期{weekday}）"
            f"{now.hour:02d}:{now.minute:02d}，北京时间。用户说的“今天”“现在”“明天”“这周”以此为准。")
