# -*- coding: utf-8 -*-
"""
每条消息的分步计时（2026-09-30，处理"回复慢"的第一步：先量再改）

【做法】跟 OpenTelemetry / LangSmith 的 trace-span 是同一个思路，只是做成了几十行的
简化版、只打日志：一条消息是一个 trace，里面每个步骤（一次LLM调用、一次召回、
一次精排、一次wiki请求……）是一个 span，记下"相对这条消息开始过了多久开始、
花了多久、在哪个并行任务里"。并行执行时几个 span 会重叠，所以日志里同时列出
每步的开始时刻，看得出哪些是同时进行的、哪一段在关键路径上。

【怎么用】
- 入口建一个 Trace，用 trace.run(fn, ...) 执行这条消息的处理，结束时 trace.emit()
  打印汇总表。qq_bot 在每条消息上建（这样能把"发送"也算进去）；没有外层 trace 时
  user_says() 自己建（终端版、演示脚本）。没有 trace 时 span() 什么都不记。
- 需要计时的地方写 `with span("精排", detail="3个问法×20条"):`。
- trace 靠 contextvars 往下传；concurrency.submit() 把当前 context 带进线程池，
  所以子任务里的 span 也记在同一条消息上。
- 环境变量 ARKSEEK_TIMING=0 关掉打印（照样记，只是不输出）。
"""

import contextvars
import os
import threading
import time
from contextlib import contextmanager
from typing import List, Optional

_current: contextvars.ContextVar = contextvars.ContextVar("arkseek_trace", default=None)
_PRINT = os.environ.get("ARKSEEK_TIMING", "1") != "0"


class Trace:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.perf_counter()
        self.spans: List[tuple] = []  # (开始秒, 耗时秒, 名称, 细节, 任务标签)
        self._lock = threading.Lock()

    def add(self, start: float, dur: float, name: str, detail: str, task: Optional[str]):
        with self._lock:
            self.spans.append((start - self.t0, dur, name, detail, task))

    def report(self) -> str:
        total = time.perf_counter() - self.t0
        lines = [f"[timing] {self.label}：总耗时 {total:.1f} 秒",
                 "[timing]   开始    耗时  步骤"]
        for start, dur, name, detail, task in sorted(self.spans, key=lambda s: s[0]):
            where = f" [{task}]" if task else ""
            extra = f"（{detail}）" if detail else ""
            lines.append(f"[timing] {start:6.1f}s {dur:6.1f}s  {name}{extra}{where}")
        # 按步骤类别汇总（并行时相加会超过总耗时，这里给的是"各类一共占用了多少秒"）
        by_kind = {}
        for _, dur, name, _, _ in self.spans:
            kind = name.split(" ")[0]
            by_kind[kind] = by_kind.get(kind, 0.0) + dur
        summary = "，".join(f"{k} {v:.1f}s" for k, v in sorted(by_kind.items(), key=lambda kv: -kv[1]))
        lines.append(f"[timing]   各类合计：{summary}")
        return "\n".join(lines)


    def run(self, fn, *args, **kwargs):
        """在"当前trace=自己"的context里执行fn（可以在任何线程里调用，比如
        qq_bot把它交给run_in_executor——run_in_executor本身不会带上contextvars）。"""
        def inner():
            _current.set(self)
            return fn(*args, **kwargs)
        return contextvars.copy_context().run(inner)

    def emit(self, extra: str = "") -> str:
        text = self.report() + (f"\n[timing]   {extra}" if extra else "")
        if _PRINT:
            print(text)
        return text


def current_trace() -> Optional[Trace]:
    return _current.get()


def timed(name: str):
    """装饰器版的span：整个函数算一步。"""
    def deco(fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with span(name):
                return fn(*args, **kwargs)
        return wrapper
    return deco


@contextmanager
def span(name: str, detail: str = ""):
    """yield一个dict；步骤做完才知道的信息（比如LLM输出了多少token）写进
    info["after"]，会接在detail后面。"""
    info = {}
    trace = _current.get()
    if trace is None:
        yield info
        return
    from concurrency import current_tag
    start = time.perf_counter()
    try:
        yield info
    finally:
        if info.get("after"):
            detail = f"{detail}，{info['after']}" if detail else info["after"]
        trace.add(start, time.perf_counter() - start, name, detail, current_tag())
