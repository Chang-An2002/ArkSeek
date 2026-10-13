# -*- coding: utf-8 -*-
"""
一条消息内部的并行执行（2026-09-28，用户定的"第二档"+推测执行）

【为什么要有】一轮对话的耗时几乎全是串行的LLM调用（每次1~3秒，prompt长短影响
很小，见README）。两处可以把等待重叠起来：
  ①推测执行：每条消息开头的"画像判断 / 拆分判断 / 路由判断"输入几乎一样，
    互不需要对方的输出，同时发出去，回来后仍按原来的优先级挑结果、用不上的丢掉
    （conversation_memory.ConversationSession._first_steps_parallel()）。
  ②子问题按依赖图调度：decompose()拆出的子问题，依赖都齐了就开跑，互不依赖的
    同时跑；多候选展开（fan-out）的各分支也同时跑（_run_plan_parallel()）。
业界对应：LLMCompiler（规划出依赖图、依赖满足就派发）、LangGraph的并行分支；
推测执行常见于"安全审核和主回答同时跑"。

【这里提供的东西】
- submit()：往共享线程池里扔一个任务，返回Future。并行关闭、或者当前已经在
  池子里的任务线程里时，当场同步执行（返回一个已经完成的Future）——池子里的
  任务从不等待池子里的其它任务，所以不会因为池子占满而互相等死。
- set_parallel()/parallel_enabled()：总开关，默认开；环境变量ARKSEEK_PARALLEL=0
  关闭。关闭时整条链路逐字回到改动前的串行代码（不做推测执行），编排层回归
  测试的"按顺序剧本"用例就跑在这个模式下，见run_orchestration_regression_tests.py。
- LockedCalls：给跨线程共享的对象套一把锁（embedding/精排模型，见pipeline.py）。
- 日志标签：并行后几个任务的print会交错，任务线程里print出来的每一行前面加
  上"[q1]"这类标签，看真机日志时分得清是哪个子问题。

同时打到LLM接口上的请求数由llm_client.py里的全局上限控制（套餐写的是"3-4个
Agent并发"），线程池本身开得比它大，排队发生在那一层。
"""

import contextvars
import os
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor

_parallel = os.environ.get("ARKSEEK_PARALLEL", "1") != "0"
_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="arkseek-task")
_local = threading.local()


def parallel_enabled() -> bool:
    return _parallel


def set_parallel(on: bool) -> None:
    global _parallel
    _parallel = bool(on)


def _run(fn, args, kwargs, tag):
    prev_task, prev_tag = getattr(_local, "in_task", False), getattr(_local, "tag", None)
    _local.in_task = True
    if tag:
        _local.tag = f"{prev_tag}/{tag}" if prev_tag else tag
    try:
        return fn(*args, **kwargs)
    finally:
        _local.in_task, _local.tag = prev_task, prev_tag


def submit(fn, *args, tag: str = None, **kwargs) -> Future:
    if not _parallel or getattr(_local, "in_task", False):
        fut = Future()
        try:
            fut.set_result(_run(fn, args, kwargs, tag))
        except BaseException as e:  # 跟真正在线程里跑一样：异常存进Future，.result()时抛出
            fut.set_exception(e)
        return fut
    _install_tagged_stdout()
    parent_tag = getattr(_local, "tag", None)
    # 把当前contextvars带进线程池（分步计时的trace靠它传下去，见timing.py）
    ctx = contextvars.copy_context()

    def task():
        _local.tag = parent_tag
        return ctx.run(_run, fn, args, kwargs, tag)
    return _pool.submit(task)


def current_tag():
    """当前所在并行任务的标签（如"画像写入"、"q1"），不在任务里时是None。"""
    return getattr(_local, "tag", None)


class LockedCalls:
    """调用指定方法前先拿锁，其它属性原样透传。只拦从外面来的调用，对象内部
    方法互相调用不经过这里。methods=None表示所有方法都加锁。"""

    def __init__(self, obj, methods=None, lock=None):
        self._obj, self._methods = obj, methods
        self._lock = lock or threading.Lock()

    def __getattr__(self, name):
        attr = getattr(self._obj, name)
        if not callable(attr) or (self._methods is not None and name not in self._methods):
            return attr

        def locked(*args, **kwargs):
            with self._lock:
                return attr(*args, **kwargs)
        return locked


class _TaggedStdout:
    """任务线程里print出来的每一行加上"[标签] "前缀；别的线程原样输出。"""

    def __init__(self, stream):
        self._stream = stream
        self._line_start = threading.local()

    def write(self, text):
        tag = getattr(_local, "tag", None)
        if not tag or not text:
            return self._stream.write(text)
        at_start = getattr(self._line_start, "v", True)
        out = []
        for piece in text.splitlines(keepends=True):
            if at_start:
                out.append(f"[{tag}] ")
            out.append(piece)
            at_start = piece.endswith("\n")
        self._line_start.v = at_start
        return self._stream.write("".join(out))

    def __getattr__(self, name):
        return getattr(self._stream, name)


_install_lock = threading.Lock()


def _install_tagged_stdout():
    with _install_lock:
        if not isinstance(sys.stdout, _TaggedStdout):
            sys.stdout = _TaggedStdout(sys.stdout)
