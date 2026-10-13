# -*- coding: utf-8 -*-
"""
可编排的假 LLM 客户端 —— 给 run_orchestration_regression_tests.py 用的测试替身。

【这个模块是为了补哪个洞】
2026-09架构审查发现：`conversation_memory.py`/`query_planner.py`/`wiki_router.py`
这个编排层，在真机测试的 26 轮左右的修复（HANDOFF.md 里的①-㉖）背后，
唯一的"没有引入回归"证据是 HANDOFF 里反复提到的"沙盒结构性验证"——但那些
`test_fanout_sandbox.py`/`test_fanout_history_competing_sandbox.py`之类的脚本
从来没有被提交进这个坚果云同步文件夹，每次都是在某次会话的云端临时工作区
里现写现跑、验证完就随会话一起消失。这意味着"之前修过的坑有没有被这次改动
悄悄弄回去"这件事，除了把 HANDOFF 从头读一遍、凭记忆判断之外，没有任何
可以在几秒内重新跑一遍的东西——这是这次审查里认定的唯一一处真正的结构性
风险（不是代码写得乱，是测试基础设施没跟上编排层的复杂度增长）。

这个文件就是把那些"每次现写、写完就丢"的假 LLM 客户端固化成一个正式、
可复用、留在项目里的模块。

【这一层测试能验证什么，不能验证什么——务必如实分清楚，不要混淆】
这套脚手架验证的是"接线/编排逻辑对不对"（wiring regression）：给定LLM
在某一步"说了什么"，代码有没有按预期的顺序调用、有没有把正确的参数往下传、
分支判断对不对、失败/None 这条降级路径走不走得通。它**不能**验证"真实
LLM 在这一步会不会真的说对"（semantic regression）——比如 decompose() 的
prompt 会不会真的让 MiniMax 把两种问法拆成同一份 plan、route_query() 的
prompt 会不会真的把"它现在进展到哪一步了"分类对，这些语义层面的问题，
这里全部靠 ScriptedLLMClient 按脚本"喂答案"绕过去了，跟改没改 prompt
文字完全无关，测不出来，也不该测——那是 run_*_demo.py 这批脚本（需要真实
网络+真实 API key，只能在你自己电脑上跑）的职责，两者互补，不是谁取代谁。
一句话概括这条边界：**这里测"代码收到 X 之后做什么"，不测"LLM会不会真的
给出 X"**。

【为什么不用现成的 mock 框架（unittest.mock.MagicMock）硬凑】
`client.chat()`在一次完整的多跳/编排流程里会被依次调用好几次，每次的
调用方（decompose/route_query/resolve_for_retrieval/synthesize/...）不同、
期望返回的内容形状也不同（有的要JSON，有的要纯文本）——用 MagicMock 的
`side_effect=[...]`列表也能做到"按顺序返回不同值"，但拿不到"这次调用
到底是谁发起的、跟我预期的是不是同一步"这个校验，调用顺序一旦被这次改动
悄悄打乱，MagicMock只会把错位的返回值原样喂给错的调用方，得到一个看似
"跑通了"、实际上文不对题的结果，反而比直接报错更难排查。这里的
ScriptedLLMClient 强制要求"这次调用的 caller 必须跟脚本预期的一致"，
错位时立刻用能定位到第几步的报错声明失败——这一点本身就是全项目"防
幻觉"设计哲学（宁可诚实报错，也不要悄悄用错的东西凑一个看起来对的结果）
在测试基础设施这一层的体现。
"""

import json
import threading
import time


class UnexpectedCallError(AssertionError):
    """脚本跟实际调用对不上时抛出——继承 AssertionError，这样即使某条测试
    用例忘了包 try/except，pytest 风格的 assert 失败捕获逻辑也能正常接住。
    """


class ScriptedLLMClient:
    """`llm_client.py` 里 `OpenAIStyleLLMClient`/`NoLLMClient` 的测试替身，
    对外暴露完全相同的接口（`is_real` 属性 + `chat(messages, caller=None)`
    方法），所以能原样传给 `ConversationSession(..., llm_client=...)`，不需要
    改动任何被测代码。

    `script`：一个列表，按调用顺序排列，每一项是一个 dict，可以有：
      - "caller"：期望这次调用的 caller 精确等于这个字符串（比如
        "route_query()"）。传 None（或不写）表示不校验 caller 具体是什么，
        只按顺序消费一条脚本——不建议在关键路径上这么用，精确校验才能真正
        捕捉"调用顺序被打乱了"这类回归。
      - "caller_prefix"：期望这次调用的 caller 以这个前缀开头，用来匹配
        `llm_generate_answer()`那种带动态后缀的 caller（比如
        "llm_generate_answer(wiki:能天使)"，后缀里的页面标题每次都不一样，
        没法精确匹配，只能校验前缀）。"caller"和"caller_prefix"同时给会
        优先用"caller"精确匹配。
      - "response"：这次调用该返回什么。dict 会自动 json.dumps() 成字符串
        （给期望 JSON 的调用方用，比如 route_query()/decompose()）；字符串
        原样返回（给期望纯文本的调用方用，比如 resolve_for_retrieval()/
        llm_generate_answer()/synthesize()）；None 表示模拟"这次LLM调用
        失败"（对应真实 `OpenAIStyleLLMClient.chat()` 出错时返回 None 的
        约定），用来测降级路径。

    `calls`：每次 `chat()` 被调用后追加一条 {"caller":..., "messages":...}
    记录，供测试事后断言"到底调用了几次""每次传的内容里有没有出现预期的
    关键词"（比如验证 resolve_for_retrieval() 改写后的文本真的被用于下一步
    检索，而不是停留在被丢弃的返回值里）。

    调用次数对不上（脚本提前用完，或者跑完测试脚本还有剩）都会报错，
    分别在 `chat()` 内部（多调用）和 `assert_exhausted()`（少调用，测试用例
    结尾主动调用一次）——两个方向都要查，只查一个方向会漏掉"这次改动
    让某条路径少打了一次该打的 LLM 调用"这类回归。
    """

    is_real = True
    name = "scripted-fake"

    def __init__(self, script):
        self._script = list(script)
        self._consumed = 0
        self.calls = []

    def chat(self, messages, caller=None):
        self.calls.append({"caller": caller, "messages": messages})

        if not self._script:
            raise UnexpectedCallError(
                f"[fake_llm_client] 脚本已经消费完（前面共{self._consumed}次调用），"
                f"但代码又发起了一次新调用：caller={caller!r}。\n"
                f"这通常意味着这次改动让某条路径多打了一次不该打的LLM调用——"
                f"如果这确实是这次改动想要的新行为，去测试用例里补一条脚本；"
                f"如果不是，说明改动引入了多余的调用（真实场景下意味着多花钱、"
                f"多一次内容安全拒绝的风险）。"
            )

        step = self._script.pop(0)
        self._consumed += 1
        position = self._consumed

        expected_caller = step.get("caller")
        expected_prefix = step.get("caller_prefix")
        if expected_caller is not None:
            if caller != expected_caller:
                raise UnexpectedCallError(
                    f"[fake_llm_client] 第{position}次调用期望 caller={expected_caller!r}，"
                    f"实际收到 caller={caller!r}。\n"
                    f"这通常意味着调用顺序/调用对象变了——先确认这是不是这次改动的"
                    f"预期结果：是的话更新脚本，不是的话说明改动引入了调用顺序上的回归。"
                )
        elif expected_prefix is not None:
            if not (caller or "").startswith(expected_prefix):
                raise UnexpectedCallError(
                    f"[fake_llm_client] 第{position}次调用期望 caller 以"
                    f"{expected_prefix!r}开头，实际收到 caller={caller!r}。"
                )

        response = step.get("response")
        if response is None:
            return None
        if isinstance(response, dict):
            return json.dumps(response, ensure_ascii=False)
        return response

    def assert_exhausted(self):
        """测试用例结尾调用：脚本里还有没被消费的条目，说明预期中的某次
        LLM调用实际没有发生——同样是一种回归，跟"多调用"同等重要，缺一个
        方向都会漏掉一半的问题。
        """
        if self._script:
            remaining = [
                s.get("caller") or f"prefix={s.get('caller_prefix')!r}"
                for s in self._script
            ]
            raise AssertionError(
                f"[fake_llm_client] 脚本还剩{len(self._script)}条没被消费："
                f"{remaining}。\n"
                f"这通常意味着这次改动让某条路径少打了一次预期中的LLM调用——"
                f"确认是不是漏了一步；如果这条分支现在确实不需要这次调用了，"
                f"说明这个测试用例本身该跟着精简，不是被测代码有问题。"
            )

    def caller_sequence(self):
        """按顺序返回目前为止每次调用的 caller，方便测试用例整体断言调用
        顺序（比如一次性确认"这一整轮总共调用了这几次，顺序是这样"），
        比逐条翻 `self.calls` 更省事。
        """
        return [c["caller"] for c in self.calls]


class KeyedLLMClient:
    """并行版的测试替身（2026-09-28，推测执行+子问题并行之后）。

    ScriptedLLMClient按调用顺序消费剧本；并行之后几次调用谁先到是随机的，按顺序
    认领就会张冠李戴。这里改成"按调用方+请求内容认领"：每次调用找第一条还没被
    用过、caller对得上、并且"when"（可选）出现在这次请求的最后一条user消息里的
    规则，返回它的response。多调用/少调用照样报错，只是不再要求先后顺序。

    `rules`：列表，每一项是dict：
      - "caller" / "caller_prefix"：同ScriptedLLMClient；
      - "when"：可选，请求内容里要包含的字符串——同一个调用方被并行调了好几次
        （比如几个子问题各自route_query()）时，靠它区分哪次给哪个回复；
      - "response"：同ScriptedLLMClient（dict自动转JSON，None模拟调用失败）；
      - "delay"：可选，返回前睡几秒，模拟网络等待，用来测"是不是真的同时在等"；
      - "optional"：可选，True表示这条允许没被用上（推测执行里被丢掉、可能根本
        没开跑的调用），assert_exhausted()不因为它报错。

    `calls`：每次调用记一条{"caller", "text", "start", "end"}（时间是time.monotonic()），
    `max_active`：同一时刻最多有几个调用在"等回复"——测试靠这两个断言确实并行了。
    """

    is_real = True
    name = "keyed-fake"

    def __init__(self, rules):
        self._rules = [dict(r) for r in rules]
        self._used = [False] * len(self._rules)
        self._lock = threading.Lock()
        self._active = 0
        self.max_active = 0
        self.calls = []

    @staticmethod
    def _matches(rule, caller, text):
        if rule.get("caller") is not None and rule["caller"] != caller:
            return False
        if rule.get("caller") is None and rule.get("caller_prefix") is not None \
                and not (caller or "").startswith(rule["caller_prefix"]):
            return False
        return rule.get("when") is None or rule["when"] in text

    def chat(self, messages, caller=None):
        users = [m.get("content", "") for m in messages if m.get("role") == "user"]
        text = users[-1] if users else ""
        with self._lock:
            idx = next((i for i, r in enumerate(self._rules)
                        if not self._used[i] and self._matches(r, caller, text)), None)
            record = {"caller": caller, "text": text, "start": time.monotonic(), "end": None}
            self.calls.append(record)
            if idx is None:
                raise UnexpectedCallError(
                    f"[fake_llm_client] 没有能认领这次调用的规则：caller={caller!r}，"
                    f"请求内容开头：{text[:80]!r}")
            self._used[idx] = True
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        rule = self._rules[idx]
        try:
            if rule.get("delay"):
                time.sleep(rule["delay"])
        finally:
            with self._lock:
                self._active -= 1
                record["end"] = time.monotonic()
        response = rule.get("response")
        if response is None:
            return None
        if isinstance(response, dict):
            return json.dumps(response, ensure_ascii=False)
        return response

    def assert_exhausted(self):
        left = [r.get("caller") or f"prefix={r.get('caller_prefix')!r}"
                for r, used in zip(self._rules, self._used) if not used and not r.get("optional")]
        if left:
            raise AssertionError(f"[fake_llm_client] 还有{len(left)}条必需的规则没被用上：{left}")

    def calls_of(self, caller):
        return [c for c in self.calls if c["caller"] == caller]
