# -*- coding: utf-8 -*-
"""
LLM调用抽象层 —— 阶段二 2.2（接LLM）的地基

对应方案文档里"要不要接LLM"这条路线分叉：2.1验证了完全不接LLM能做到什么程度，
2.2要验证接了LLM之后能不能解决2.1暴露出的语义理解局限（覆盖vs合并分不清、
L0超窗口没法压缩、回答只能"展示证据"不能"说人话"）。

【设计目标：调用方不关心具体接的是哪家LLM】
跟 dense_retriever.py / reranker.py 是同一个"可插拔后端"思路的延续，但这里的
"插拔"点不一样：dense_retriever/reranker 是"真实模型 vs 占位算法"二选一，装得上
就用真的、装不上就退化成弱一点但行为一致的替代品；这里是"具体接哪一家LLM
服务商"可能会换（用户明确说了先用国内MiniMax测试、以后可能换Claude），所以
本文件把"跟哪家LLM服务商打交道的细节"完全封装起来，对外只暴露一个函数：

    llm_client.chat(messages) -> Optional[str]

调用方（conversation_memory.py 等）永远只跟这一个函数打交道，不用关心是MiniMax
还是Claude、用的是什么SDK、base_url是什么。以后要换供应商，只需要在这个文件
里加一个新的 XxxLLMClient 类 + 改 build_llm_client() 里选哪个，业务代码一行都
不用动——这是用户明确提出的要求（"希望设计时能考虑到这一点的方便程度"）。

【返回 None 而不是抛异常，也不是返回假文本】
dense_retriever/reranker 装不上真实模型时，退化版还能返回一个"形状一样但质量
打折"的结果（TF-IDF向量、不做精排的原始排序），流水线可以直接往下走。但LLM
生成没有这种"退化但至少还能用"的中间状态——网络不通/没配key/调用报错时，
硬造一段占位文本反而是在冒充LLM说了话，比明确说"这次没生成成功"更危险。所以
chat() 在不可用时统一返回 None，把"要不要退回2.1的规则版逻辑"这个决定交给
调用方——这也是为什么2.2的L1事实抽取、L0摘要、L2/L3生成，都设计成"LLM能用就
用，不能用就退回2.1对应的规则版实现"，而不是直接报错崩溃：2.1的代码不是被
2.2淘汰的废弃品，而是2.2天然的降级路径。

【API Key 从哪里来：只认环境变量，绝不硬编码、绝不出现在任何会同步/提交的文件里】
key通过环境变量 MINIMAX_API_KEY 传入（`os.environ.get`），代码里不出现真实key，
本文件、multiturn_test_cases.json 等任何会被同步到坚果云、以后可能传到GitHub
上当作品集展示的文件里也不应该出现真实key。在自己电脑上，可以用系统环境变量
（Windows: `setx MINIMAX_API_KEY "sk-xxx"`，设完要开新的终端窗口才生效），也可以
在项目根目录建一个不会被分享出去的 `.env` 文件（配合 python-dotenv 读取，见
requirements.txt），二选一都行。如果这份代码以后要传到GitHub上，`.env` 必须
进 .gitignore。

【MiniMax 用的是 OpenAI 兼容接口，不用装 MiniMax 专属 SDK】
MiniMax 官方文档（platform.minimaxi.com）提供了跟 OpenAI Chat Completions 接口
形状完全一致的兼容endpoint，所以直接用业界最通用的 `openai` 这个pip包，把
base_url 指向 MiniMax 而不是 OpenAI 官方地址就行，不用为每家供应商各装一个
专属SDK——这也是为什么以后换Claude/别的家相对容易：只要对方也提供OpenAI兼容
接口（很多国内厂商都提供），甚至可以复用同一个 OpenAIStyleLLMClient 类，只是
换一下 base_url/api_key/model；Claude官方的Messages API格式跟OpenAI不完全一样，
真要换Claude的话会是一个新的类（AnthropicLLMClient），但对调用方而言依然只是
build_llm_client() 里的一行切换，chat()的输入输出形状不变。

    base_url = "https://api.minimax.cn/v1"（官方文档 2026年9月 版本，以后如果
    MiniMax改了地址或模型名，以你自己账号控制台看到的为准，这类接口地址不算
    需要每次都验证的"过时风险点"，但换了确实需要来改这里）
"""

import os
import re
from typing import List, Dict, Optional

from timing import span

# 【真实联调踩到的坑：推理模型会把"心里话"混进正文里】
# MiniMax-M3 是个推理模型，接口默认会把它的思考过程也放进 message.content 里，
# 用 <think>...</think> 包起来，正文答案跟在后面——第一次真实调用就实测到了：
# 问"用一句话介绍自己"，content 里先是一大段"用户让我...我应该..."的内心独白，
# 然后才是真正的自我介绍。如果不处理，L1事实抽取指望拿到干净的JSON、L2/L3
# 生成指望拿到一段能直接展示给用户的话，都会被这段思考过程污染。
# MiniMax官方文档说可以传 reasoning_split=True 把思考过程分流到独立的
# reasoning_content/reasoning_details 字段，content 只留最终答案——但文档同时
# 说明这个参数对 M2.x 系列"无论如何思考都会保留"，不保证在所有型号上都生效。
# 所以这里两道保险都上：请求时带上 reasoning_split，同时不管这个参数生不生效，
# 拿到 content 之后都用正则把 <think>...</think> 硬性剥离一遍——不依赖"某个
# API参数在所有模型上都按文档说的那样工作"这个假设，这也是精排sigmoid那次
# 踩坑给的同一个教训：接口的实际行为要拿真实调用验证，不能只读文档就当真。
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_TAG_RE.sub("", text).strip()


def _usage_note(resp) -> str:
    """计时用：这次调用输出了多少token、其中思考过程多少（有思考过程的模型，
    生成时间主要跟输出长度成正比）。接口没返回usage就返回空串。"""
    usage = getattr(resp, "usage", None)
    out = getattr(usage, "completion_tokens", None) if usage else None
    if out is None:
        return ""
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    # MiniMax reasoning_split：思考过程单独放在 reasoning_details。usage里的reasoning_tokens
    # 经常报0（实测思考了几千字也报0），所以以实际收到的思考文字为准
    msg = resp.choices[0].message if getattr(resp, "choices", None) else None
    rd = getattr(msg, "reasoning_details", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning_details")
    if rd:
        text = "".join((d.get("text") if isinstance(d, dict) else getattr(d, "text", "")) or "" for d in rd)
        return f"输出{out}token（思考过程{len(text)}字）"
    if reasoning:
        return f"输出{out}token（思考{reasoning}）"
    return f"输出{out}token"


# 【2026-09真机排查：MiniMax内容安全分类器拒绝，跟网络/鉴权/限流是完全
# 不同性质的失败】用户反复撞到`UnprocessableEntityError: Error code: 422`
# +`'type': 'unprocessable_entity_error', 'message': 'input new_sensitive
# (1026)'`这个错误——查了MiniMax官方错误码文档确认：1026="input new_
# sensitive"表示这次请求的输入内容被判定命中敏感内容，1027="output new_
# sensitive"表示模型本来想生成的回复内容被判定命中敏感内容；这是MiniMax
# 那一侧的内容安全分类器对单次请求的判断，不是账号被封、不是网络不通、
# 也不是余额/限流问题，官方文档给的应对方式明确是"修改输入内容，不要尝试
# 绕过安全限制"——原样重试同一次请求基本没有意义，大概率还会被拒。
#
# 真机测试里这类失败集中出现在查明日方舟wiki角色故事的场景（"能天使的
# 同事"这条链路查的德克萨斯/大帝/空/可颂几个角色的wiki正文），具体是
# 正文里哪个片段触发了分类器目前没有价值去逐字排查（分类器本身是黑盒，
# 排查成本高、收益低），但这类失败之前只是偶发，这轮真机测试里在同一个
# fan-out查询里连续命中了4次——加这个函数只是为了让日志里能一眼看出
# "这是内容安全拒绝"，不用每次看到调用失败就先怀疑是不是网络又断了、
# key是不是过期了，把排查方向从"网络/鉴权类问题"里筛出去。
_SENSITIVE_MARKERS = ("new_sensitive", "sensitive_error")


def _describe_llm_error(e: Exception) -> str:
    """把常见的LLM调用失败原因翻译成更好读的一句话，尤其是要把"内容安全
    分类器拒绝"这一类失败跟网络/鉴权/限流这些"重试可能有用"的失败区分
    开——具体动机见上面的模块级注释。不认识的异常类型原样保留
    `type(e).__name__: e`这个格式，不改变原有行为，只是新增一种更好读
    的翻译。
    """
    text = str(e)
    if any(marker in text for marker in _SENSITIVE_MARKERS):
        if "output" in text:
            return ("MiniMax内容安全分类器判定这次要生成的回复内容命中敏感内容"
                     "（错误码1027，output_sensitive）——不是网络/鉴权/余额问题，"
                     "原样重试大概率还会失败，官方建议是换一种表述而不是重试")
        return ("MiniMax内容安全分类器判定这次请求的输入内容命中敏感内容"
                 "（错误码1026，input_sensitive）——不是网络/鉴权/余额问题，"
                 "原样重试大概率还会失败，官方建议是换一种表述而不是重试")
    if type(e).__name__ == "RateLimitError" or "429" in text or "rate limit" in text.lower():
        return (f"被限流（{type(e).__name__}: {e}）——已按退避间隔自动重试"
                f"{_SDK_MAX_RETRIES}次仍失败。经常出现的话，把.env里的"
                f"ARKSEEK_LLM_MAX_CONCURRENCY调小")
    if type(e).__name__ in ("APITimeoutError", "LLMStalled"):
        return (f"连续{LLM_STALL_TIMEOUT:.0f}秒没有收到任何数据（接口卡住），重试{_TIMEOUT_RETRIES}次后仍卡住"
                f"（可在.env里用ARKSEEK_LLM_STALL_TIMEOUT调）")
    if type(e).__name__ == "LLMEmpty":
        return f"接口返回了空回复，重试{_TIMEOUT_RETRIES}次后仍是空的"
    if type(e).__name__ == "LLMOvertime":
        return (f"一直在生成，但超过了{LLM_MAX_DURATION:.0f}秒安全上限"
                f"（可在.env里用ARKSEEK_LLM_MAX_DURATION调）")
    return f"{type(e).__name__}: {e}"

try:
    # 可选依赖：装了 python-dotenv 就自动从项目根目录的 .env 文件加载环境变量，
    # 不用每次开新终端都手动 setx/export。找不到 .env 或者没装这个包都不报错——
    # 环境变量本来就可以用系统级设置的方式提供，.env 只是更方便的一种途径。
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# 【同时打到LLM接口上的请求数上限（2026-09-28）】并行之后（qq_bot不同对话并行 +
# 一条消息内推测执行/子问题并行），同一时刻可能有十几个请求。MiniMax套餐页只写了
# "可同时支持3-4个Agent并发运行"，没写超出会怎样，所以默认同时最多4个、多的在本机
# 排队（不报错，只是等）。以后换套餐改.env里的ARKSEEK_LLM_MAX_CONCURRENCY即可。
# 被限流（HTTP 429）时按退避间隔自动重试3次；仍然失败会在日志里写明"被限流"，
# 见_describe_llm_error()。
#
# 【请求超时（2026-09-30，回复慢）】分步计时里一次字数压缩卡了265秒：原来没设超时，
# openai SDK默认等600秒、超时后还会再重试。QQ被动回复5分钟就过期，所以每次请求
# 最多等LLM_TIMEOUT秒（当时默认60；2026-10-01改成流式后默认120，见下一节），超时/连不上只重试1次，
# 还不行就返回None，交给调用方现有的降级逻辑（压缩失败保留原回答、路由失败走
# 规则等）。限流和超时分开计次：SDK自带的重试不区分两者，所以关掉SDK重试，
# 在chat()里自己按错误类型重试（tenacity之类重试库的常见做法）。
import threading
import time

LLM_MAX_CONCURRENCY = max(1, int(os.environ.get("ARKSEEK_LLM_MAX_CONCURRENCY", "4")))
_llm_slots = threading.BoundedSemaphore(LLM_MAX_CONCURRENCY)
_SDK_MAX_RETRIES = 3          # 被限流时的重试次数（沿用原来的名字，_describe_llm_error里引用）
_TIMEOUT_RETRIES = 1          # 卡住/连不上/服务端5xx时的重试次数
LLM_TIMEOUT = float(os.environ.get("ARKSEEK_LLM_TIMEOUT", "120"))
LLM_MAX_DURATION = float(os.environ.get("ARKSEEK_LLM_MAX_DURATION", "240"))

# 【区分"卡住"和"在思考"（2026-10-01）】真机两次超时：wiki生成连续两次各等满
# 60秒；单独测一个只输出一百多token的压缩请求也卡满了60秒——接口偶尔会一个字
# 都不回。而推理模型正常思考时间也长（1700token的回答要20秒左右）。原来非流式
# 请求只能等整个回复，两者分不清，只能都等满60秒再重试。现在改成流式接收（思考
# 过程也是边生成边回传）：连续LLM_STALL_TIMEOUT秒一个字节都没收到=卡住，马上
# 重试；一直有数据回来=在正常生成，等到LLM_TIMEOUT总时长上限为止，超过总上限
# 不重试（还在生成的请求重来一遍多半一样慢）。跟流式服务里常见的"首包/包间超时
# +总时长上限"两级超时是一个做法。
# 总上限从60秒放宽到120秒（用户2026-10-01：慢一点可以，但不要因为超时让回答降级）：
# 推理模型正常思考偶尔超过60秒（实测生成一题62秒），截断后只能退回降级展示。首包实测2~3秒（含1.3万字prompt的处理）。
LLM_STALL_TIMEOUT = float(os.environ.get("ARKSEEK_LLM_STALL_TIMEOUT", "15"))


class LLMStalled(Exception):
    """连续LLM_STALL_TIMEOUT秒没收到任何数据。"""


class LLMOvertime(Exception):
    """活跃流超过独立安全上限LLM_MAX_DURATION；LLM_TIMEOUT仅提示慢请求。"""


class LLMEmpty(Exception):
    """正常结束了，但正文是空的（2026-10-01实测：自身知识回答输出了15个token、正文为空，
    调用方拿到None，最后只剩"我帮你查一下～"这种占位语）。当作一次可重试的失败。"""


def _retry_kind(e: Exception) -> Optional[str]:
    """这个异常要不要重试：'rate'（限流，退避后重试）/'transient'（超时、连不上、
    服务端5xx，马上重试）/None（鉴权、参数、内容安全等，重试也没用）。"""
    name = type(e).__name__
    if name == "RateLimitError":
        return "rate"
    if name in ("APITimeoutError", "APIConnectionError", "InternalServerError", "LLMStalled", "LLMEmpty"):
        return "transient"
    return None


class NoLLMClient:
    """LLM不可用时的占位版：不生成任何文本，只诚实地告诉调用方"这次没有LLM可用"。
    对应 reranker.py 里 NoOpReranker 的角色，但形状不同——见模块开头的说明，
    生成任务没有"降级但至少还能用"的中间态，只有"能生成"和"不能生成"两种状态。
    """

    name = "LLM不可用（占位版，无生成能力）"
    is_real = False

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        return None


class OpenAIStyleLLMClient:
    """用 openai SDK 调用任何"OpenAI Chat Completions 兼容"接口的LLM服务商。
    当前配置成打 MiniMax，以后要换成别的OpenAI兼容供应商（很多国内厂商的
    OpenAI兼容endpoint都是这个形状），理论上只用改 base_url/api_key_env/model
    这三个参数，不用改这个类本身。
    """

    is_real = True

    def __init__(self, provider_label: str, base_url: str, api_key_env: str,
                 model: str, api_key: Optional[str] = None):
        from openai import OpenAI  # 延迟导入，装不上时不影响其他模块

        self.provider_label = provider_label
        self.model = model
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(
                f"环境变量 {api_key_env} 没有设置——去对应平台的控制台创建一个API "
                f"key，然后在自己电脑上设成环境变量，不要把key直接写进代码或任何"
                f"会同步/提交的文件里。"
            )
        # 重试由chat()自己按错误类型做（见模块里"请求超时"一节），SDK不再重试
        # 流式接收，timeout作用于每一次读：连续这么久没数据就抛超时（=卡住）
        self.client = OpenAI(api_key=key, base_url=base_url, max_retries=0, timeout=LLM_STALL_TIMEOUT)
        self.name = f"{provider_label}（{model}，真实LLM）"

    def _stream_once(self, messages, extra_body, kwargs, caller=None):
        """流式发一次请求，拼成跟非流式一样形状的返回值（choices[0].message.content/
        reasoning_details、usage）。保留停滞检测，120秒默认只提示，240秒默认安全截断。"""
        from types import SimpleNamespace
        start = time.perf_counter()
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, extra_body=extra_body,
            stream=True, stream_options={"include_usage": True}, **kwargs)
        content, reasoning, usage = [], [], None
        completed = False
        slow_reported = False
        try:
            for chunk in stream:
                if chunk.usage:
                    usage = chunk.usage
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        content.append(delta.content)
                    rc = (delta.model_extra or {}).get("reasoning_content")
                    if rc:
                        reasoning.append(rc)
                elapsed = time.perf_counter() - start
                if not slow_reported and LLM_TIMEOUT > 0 and elapsed > LLM_TIMEOUT:
                    print(f"[llm_client] {caller or '未标注'}已运行{elapsed:.1f}秒，仍有流式数据，继续等待完整回答。")
                    slow_reported = True
                if LLM_MAX_DURATION > 0 and elapsed > LLM_MAX_DURATION:
                    raise LLMOvertime()
            completed = True
        except (LLMOvertime, LLMStalled):
            raise
        except Exception as e:
            # 读数据途中的超时是httpx的ReadTimeout（不会被SDK包成APITimeoutError）
            if "Timeout" in type(e).__name__:
                raise LLMStalled() from e
            raise
        finally:
            stream.close()
            status = "完整结束" if completed else "中断"
            print(f"[llm_client] {caller or '未标注'}流式请求{status}：{time.perf_counter() - start:.1f}秒，"
                  f"正文{sum(map(len, content))}字，思考{sum(map(len, reasoning))}字。")
        message = SimpleNamespace(content="".join(content),
                                  reasoning_details=[{"text": "".join(reasoning)}] if reasoning else None,
                                  model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    def _create_with_retry(self, messages, extra_body, caller, kwargs):
        """发一次请求；限流退避重试_SDK_MAX_RETRIES次，超时/连不上重试_TIMEOUT_RETRIES次，
        其它错误直接抛给chat()。每次尝试各记一个计时span。"""
        prompt_chars = sum(len(m.get("content") or "") for m in messages)
        rate_left, transient_left, attempt = _SDK_MAX_RETRIES, _TIMEOUT_RETRIES, 0
        while True:
            attempt += 1
            label = f"{caller or '未标注'}，prompt {prompt_chars}字" + (f"，第{attempt}次尝试" if attempt > 1 else "")
            try:
                with span("LLM", label) as info:
                    resp = self._stream_once(messages, extra_body, kwargs, caller=caller)
                    info["after"] = _usage_note(resp)
                    if not _strip_thinking(resp.choices[0].message.content or ""):
                        raise LLMEmpty()
                    return resp
            except Exception as e:
                kind = _retry_kind(e)
                if kind == "rate" and rate_left > 0:
                    rate_left -= 1
                    wait = 2 ** (_SDK_MAX_RETRIES - rate_left - 1)  # 1、2、4秒
                    print(f"[llm_client] {caller or '这次调用'}被限流，{wait}秒后重试")
                    time.sleep(wait)
                elif kind == "transient" and transient_left > 0:
                    transient_left -= 1
                    why = {"LLMEmpty": "接口返回了空回复",
                           "APIConnectionError": "连不上接口",
                           "InternalServerError": "接口服务端出错"}.get(
                        type(e).__name__, f"连续{LLM_STALL_TIMEOUT:.0f}秒没收到数据，算卡住")
                    print(f"[llm_client] {caller or '这次调用'}{type(e).__name__}（{why}），重试一次")
                else:
                    raise

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        # 【2026-09真机排查：失败提示只说"调用失败"，不说是哪一步调用失败】
        # 一次多跳查询里decompose()/resolve_for_retrieval()/route_query()/
        # llm_generate_answer()（wiki一次、local_kb可能又一次）/synthesize()/
        # detect_answer_candidates()都可能各调一次chat()，本身互相之间没有
        # 任何区分标记——真机排查"W的异格"/"苇草的异格"这两个case时，光看
        # 打印顺序去反推"到底是哪一次调用失败的"很容易搞错（需要同时对照
        # 前后的其它打印、还原每个函数的完整调用条件），加一个可选的
        # `caller`标签，纯粹是给日志读者看的，不会拼进发给MiniMax的请求
        # 内容本身。调用方不传就是空字符串，跟之前完全一致。
        caller = kwargs.pop("caller", "")
        try:
            # reasoning_split 是MiniMax的扩展参数，openai SDK不认识，走 extra_body
            # 透传给底层HTTP请求；调用方如果自己传了 extra_body，尊重调用方的值，
            # 不覆盖。
            extra_body = kwargs.pop("extra_body", {}) or {}
            extra_body.setdefault("reasoning_split", True)

            if not _llm_slots.acquire(blocking=False):
                print(f"[llm_client] 同时在等回复的请求已达上限{LLM_MAX_CONCURRENCY}个，"
                      f"{caller or '这次调用'}排队")
                with span("LLM排队", caller):
                    _llm_slots.acquire()
            try:
                response = self._create_with_retry(messages, extra_body, caller, kwargs)
            finally:
                _llm_slots.release()
            content = response.choices[0].message.content or ""
            return _strip_thinking(content)
        except Exception as e:
            # 调用失败（网络不通、key过期、余额不足、限流、内容安全分类器拒绝……）
            # 不让整个记忆系统崩溃——打印出来方便排查，返回None交给调用方决定
            # 要不要退回规则版。`_describe_llm_error()`会把内容安全类失败翻译成
            # 更好读、更容易跟网络/鉴权类问题区分开的一句话，见该函数的说明；
            # `caller`（如果调用方传了）标出这是哪一步的调用，见上面的说明。
            caller_note = f"来自{caller}，" if caller else ""
            print(f"[llm_client] {self.provider_label} 调用失败"
                  f"（{caller_note}{_describe_llm_error(e)}），本次生成不可用。")
            return None


def build_llm_client(provider: Optional[str] = None) -> "NoLLMClient | OpenAIStyleLLMClient":
    """按配置选一个LLM后端，装不上/没配key就退回 NoLLMClient 占位版。

    provider 不传时读环境变量 LLM_PROVIDER，都没有就默认"minimax"——这一行是
    整个项目里唯一需要因为"换LLM供应商"而修改的地方（比如以后要换Claude，
    这里加一个 provider == "claude" 的分支，new一个 AnthropicLLMClient）。
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "minimax")

    try:
        if provider == "minimax":
            model = os.environ.get("MINIMAX_MODEL", "MiniMax-M2")
            client = OpenAIStyleLLMClient(
                provider_label="MiniMax",
                base_url="https://api.minimax.cn/v1",
                api_key_env="MINIMAX_API_KEY",
                model=model,
            )
        else:
            raise ValueError(f"未知的 LLM_PROVIDER: {provider}")

        print(f"[llm_client] 已配置真实LLM：{client.name}")
        return client
    except Exception as e:
        print(f"[llm_client] 无法初始化LLM后端（{type(e).__name__}: {e}），"
              f"退回不接LLM的占位版——2.2的各层会自动退回2.1对应的规则版实现。"
              f"设好 MINIMAX_API_KEY 环境变量后会自动切换。")
        return NoLLMClient()


if __name__ == "__main__":
    client = build_llm_client()
    print(f"\n当前后端：{client.name}（is_real={client.is_real}）\n")

    reply = client.chat([
        {"role": "user", "content": "用一句话介绍一下你自己。"},
    ])
    if reply is None:
        print("没有拿到LLM回复（大概率是没配key，或者当前环境访问不了外网）——"
              "这是预期内的降级行为，不是bug。")
    else:
        print(f"LLM回复：{reply}")
