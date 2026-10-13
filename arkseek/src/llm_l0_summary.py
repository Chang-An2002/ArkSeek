# -*- coding: utf-8 -*-
"""
L0会话窗口溢出摘要的LLM实现 —— 阶段二 2.2（接LLM）

对应 conversation_memory.py 模块开头"L0：最近window_size轮"那条注释里留下的
一句话——2.1的L0窗口"目前只用于展示，没有真正喂给谁"，等2.2接了LLM才会真正
派上用场。这个文件就是让它派上用场的第一步：窗口之外的历史不再是"只能靠L2
精确检索翻回来"这一种去处，还多一份持续滚动更新的摘要，为将来需要往prompt
里塞"大致记得聊过什么"这种粗粒度上下文的场景做准备（比如L2/L3生成式回答
如果以后想让语气更连贯，可以把这份摘要也拼进prompt——当前版本还没有任何
调用方消费这份摘要，只是先把"生成+存储"这一步做对、做稳，如实记录这一点，
不夸大它现在的实际作用）。

【滚动摘要，不是每次从头重新摘要】
每次窗口往前滑动一轮，理论上都多了一轮"刚滑出窗口"的内容需要并入摘要。如果
每次都拿全部历史重新生成一遍摘要，会随着对话变长而越来越贵（每轮的调用输入
线性增长），而且没有必要——摘要真正需要的是"把新滑出的这一小段内容，合并
进已经压缩好的旧摘要"，不需要重新读一遍已经压缩过的原文。所以这里做的是
增量式滚动：LLM每次只看"旧摘要（已经很短）+ 新滑出的这几轮原文（也不长）"，
输出一份合并后的新摘要，成本不随对话总长度增长。

【不可用时怎么办：直接不生成摘要，不是它的降级路径】
和 llm_l1.py 一样，chat()调用失败/LLM不可用时返回None，调用方
（conversation_memory.py 的 _maybe_update_l0_summary()）看到None就什么都不做——
不写入摘要表、不报错、也不尝试用规则拼一份"假摘要"。2.1原本就没有摘要这个
概念，所以这里没有"规则版兜底"可退，None就等于"这轮没能力生成摘要，摘要
保持上一次的状态"，跟L1能退回regex不同，是彻头彻尾的2.2专属能力，不可用时
就是没有，如实记录。

【真机测试发现的真实问题：LLM在摘要文本里夹了没转义的引号，导致json.loads
每轮都失败、摘要卡住不动】用户真机聊天时发现终端反复刷一模一样的"LLM返回
的内容不是预期的JSON格式"提示，还夹在正常聊天中间，观感很乱。查原始输出
发现根因：LLM生成的摘要文本里提到"关注角色"真言"等技能机制"，把干员名
"真言"用中文引号包起来了，但这两个引号在JSON字符串里没有被转义成`\"`，
导致整个JSON在summary字段的值中间就被提前截断——`json.loads()`严格解析
必然失败。因为解析失败，`_maybe_update_l0_summary()`不会推进
`summarized_up_to`，下一轮又会带着完全相同的"新滑出内容"重新请求一次
摘要更新，LLM又大概率给出同样的错误格式，摘要就卡在原地反复重试，每轮
多花一次注定失败的LLM调用，还每轮都刷一遍一模一样的警告——这是这次真机
测试暴露的真实资源浪费+观感问题，不是"少见的边界情况"。

修复思路不是要求LLM更听话（prompt已经写了"不要用markdown代码块包裹"这类
格式要求，但没办法保证它输出的自然语言内容里不出现引号——这类内容本来就
可能提到专有名词、引用别人的话），而是让解析这一步更宽容：`json.loads()`
失败时，不直接认输，而是用一个正则兜底提取——因为这里的JSON形状很简单
（只有一个summary字段，且是整个对象的最后一个字段），可以直接找"summary"
键之后、到整个字符串末尾`"}`之前的所有内容当成摘要正文，不要求这段内容
内部的引号必须转义。这个思路只适用于"最后一个字段是字符串"这种简单形状，
不是一个通用的JSON修复方案，`llm_l1.py`那种"多个字段/数组"的复杂形状没有
用同样的技巧，等真的观察到同类失败再针对性处理。

【2026-09真机排查发现：这一步比其它LLM调用更容易撞上内容安全分类器，
加了一次"精简重试"】排查"苇草的异格"那次误诊时（见下面`llm_update_
summary()`文档字符串），确认了这里确实会被MiniMax内容安全分类器拒绝
（错误码1026）——不难理解：这里要压缩的是刚滑出窗口的几轮完整对话
原文，如果这几轮里恰好包含明日方舟wiki正文生成的、带较多剧情/战斗
描述的长回答（这类内容本来就在⑫里确认过更容易触发分类器），原样整段
转发给这次摘要调用当输入，触发概率不会比生成那次低，可能更高（因为
一次要打包好几轮）。之前的处理方式只是"失败就不刷新摘要，如实记录"
（见下面`_maybe_update_l0_summary()`调用方的说明），这次真机测试追问
"如何让这部分正常完成"后，加了一版真正尝试补救的重试：MiniMax官方
自己的建议是"修改输入内容，不要原样重试"，所以第一次完整版被拒后，
不是原样重试，而是把每一轮原文都截断成开头一小段再试一次——`_condense_
row_text()`只保留每轮文字的前`max_len`个字符，砍掉后面最容易带出直接
引用/战斗数值描述这类细节的部分，用一次内容明显更短、更不像完整叙事
的表述重试。这是一个纯粹基于长度截断的启发式策略，不是真的理解了分类器
在拒绝什么、也不保证一定能让这次重试通过（分类器本身是黑盒，短版本
里如果开头几个字就撞了同样会失败）——但相比"完全不试第二次"，这是
成本很低、符合官方建议方向的一次额外尝试，重试也失败时行为跟以前完全
一样：返回None，调用方沿用旧摘要，不影响本轮已经生成的回答。

【2026-09真机测试发现的设计问题：prompt写的过滤标准跟模块自己声明的定位
不一致，已参照业界惯例改成一视同仁压缩】用户真机测试时发现：连续问了好几
个明日方舟角色的细节都被写进了摘要（甚至具体到技能数值），但后来连续问的
几个旅游城市介绍问题完全没被摘要收录——一度以为是bug。查证后发现这其实是
`_SYSTEM_PROMPT`一直以来的字面规则在正常工作："只保留用户身份/偏好/决定
这类信息"，旅游问答显然不属于这几类，被正常过滤掉了；但反过来看，游戏角色
的具体数值细节其实也不是"关于用户本人"的信息，会被保留更可能是因为这类
内容是一个一个陆续问出来的，每次滚动合并时"把旧摘要里仍然成立的信息保留
进去"这条指令让它们逐次沉淀了下去，而旅游问题是连续问的、可能在同一批"新
对话"里被一次性判定为不相关直接跳过，从未真正进入过摘要——不是一条能讲
清楚的规则，是路径依赖导致的不一致。

用户问"业界的合理设计是怎样的"，查证后确认：几乎所有主流实现都把这里做的
事情拆成两个独立的层，而不是用一个组件同时干两件事——LangChain经典的
`ConversationSummaryMemory`/`ConversationSummaryBufferMemory`默认就是
对整段对话不加区分地压缩（目的是保持话题连贯，不是筛选"关于用户"的信息）；
Letta/MemGPT把"core memory"（人物设定+用户信息，选择性、由agent自己
管理）和"recall memory"（完整对话历史，全量可检索）明确拆成两个不同组件；
Anthropic自己的"compaction"功能（长对话接近上下文上限时的整体摘要）同样
是不加筛选的连续性压缩，跟专门存放持久化知识的"Memory tool"是两个独立
机制；学术界的MemoryBank论文也是把"递归连续性摘要"和"用户画像抽取"分开
做的两个组件。这几乎是一个业界公认的架构划分：**连续性/话题记忆**（不加
筛选、压缩整段对话，服务于"别忘了刚才聊到哪儿了"）和**持久化用户事实记忆**
（有选择、只存跟这个人相关的稳定信息，服务于"记住这个人是谁"）是两个不同
目的的组件，不应该用同一份逻辑处理。

回头看这个项目自己的架构，其实从模块开头第一段就已经把L0摘要定位成"大致
记得聊过什么"这种连续性/话题记忆的角色，L1（用户画像）才是那个"记住这个
人是谁"的选择性持久化事实记忆——`_SYSTEM_PROMPT`最初写成"只保留用户身份/
偏好/决定"，其实是错误地把L1该干的活搬到了L0里，两层职责重叠了，这才是
旅游问题被过滤、游戏角色细节被保留这种不一致现象的真正根源，不是哪次判断
偶然出了错。已按业界这个惯例改写`_SYSTEM_PROMPT`：不再区分内容是不是"关于
用户本人"，只要是双方实际讨论过的具体话题/实体/结论就一视同仁压缩保留；
篇幅上限从150字放宽到约300字（覆盖范围变宽了，硬卡在150字容易内容挤不下），
并且明确要求"超出预算时优先精简旧细节、不要无限累积"，直接针对之前观察到
的"摘要越滚越长、早就超过150字限制"这个真实现象。这处改动只是prompt文字
调整，沙盒里没有真实LLM可验证语义效果，需要用户在自己电脑上重新测几轮
（同时问几个不同领域的具体问题）、用`/summary`命令确认新话题是不是真的
被一视同仁地收进去了。
"""

import re
from typing import Optional

from llm_json import strip_json_fence, try_parse_json

# 兜底正则：json.loads()因为字符串内部有未转义引号而失败时，尝试直接从
# "summary"这个键之后、到最后一个"}"之前抓取原始文本当摘要——只对"这是
# 对象里最后一个字段"这种简单形状有效，见模块开头"真机测试发现的真实问题"。
_SUMMARY_FALLBACK_RE = re.compile(r'"summary"\s*:\s*"(.*)"\s*\}\s*$', re.DOTALL)

_SYSTEM_PROMPT = """\
你是一个负责维护"对话摘要"的组件，目的是帮对话保持连贯——大致记得"到目前为止
聊过什么话题、出现过哪些具体实体和结论"，不是记录"这个用户是谁、有什么偏好/
决定"（那是另一层"用户画像"专门负责的工作，不属于这里）。你会收到两样东西：
已有的摘要（可能为空，表示这是第一次生成）、一段新发生的对话原文（用户和
助手交替发言）。

请把新对话合并进已有摘要，输出一份**新的完整摘要**（不是只写新增部分；跟旧摘要
冲突的地方以新信息为准）。不区分领域——用户自己的事、游戏角色、课题、文献、
生活话题都一视同仁地记，只排除寒暄客套话。
群聊中每个人有稳定编号，必须按编号分别记录话题和个人陈述，不合并不同人的事实。
助手回答标明回复对象：其中的“你”只指该对象，不能归给下一位发言人。
个人身份、年龄、健康只从本人明确陈述归纳；旧助手回答可能错认人，不能独立证明个人事实。
昵称仅是称呼，不能因为昵称相同或话题相似就认为是同一个人；泛泛问健康问题不等于自述患病。

【2026-10-01改：摘要是"聊过什么"的索引，不是知识的副本】完整对话原文另外存着、
需要时会去翻，游戏和课题的内容也能随时重新查到，所以摘要只需要让人知道"聊过
哪些话题、讨论到哪、定下了什么"：
- 记：每条话题线（聊过哪些干员/哪篇文献/课题的哪部分/哪类生活话题）、用户做的
  决定和关心的事、还没聊完可能接着聊的事、对话里出现的关键人名/实体名（以后说
  "那个""她"时能对上）；
- 不记：助手回答里的具体知识内容——数值、公式、技能逐条效果、实验步骤、人物生平
  细节、日期对应的节日等（写成"聊过玛恩纳的技能""讨论过LoxCode的多样性原理"
  即可；助手的回答也可能有错，照抄进摘要会一直错下去）；
- 越早的话题写得越概括（可以几个话题并成一句），最近的话题可以稍详细；
- 篇幅控制在600字左右（不用逐字数，输入里会告诉你旧摘要现在有多长；旧摘要超出
  时这次就按上面的标准把它压回来，不要为了保留旧细节而越写越长）。

只输出如下JSON格式，不要用markdown代码块包裹：
{"summary": "这里是合并后的摘要文本"}

如果新对话里确实没有任何实质性内容（比如全是寒暄问候），summary就原样返回
旧摘要（旧摘要为空时返回空字符串）。
"""


# 摘要篇幅预算（用户2026-10-01定：600字；太长每次整份重写越来越慢，太短信息压得过于模糊）
SUMMARY_BUDGET = 600
SUMMARY_TOLERANCE = 0.3

_SHRINK_SYSTEM_PROMPT = """你负责精简一份过长的对话摘要，大约删去一半。摘要是"聊过什么"的索引（完整对话
原文另外存着，需要时会去翻），所以：删掉具体知识内容（数值、公式、技能逐条效果、
实验步骤、人物生平细节、日期对应的节日等，改写成"聊过某某""讨论过某某"），把
较早的几个话题并成一句；保留每条话题线、用户的决定和关心的事、关键人名/实体名。
不用数字数。不能添加原摘要里没有的内容。

只输出如下JSON格式，不要用markdown代码块包裹：
{"summary": "精简后的摘要文本"}
"""


def _condense_turns_text(text: str, max_len: int = 300) -> str:
    """把要压缩的这批原始对话文本整体截短成开头一段，用于内容安全分类器
    拒绝之后的重试——纯长度截断，不保证语义完整，只是为了用一次明显更短、
    更不像完整叙事描述的表述再试一次，呼应MiniMax官方"修改输入内容而不是
    原样重试"这条建议。全文本来就不长于max_len时原样返回，不影响正常情况。
    """
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "……（后续内容较长，已截断）"


def llm_update_summary(old_summary: Optional[str], new_turns_text: str, client) -> Optional[str]:
    """client 是 llm_client.py 的 build_llm_client() 返回的对象。
    返回 None 表示这轮没能生成摘要（LLM不可用/调用失败/返回不是合法JSON），
    调用方应该保持摘要不变，不要覆盖旧值。

    【2026-09真机排查"苇草的异格"时补的caller标注】这次调用发生在
    `conversation_memory.py`的`_maybe_update_l0_summary()`里，时机是
    **本轮回复已经算完之后**（`user_says()`先算出reply、存历史，最后才
    调用这里）——如果这次调用因为内容安全分类器被拒（摘要要压缩的是
    最近几轮完整对话原文，可能比单条wiki正文更容易撞上敏感词），失败
    只会导致"这一轮没能刷新滚动摘要、保持旧摘要不变"，不会影响已经
    生成好、已经返回给用户的那句回复本身。当初排查"合并步骤是不是在
    偷偷编答案"这类问题时，因为这里的调用一直没标caller，日志里的失败
    提示和`synthesize()`真正的失败提示长得一模一样，容易被误认成是
    同一次调用失败——补上标注，以后一眼就能看出这次失败是不是这里。
    """
    if not getattr(client, "is_real", False):
        return None

    length_note = f"（已有摘要现在约{len(old_summary)}字，合并后控制在{SUMMARY_BUDGET}字左右）" if old_summary else ""
    user_content = (
        f"已有摘要{length_note}：{old_summary or '（空，这是第一次生成摘要）'}\n\n"
        f"新发生的对话：\n{new_turns_text}"
    )
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    raw = client.chat(messages, caller="llm_update_summary()")
    if raw is None:
        # 完整版被拒时按官方建议换一种更短的表述重试一次，而不是原样重试
        # 或者直接放弃——见模块开头"精简重试"一节。
        condensed = _condense_turns_text(new_turns_text)
        if condensed != new_turns_text:
            print("[llm_l0_summary] 完整版摘要请求失败，按官方建议换成截断精简版重试一次……")
            retry_user_content = (
                f"已有摘要：{old_summary or '（空，这是第一次生成摘要）'}\n\n"
                f"新发生的对话（内容较长，只保留开头一部分）：\n{condensed}"
            )
            retry_messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": retry_user_content},
            ]
            raw = client.chat(retry_messages, caller="llm_update_summary()[精简重试]")
            if raw is not None:
                # 明确打出"精简重试成功"这件事本身——不然结果只能靠"后面
                # 没有再报错"这种沉默来倒推，用户实测时已经问过一次"怎么
                # 判断成功还是失败"，不能只让人靠猜。
                print("[llm_l0_summary] 精简重试成功，本轮摘要已更新。")
        if raw is None:
            return None

    # "剥代码块外壳+json.loads()+失败返回None"这两步是从llm_json.py共用的
    # ——跟llm_l1.py/wiki_router.py是同一套逻辑，见该文件模块开头的说明，
    # 不再自己写一遍。
    cleaned = strip_json_fence(raw)
    data = try_parse_json(cleaned)
    if data is None:
        # 严格解析失败（或者解析出来不是个dict），试一下宽容兜底——大概率
        # 是摘要正文里出现了没转义的引号（比如提到某个专有名词时用中文/
        # 英文引号包了一下），不是LLM彻底不听指令，见模块开头"真机测试
        # 发现的真实问题"一节。
        fallback = _SUMMARY_FALLBACK_RE.search(cleaned)
        if fallback is not None:
            return _shrink_if_too_long(fallback.group(1).strip(), client)
        print(f"[llm_l0_summary] LLM返回的内容不是预期的JSON格式（原始输出：{raw!r}），"
              f"本轮摘要保持不变。")
        return None

    if not isinstance(data.get("summary"), str):
        print(f"[llm_l0_summary] LLM返回的JSON形状不对（{data!r}），本轮摘要保持不变。")
        return None

    return _shrink_if_too_long(data["summary"].strip(), client)


def _shrink_if_too_long(summary: str, client) -> str:
    """代码量一下长度（2026-10-01）：prompt里原来就写了"超出大约300字就精简"，模型
    没照做、也没人检查，真实摘要涨到了约1250字，每次整份重写越来越慢。超出预算30%
    以上时再让模型精简一次。精简请求里不给目标字数，只说"删去一半左右"和删什么：
    给了字数，推理模型会在思考里反复逐字数（实测精简一次超过120秒被截断，跟生成
    回答后的字数压缩是同一个问题）；精简失败或
    没变短就用合并好的版本，不丢内容。摘要在回复之后的后台更新，不影响回复速度。"""
    if len(summary) <= SUMMARY_BUDGET * (1 + SUMMARY_TOLERANCE):
        return summary
    messages = [
        {"role": "system", "content": _SHRINK_SYSTEM_PROMPT},
        {"role": "user", "content": f"请精简这份摘要：\n{summary}"},
    ]
    raw = client.chat(messages, caller="llm_update_summary()[精简长度]")
    data = try_parse_json(strip_json_fence(raw)) if raw else None
    shorter = data.get("summary", "").strip() if isinstance(data, dict) and isinstance(data.get("summary"), str) else ""
    if shorter and len(shorter) < len(summary):
        print(f"[llm_l0_summary] 摘要{len(summary)}字超出预算，精简到{len(shorter)}字")
        return shorter
    print(f"[llm_l0_summary] 摘要{len(summary)}字超出预算，精简没成功，先用合并好的版本")
    return summary


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_client import build_llm_client

    client = build_llm_client()
    print(f"当前LLM后端：{client.name}（is_real={client.is_real}）\n")

    old = None
    new_turns = (
        "user：我叫长安，是生命科学专业的研究生，做单细胞谱系追踪方向\n"
        "assistant：【L1核心记忆——已记住】...\n"
        "user：我们组决定用Chroma做向量数据库先跑通逻辑\n"
        "assistant：好的\n"
    )
    result = llm_update_summary(old, new_turns, client)
    print(f"第一次生成的摘要：{result}\n")

    if result is not None:
        new_turns_2 = (
            "user：今天晚饭吃什么？\n"
            "assistant：【未找到可靠信息】...\n"
            "user：准确地说，我在做肝脏纤维化方向的研究\n"
            "assistant：【L1核心记忆——已记住】...\n"
        )
        result2 = llm_update_summary(result, new_turns_2, client)
        print(f"合并新对话后的摘要：{result2}")
