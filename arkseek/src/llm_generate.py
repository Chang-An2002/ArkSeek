# -*- coding: utf-8 -*-
"""
L2/L3 生成式回答 —— 阶段二 2.2（接LLM）

2.1的README开头说得很直白："没有LLM，'回答'就没法把L0/L1/L2/L3揉合成一句
自然语言总结，只能老实展示'命中了哪一层、证据是什么'"——这个文件就是2.2
补上"揉合成一句自然语言总结"这一步的地方，但只针对L2（情景记忆）和L3（知识
库）：这两层的原始回答形式是"编号列出几条候选片段+来源+分数"，人读起来是
一份检索报告，不是一个直接的答案；L1不需要这个文件，L1的回答本来就是画像
里的具体key-value，已经是"直接的答案"了，不存在"揉合成一句话"的问题。

【置信度gate是硬性防线，生成模型不能绕过去】
L2/L3能不能往下走到"要不要生成"这一步，完全由 conversation_memory.py 的
L2阈值判断、pipeline.py 的 retrieve_with_gate() 的L3阈值判断决定——这个文件
里的函数只在"gate已经判定为可信"之后才会被调用，如果gate判定不可信（没有
检索到、或者分数不够），调用方压根不会调用这里的函数，会直接走2.1原本的
拒答文案。这是故意的设计：接了生成模型不代表可以放松"检索不到就不回答"这
条防幻觉底线，生成的职责只是"把已经确认可信的证据说得更像人话"，不能替代
gate去决定"这个证据到底可不可信"。

【生成失败时怎么办：退回2.1的"展示证据"格式，不是报错也不是编造】
跟 llm_l1.py/llm_l0_summary.py 一样的降级哲学：chat()调用失败返回None，这里
的函数也返回None，调用方看到None就退回2.1原本的format_evidence()格式化文本
——2.1的"展示证据"不是被2.2淘汰的旧实现，是2.2生成失败时的降级路径。

【引用来源是唯一的防幻觉手段，必须在prompt里强制约束】
生成模型的通病是会"自信地编造"看起来合理但片段里其实没有的内容。这里的
prompt明确要求：只能用片段里明确出现的信息作答，每句话标注引用了哪个编号的
片段，片段实在答不了就如实说答不了——不是信任模型"不要瞎编"这句话本身
（模型不一定听），而是给了一个可验证的机制：生成的回答里如果某句话没有
带引用编号，或者引用的编号在片段列表里根本找不到对应内容，就是明显的信号，
读者（或者以后如果做自动化检查）能一眼看出这句话可能是编的。

【检索架构升级新增：可选的`extra_context`参数】
`conversation_memory.py`的`_try_answer_from_history()`/`_try_answer_from_
local_kb()`现在会把L0最近对话（单轮场景）或者多跳编排器已经查到的前置
结果（`query_planner.py`拆解出的依赖子查询场景）传进来，让生成这一步在
措辞时也能借助这段背景——即使传进来的query文本本身带着"她""这个"这类没
被检索匹配环节解析掉的指代，只要片段本身或者这段背景能确定指代的是谁，
生成出来的答案依然可以说对，不会被那部分没解析掉的文本拖累。这不是新的
证据来源，纯粹是帮助生成措辞更准确——置信度gate判定"可信"这件事完全不受
影响，gate还是只看检索到的片段本身，见上面"置信度gate是硬性防线"一节，
这条防线没有变。默认不传，行为跟改动前逐字一致。

【画像可见性修复新增：`extra_context`里的"用户画像"标记行是唯一的例外，
可以被当成证据】上面这条"背景不是证据"的规则本来是对的——但真机测试
（问"我导师推荐我读的是哪篇文献，我读完了吗"）暴露了一种background确实
应该被当成证据的情况：`conversation_memory.py`现在会把用户画像格式化成
一条打了"（用户画像，代表当前已确认的最新状态）"标记的背景行，顺手注入
`recent_context`/`extra_context`（完整设计见该文件模块开头"画像可见性"
一节）。如果这行内容也被当成"不能作为回答依据"的纯消歧背景，那次真实
bug就治标不治本——检索到的两条片段（用户自己反复问的同一个问题）确实
答不上，但画像里其实有答案，这时候应该允许LLM引用画像内容，而不是继续
诚实地说"答不上"。所以这里给"背景不是证据"这条规则开了一个例外：只有
带"用户画像"这个标记的那一行，允许被当成回答依据直接引用；背景里其它
内容（普通历史、"已查到的信息"这类依赖注入）依然只能用来理解代词/省略
指代，不能当证据——完整的prompt文字改动见下面`_SYSTEM_PROMPT`。
"""

import re
from typing import List, Optional, Tuple

from bot_profile import fill_owner
from context_format import current_time_line, format_recent_context
from kb_questions import strip_questions


_SYSTEM_PROMPT = """\
你是一个记忆系统里负责生成回答的组件。你会收到用户的问题，以及若干条已经
确认跟这个问题相关的片段（每条片段有编号和来源）。
群聊按说话人和助手回复对象区分身份，不能把别人获得的建议或自述当成当前用户的。
用户身份、年龄、健康状况只根据本人明确自述或当前画像；旧助手回复可能串人，不能仅凭助手说“你……”就认定是此人的事实。
参与者名册中的称呼可用于识别人，但名册不证明任何健康状况或兴趣。
问以前推荐过的具体内容时优先引用当时的原始问答，而非后来的概括或“没查到”回复。
只找到概括、没有具体细节时说“本次没查到具体内容”，不能断言原记录不存在。
请求“通俗说一下”“这些太专业”是在要求重新组织同一资料，不是寻找有通俗句子的资料。
允许把专业内容忠实转述为通俗语言，不能因此断言知识库不支持回答。

请只根据这些片段的内容，用自然、连贯的语言回答用户的问题（用户用什么语言提问
就用什么语言回答：英文问题用英文答，中文问题用中文答，片段是哪种语言不影响）——像正常聊天一样
说人话，不要逐条复述片段原文，但也不要编造片段里没有的信息。回答中提到某个
具体信息时，在那句话末尾用[编号]标注是根据哪条片段说的（可以同时标多个，
比如[1][2]）。编号只能是片段的编号，范围就是下面给出的那几条；片段原文里
自带的方括号数字（论文正文里的参考文献标号、讲稿里的引用标号）不是片段编号，
不要照抄进回答。
"片段"是系统内部的说法，用户看不懂：回答里提到资料时说它实际是什么（"讲稿里"
"笔记里""原文里""之前的聊天记录里"），不要说"片段里""这些片段"。

有时你还会在问题之前额外看到一段"最近对话"——那是帮你理解问题里代词/省略
指代用的背景（比如"她"指代的是谁），不是新增的证据来源，最终作答依然只能
依据片段内容，不能依据这段背景本身编出片段里没有的信息。

**唯一的例外**：如果这段背景里有一行带"（用户画像，代表当前已确认的最新
状态）"标记，这一行内容可以当成回答依据直接引用（提到时可以标注"[画像]"
说明来源，不需要跟片段共用同一套数字编号）——它是用户已经明确告诉过系统、
并且已经归纳整理好的稳定事实，不是"背景"意义上只帮助理解代词的辅助信息。
如果画像内容和片段内容看起来矛盾，以画像为准（画像代表更新、更确定的
状态；片段可能是更早、或者已经过时的原始记录）。除了这一行例外，背景里
其它内容（普通历史对话、多跳查询里"已查到的信息"这类依赖注入）依然只能
用来理解指代，不能当证据。

片段来源里如果写着"我（助手）的回答"（群聊里也可能是"我（助手）回答某某的话"），说明那是你自己之前回复过的话，不是
用户说的，也不是经过核实的资料：引用时要说成"我之前回答过/我之前提到过……"，
不能说成"你之前说过……"，也不要把它当成确定无疑的事实来陈述（你之前的回答
本身可能有错）。来源写着"你说的话"的片段才是用户自己说过的内容。群聊里来源还可能
写着"群友A说的话"或者某个称呼加"说的话"——那是群里其他人说的，引用时要说成
"群友A之前说过……"，不能说成"你说过"。
群聊里来源写着"我（助手）回答某某的话"、或者片段开头写着"（某某当时问：……）"，说明那次是在回答某某：
如果某某不是当前提问的人，那段回答里讲的就是某某的事，不是当前提问者的。当前提问的
人问"我的开题报告""我的研究"这类自己的事时，只有来源是"你说的话"、或者写着"（你当时
问：……）"的回答才算TA自己的；回答别人的内容不能说成"你的……"——TA自己的记录没有，
就直说没有TA的记录（可以顺带说明之前是谁问过TA自己的什么）。提到别人时直接写称呼，
不要带方括号。

来源标签以"{OWNER}"开头的片段（{OWNER}的课题报告、JC讲稿/PPT/笔记）是开发这个
机器人的{OWNER}本人的资料。提问的人不一定是{OWNER}（机器人可能在群里回答别人的
问题），所以提到这些内容时统一称"{OWNER}"（比如"{OWNER}在JC上讲到……""{OWNER}的
课题用的是……"），不要说成"你的课题""你们"，也不要用"他""她"指代{OWNER}。
回答主要依据这类片段时，正文里第一次提到这份资料就要点明是{OWNER}的（比如
"{OWNER}的开题报告里用的是……""{OWNER}在JC上讲过……"），不能只说"开题报告里……"
然后把出处只留在引用编号里——提问的人可能以为说的是TA自己的报告，群里其他人也
分不清是谁的课题。

提到文献的作者（写论文的研究者）时，不要单说"作者"——说成"Weber等人""这篇文献
的研究团队"，免得跟{OWNER}混淆。

注意区分"{OWNER}自己的课题"和"{OWNER}讲过的文献"：只有来源是"{OWNER}的课题开题
报告/中期进展报告"的片段，讲的才是{OWNER}自己的课题和实验。来源是"{OWNER}……JC
讲稿/PPT/笔记"的片段，是{OWNER}在文献分享会上讲解**别人发表的论文**，里面的实验、
系统、结论属于那篇论文的研究团队，不是{OWNER}做的——回答时要说成"{OWNER}在JC上
介绍的这篇文献里……"，不能说成"{OWNER}的实验用了……"。来源是"文献"的片段同理。

人名、机构名、专有名词、数字、日期一律照抄片段原文，不要凭印象改写或"纠正"
（比如原文是"生命科学研究院"，不能写成"生命科学学院"）。

回答里的每一个具体信息都必须能在片段里找到，并且标注的是它真正出自的那条
片段的编号。"最近对话"（包括其中的摘要）里出现、但片段里没有的具体内容，
不能写进回答，更不能给它标上某条片段的编号。

如果仔细看完这些片段，发现它们**完全**答不了用户的问题（比如片段内容和问题
字面相关，但实际是在说别的事情），只输出 NO_ANSWER 这一个词，不要输出任何
别的内容——系统看到这个信号会自己去别的渠道找答案，并如实标注来源；你不需要、
也不应该自己用一般知识去补充，更不要为了给出一个回答而勉强关联。如果片段能
回答问题的一部分，就正常回答能回答的部分，并如实说明哪部分片段里没有提到
（这种情况不要输出 NO_ANSWER）。

只输出回答正文本身，不要输出"根据提供的片段"之类的开场白，不要用markdown
标题，可以分成一两个自然段。
{LENGTH_RULE}
"""

# 用户在问题里提的字数要求（2026-09-29）：各生成prompt共用这一条，生成后代码只数一遍
# 打日志（2026-10-01起不再调LLM压缩），见length_limit.py
# 【2026-10-01放宽措辞】原来写"硬性要求，正文必须遵守"，推理模型会在思考里反复逐字
# 数。用户的本意是"别长篇大论"——99字和101字没区别。对照实验（各3次）：问题里带
# 字数时，prompt里有没有这条规则，思考长度和耗时差不多——逐字数主要是用户原话里的
# 数字触发的，这条规则只是不再额外加码。
LENGTH_RULE = """
用户的问题里如果写了字数或篇幅要求（比如"100字以内""不超过200字""至少300字"
"一句话说说"），按这个篇幅来写：用户要的是长短合适、不要长篇大论，大致在这个
范围内就行，不需要逐字去数，差几个字没关系。字数只算回答正文，不算[1]这类引用
编号。要求至少多少字、但依据的资料写不满时，如实说明能找到的就这些，不要为了凑
字数编内容。"""

_SYSTEM_PROMPT = fill_owner(_SYSTEM_PROMPT.replace("{LENGTH_RULE}", LENGTH_RULE))


def llm_generate_answer(query: str, passages: List, client, extra_context=None,
                         caller_label: str = "") -> Optional[str]:
    """passages 是 pipeline.RetrievedPassage 的列表（L3）或者 conversation_memory
    里L2检索出来的同类对象——两边共用同一个字段形状（.chunk.source_file /
    .chunk.text），所以L2和L3可以复用同一个生成函数，不用分别写两份几乎一样
    的prompt拼接逻辑。

    extra_context：可选，格式跟`wiki_router.route_query()`的`recent_context`
    一致（元素形如{"role":..., "text":...}），见模块开头"检索架构升级新增"
    一节。默认None，不传时prompt跟改动前逐字一致。

    caller_label：可选，纯粹用于`llm_client.py`失败日志里标出"这次生成
    失败具体是wiki/local_kb/history哪一路"——这个函数本身是L2/L3共用的
    同一份实现，看函数名/调用栈猜不出这次生成的passages来自哪里，2026-09
    真机排查"苇草的异格"时就因为看不出这一点，花了不少功夫回溯代码才确认
    是wiki路的生成失败、不是local_kb那次；调用方（`conversation_memory.py`
    的三个`_try_answer_from_*()`方法）传一个类似"wiki:焰影苇草"的字符串
    进来，不传时行为跟改动前逐字一致。

    返回 None 表示这轮没能生成（LLM不可用/调用失败），调用方应该退回2.1的
    format_evidence()原始证据展示，不应该对用户隐藏"生成失败"这件事本身，
    但也不需要报错——展示原始证据本来就是一个诚实、可用的兜底回答形式。
    """
    if not getattr(client, "is_real", False):
        return None
    if not passages:
        return None

    passages_text = "\n\n".join(
        f"[{i}] 来源：{p.chunk.source_file}\n{strip_questions(p.chunk.text)}"
        for i, p in enumerate(passages, start=1)
    )
    context_block = format_recent_context(extra_context)
    user_content = f"{context_block}用户的问题：{query}\n\n检索到的相关片段：\n\n{passages_text}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT + current_time_line()},
        {"role": "user", "content": user_content},
    ]
    caller = f"llm_generate_answer({caller_label})" if caller_label else "llm_generate_answer()"
    reply = client.chat(messages, caller=caller)
    if reply is None or not reply.strip():
        return None
    return drop_invalid_citations(reply.strip(), len(passages))


_CITATION = re.compile(r"\[(\d+)\]")


def drop_invalid_citations(text: str, n_passages: int) -> str:
    """删掉超出片段范围的引用编号。知识库扩充后的真机验证里，生成的回答引用了
    [5]，而片段只有3条——多半是照抄了片段原文里论文自带的参考文献标号。这种编号
    在"引用来源"列表里对不上任何一条，读者没法核对，比不标还糟。prompt里已经
    提醒过，这里再做一道确定性的校验（引用校验是RAG生成的常规后处理）。"""
    return _CITATION.sub(lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_passages else "", text)


_MODEL_SOURCE_LIST = re.compile(r"\n\s*(?:引用来源|参考来源|参考链接|资料来源)\s*[:：]")


def compact_citations(text: str, sources: List[str]) -> Tuple[str, List[str]]:
    """引用来源只列正文实际引用到的、同一来源只列一次，编号按首次出现顺序重排，
    正文里的[n]跟着改（2026-10-01真机：正文只引用了[1]，列表却是三条一模一样的
    "长安的课题开题报告"——原来按检索片段逐条列；用户之前也要求过"未用到的引用
    不应列出"）。sources[i]是编号i+1对应的来源名。正文一个编号都没引用时，列出
    去重后的全部来源（保留出处）。返回(新正文, 新来源列表)。

    2026-10-02真机：模型模仿最近对话里助手回复的格式，自己在正文末尾写了一段
    "引用来源：[1] ……"，代码再加一段，用户看到两段。来源列表一律由代码加，
    正文里模型自己写的那段（从"引用来源/参考来源/参考链接："这一行起）去掉。"""
    text = _MODEL_SOURCE_LIST.split(text, maxsplit=1)[0].rstrip()
    order = []
    for m in _CITATION.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= len(sources) and n not in order:
            order.append(n)
    if not order:
        return text, list(dict.fromkeys(sources))
    listed, new_no = [], {}
    for n in order:
        src = sources[n - 1]
        if src not in listed:
            listed.append(src)
        new_no[n] = listed.index(src) + 1
    text = _CITATION.sub(lambda m: f"[{new_no[int(m.group(1))]}]" if int(m.group(1)) in new_no else "", text)
    text = re.sub(r"(\[\d+\])(?:\1)+", r"\1", text)  # 合并后相邻的重复编号：[1][1] -> [1]
    return text, listed


# 生成步骤判断"片段完全答不了"时输出的信号——相当于业界CRAG/Self-RAG里的
# 可答性判断，只是复用了已有的这一次生成调用，不多花一次LLM调用。之前生成步骤
# 答不上时会自由发挥（C15：先说答不上、又追加一段"从一般知识来说……"，还顶着
# "L3知识库"的标题），现在改成输出这个信号，由调用方转去联网搜索/大模型自身
# 知识，并由调用方明确标注来源。见README"QQ群压力测试准备（二）"一节。
NO_ANSWER = "NO_ANSWER"


def is_no_answer(generated: Optional[str]) -> bool:
    return generated is not None and generated.strip().strip("。.").upper() == NO_ANSWER


# ============================================================
# 兜底回答：wiki/本地知识库查不到时，联网搜索 -> 大模型自身知识
# ============================================================
# 用户定的标注文字，逐字使用，由调用方（conversation_memory.py）加在回答后面，
# 不交给LLM自己写，保证每次都一字不差。
WEB_SEARCH_LABEL = "【以上内容来自网络搜索，未经核实，请以实际为准】"
LLM_KNOWLEDGE_LABEL = "【以上由大模型根据自身知识回答，不代表知识库准确信息，请以实际为准】"

_WEB_SYSTEM_PROMPT = """你是一个聊天机器人里负责"根据联网搜索结果回答问题"的组件。你会收到用户的
问题，以及若干条搜索结果（每条有编号、标题、摘要，可能有日期）。
结果只证明它实际写明的内容；提到某概念不等于证实用户设想，相关性不等于因果或定论。
医学信息优先依据正规医疗机构、官方资料或原始研究；只有商业推广或零散经验时说明证据不足。
健康问题给一般性信息，不凭聊天诊断，不把通用药物建议组织成个人处方。

请只根据这些搜索结果回答，用自然、简洁的语言（跟用户提问的语言一致），像正常聊天一样说人话。提到某个
具体信息时，在那句话末尾用[编号]标注出自哪条搜索结果。人名、作品名、数字等
照抄搜索结果原文。搜索结果之间有出入时如实说明，不要自己裁决谁对。不要在
回答里写"以上来自网络搜索"之类的来源声明——系统会自己加。

如果搜索结果完全答不了这个问题，只输出 NO_ANSWER 这一个词。

只输出回答正文本身，不要开场白，不要markdown标题。
""" + LENGTH_RULE

_OWN_KNOWLEDGE_SYSTEM_PROMPT = """你是一个聊天机器人。这个问题在机器人的资料库里没有查到、联网搜索也没有找到
可用的结果，现在请你根据自己的知识直接回答。

要求：用自然、简洁的语言回答（跟用户提问的语言一致）；拿不准的地方直接说拿不准，不要编造具体的数字、
日期、人名、链接；涉及可能已经过时的信息（比如游戏版本、最新动态），提醒用户
可能有变化。不要在回答里写"以上由大模型回答"之类的来源声明——系统会自己加。

只输出回答正文本身，不要开场白，不要markdown标题。
""" + LENGTH_RULE


def llm_answer_from_web(query: str, results: List[dict], client, extra_context=None) -> Optional[str]:
    """根据联网搜索结果生成回答。返回None=生成失败；返回NO_ANSWER=搜索结果
    答不了（调用方据此退到大模型自身知识）。"""
    if not getattr(client, "is_real", False) or not results:
        return None
    results_text = "\n\n".join(
        f"[{i}] {r['title']}（{r['date']}）\n{r['snippet']}" if r.get("date") else
        f"[{i}] {r['title']}\n{r['snippet']}"
        for i, r in enumerate(results, start=1)
    )
    messages = [
        {"role": "system", "content": _WEB_SYSTEM_PROMPT + current_time_line()},
        {"role": "user", "content": f"{format_recent_context(extra_context)}用户的问题：{query}\n\n"
                                    f"搜索结果：\n\n{results_text}"},
    ]
    reply = client.chat(messages, caller="llm_answer_from_web()")
    return reply.strip() if reply and reply.strip() else None


def llm_answer_from_own_knowledge(query: str, client, extra_context=None) -> Optional[str]:
    if not getattr(client, "is_real", False):
        return None
    messages = [
        {"role": "system", "content": _OWN_KNOWLEDGE_SYSTEM_PROMPT + current_time_line()},
        {"role": "user", "content": f"{format_recent_context(extra_context)}用户的问题：{query}"},
    ]
    reply = client.chat(messages, caller="llm_answer_from_own_knowledge()")
    return reply.strip() if reply and reply.strip() else None


if __name__ == "__main__":
    import os
    import sys
    from dataclasses import dataclass

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from llm_client import build_llm_client

    @dataclass
    class _FakeChunk:
        source_file: str
        text: str

    @dataclass
    class _FakePassage:
        chunk: _FakeChunk

    client = build_llm_client()
    print(f"当前LLM后端：{client.name}（is_real={client.is_real}）\n")

    fake_passages = [
        _FakePassage(_FakeChunk(
            "开题报告.txt",
            "以10×Genomics的细胞条形码为桥梁，将单细胞转录组图谱与真实谱系条形码数据"
            "进行匹配（取交集），仅保留同时具有两种组学信息的细胞。",
        )),
        _FakePassage(_FakeChunk(
            "开题报告.txt",
            "本课题通过聚焦对照组与DDC药物处理组的主要差异，对DNA条形码和单细胞转录组"
            "的测序数据进行标准化与个性化分析。",
        )),
    ]
    result = llm_generate_answer("多组学数据整合有哪些方法？", fake_passages, client)
    print(f"生成结果：{result}")
