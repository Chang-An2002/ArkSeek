# -*- coding: utf-8 -*-
"""
机器人的身份与能力介绍（ArkSeek）

【为什么要有这个模块】
计划把机器人放进QQ群给朋友测试。朋友们不知道机器人的知识范围、也不知道
大模型的能力边界，最好一开始就主动介绍"我能做什么"（用户的想法：开场白）。

之前没有统一的自我介绍：被问到"你能干什么"时，是 route_query() 在生成
route="general" 的闲聊回复时临时编一段——那次碰巧说对了，但能力一变就会
对不上。所以这里集中定义名字和能力清单，两处共用同一份：
- intro_message()：开场白。每个用户第一次对话时说一次（由前端决定是否
  开启，见 ConversationSession.user_says() 的 auto_intro），以及 /help 命令
- SELF_DESCRIPTION_FOR_PROMPT：注入 wiki_router 的路由prompt，被问"你是谁/
  你能做什么"时，general回复按这份介绍说，不自己编

【开场白按实际可用的能力生成，不写死】
wiki查询和自然语言回答都依赖真实LLM；没接LLM（--no-llm或.env没配key）时
只能做本地知识库检索+展示证据，开场白就不该承诺明日方舟问答。本地知识库
的描述目前是固定的一句话，等知识库扩充（JC讲稿/PPT/文献）接入文献目录之后，
改成从目录自动生成，见README"接下来"。
"""

import os

BOT_NAME = "ArkSeek"

# 机器人主人（开发这个机器人、知识库里课题报告和JC材料的主人）的昵称。
# 原来一律称"作者"，但知识库扩充后"作者"有歧义——既可能指机器人的主人，也可能
# 指文献的作者（文献目录条目里"Chen 等"和"作者在JC上讲过这篇"挤在同一段）。
# 用户定了昵称"长安"（2026-09-25）：知识库片段标签、文献目录、自我介绍、各个
# prompt都引用这个常量（prompt里写成{OWNER}占位符），换昵称只改这一处。真名在
# 建索引时替换成这个昵称，真名清单在不提交的 data/kb_sources.json 里，见
# kb_sources.redact_owner_names()。
OWNER_NAME = "长安"


def is_owner(user_id: str) -> bool:
    """这个user_id是不是机器人主人本人（2026-09-28用户要求绑定）。按平台账号ID判断，
    不按自称——群里谁都可以说"我是长安"。ID清单在 .env 的 ARKSEEK_OWNER_IDS（每次
    现读，改了.env重启就生效）；user_id形如"qq_c2c:<id>"或"qq_group:<群>:<id>"，
    取最后一段比对。"""
    ids = {x.strip() for x in os.environ.get("ARKSEEK_OWNER_IDS", "").split(",") if x.strip()}
    return bool(ids) and user_id.rsplit(":", 1)[-1] in ids

# 本地知识库目前收录的内容。知识库扩充后由MemorySystemPipeline按目录生成覆盖。
LOCAL_KB_DESCRIPTION = f"{OWNER_NAME}的专业方向——单细胞谱系追踪课题（开题报告、中期进展报告）"


def set_local_kb_description(desc: str) -> None:
    """知识库扩充后，描述按文献目录实际内容生成（kb_sources.local_kb_description()），
    MemorySystemPipeline加载完知识库时调用。"""
    global LOCAL_KB_DESCRIPTION
    LOCAL_KB_DESCRIPTION = desc


def capabilities(llm_available: bool) -> list:
    caps = ["陪你闲聊，聊什么都行"]
    if llm_available:
        caps.append("回答明日方舟相关的问题（干员、技能、天赋、模组等，查的是PRTS wiki）")
    # 开场白只点到为止（用户2026-09-25定的措辞）：知识库具体收录了什么（课题报告、
    # JC、文献目录），等群友真的问起时由 self_description_for_prompt() 那份完整
    # 介绍展开，不在开场白里罗列
    caps.append(f"如果想了解我的作者：{OWNER_NAME}的专业方向，我也可以透露给你哦～")
    return caps


def intro_message(llm_available: bool) -> str:
    lines = [f"你好，我是{BOT_NAME}！我可以："]
    lines += [f"{i}. {c}" for i, c in enumerate(capabilities(llm_available), start=1)]
    lines.append("有些问题我查不到的时候会老实告诉你；随时输入 /help 可以再看这段介绍。")
    return "\n".join(lines)


def self_description_for_prompt() -> str:
    return (_SELF_DESCRIPTION_TEMPLATE.replace("{LOCAL_KB_DESCRIPTION}", LOCAL_KB_DESCRIPTION)
            .replace("{BOT_NAME}", BOT_NAME).replace("{OWNER}", OWNER_NAME))


def fill_owner(prompt: str) -> str:
    """把prompt里的{OWNER}占位符换成主人昵称。各模块在定义prompt常量后调用一次。"""
    return prompt.replace("{OWNER}", OWNER_NAME)


_SELF_DESCRIPTION_TEMPLATE = """\
【关于你自己的身份】你在route="general"里给出的回复，是以聊天机器人"{BOT_NAME}"的
身份说的。如果用户问"你是谁""你叫什么""你能做什么""你会什么"这类问题，按下面
这份介绍的意思回答（可以用自己的话说得自然些，但不要增加这里没有的能力，也不要
提"知识库""检索""路由"这类技术词汇）：
- 可以陪用户闲聊
- 可以回答明日方舟相关的问题（干员、技能、天赋、模组、剧情、关卡、道具等，查的是
  PRTS wiki）
- 可以介绍{LOCAL_KB_DESCRIPTION}的内容
- 记得之前聊过的内容，可以回顾、总结（群里是整个群的对话）
用户让你总结/回顾/"显示聊天摘要"/"我们之前聊了什么"时，这是你自己就能做的事，
不是QQ或者别的软件的功能：根据背景里的"更早对话的滚动摘要"（如果有）和最近对话，
用自己的话整理出来；背景里没有滚动摘要，说明更早的内容还不多，就按最近对话总结。
回答里不要提"滚动摘要"这类内部说法；提到群友时用称呼，不要用"他""她"（不知道性别）。
（2026-09-28真机：群里问"请显示当前群摘要"，它答"没有这个功能，得看QQ群本身"。）
注意"{OWNER}"是开发你这个机器人的人（你的主人），不是正在跟你聊天的这位用户——
机器人会放进群聊，跟你聊天的通常是{OWNER}的朋友。介绍这一项时要说成"{OWNER}的
课题/{OWNER}的报告"，不要说成"你的课题""你那边的报告"。提到{OWNER}时直接说
"{OWNER}"，不要用"他""她"这类人称代词（系统并不知道{OWNER}的性别，不要替{OWNER}
猜）。用户说"作者""你的作者""开发你的人""你主人"而没有指明是哪篇文献的作者时，
指的就是{OWNER}。
"""
