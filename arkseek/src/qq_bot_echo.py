# -*- coding: utf-8 -*-
"""
阶段三第一步：QQ官方机器人"复读机"——只验证接入层，不接记忆系统

【为什么先写复读机】
接入QQ有两层可能出问题：①连接层（AppID/Secret对不对、WebSocket连不连得上、
收发消息的API调用对不对）②记忆系统本身。先用一个不依赖任何项目代码的最小
脚本单独验证①，通了再把 ConversationSession 接进来替换复读逻辑——出问题时
能立刻分清是哪一层的锅。

【依赖】官方SDK qq-botpy（pip install qq-botpy），跟项目其它代码完全独立，
这个脚本不import项目里任何模块。

【凭证】从环境变量读 QQ_BOT_APPID / QQ_BOT_SECRET，跟 MINIMAX_API_KEY 一样写在
项目根目录的 .env 里（.env 已在 .gitignore 中）。代码里不出现真实Secret。

【当前范围】只处理C2C单聊（私聊）消息：事件 on_c2c_message_create，
intents 用 public_messages（官方把群@消息和单聊消息放在同一个事件通道里）。
群聊留到私聊跑通之后再加。

【用户标识】QQ官方机器人拿不到QQ号，拿到的是 user_openid——同一个人在
私聊和群里的openid不一样，且只对这一个机器人有效。这里先打印出来看看长什么样，
接记忆系统时它就是 ConversationSession(user_id=...) 的来源。

运行：cd src && python qq_bot_echo.py
成功标志：终端打印 on_ready，后台状态从"离线"变成在线；私聊机器人发一句话，
它回"我收到了你的消息：xxx"。
"""

import asyncio
import os
import sys

try:
    from dotenv import load_dotenv
    # .env 在项目根目录（src 的上一级），不依赖从哪个目录启动
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass

import botpy
from botpy import logging
from botpy.message import C2CMessage

_log = logging.get_logger()


class EchoClient(botpy.Client):
    async def on_ready(self):
        _log.info(f"[qq_bot_echo] 机器人「{self.robot.name}」已上线（on_ready）")

    async def on_c2c_message_create(self, message: C2CMessage):
        content = (message.content or "").strip()
        openid = message.author.user_openid
        _log.info(f"[qq_bot_echo] 收到私聊 openid={openid} 内容={content!r}")
        # msg_id 带上原消息id = "被动回复"：官方对被动回复限制宽松，
        # 不带msg_id就是"主动消息"，有更严格的频率/额度限制
        await message._api.post_c2c_message(
            openid=openid,
            msg_type=0,
            msg_id=message.id,
            content=f"我收到了你的消息：{content}",
        )


def main():
    appid = os.environ.get("QQ_BOT_APPID")
    secret = os.environ.get("QQ_BOT_SECRET")
    if not appid or not secret:
        print("缺少 QQ_BOT_APPID / QQ_BOT_SECRET：请在项目根目录的 .env 里填上"
              "（QQ机器人管理后台 → 开发设置 → AppID / AppSecret）")
        sys.exit(1)

    # Python 3.14 起 asyncio.get_event_loop() 在没有当前事件循环时直接报错，
    # 而 qq-botpy 1.2.1（2024年发布）内部还在用它——先手动建一个循环兜住
    asyncio.set_event_loop(asyncio.new_event_loop())

    intents = botpy.Intents(public_messages=True)
    client = EchoClient(intents=intents)
    client.run(appid=appid, secret=secret)


if __name__ == "__main__":
    main()
