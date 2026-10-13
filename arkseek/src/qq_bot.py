# -*- coding: utf-8 -*-
"""
阶段三：QQ官方机器人接入层——把 ConversationSession（记忆系统内核）接到QQ私聊上

【定位】这是一个"薄"的传输层：只负责 QQ消息进 -> user_says() -> 回复出。
记忆、检索、路由、生成全部在 ConversationSession 里，这里一行都不碰——跟
chat_cli.py 是同一个角色（另一个前端），只是输入输出从终端换成了QQ。
连接层本身已经用 qq_bot_echo.py（复读机）单独验证过，出问题先回那边排查。

【当前范围】C2C单聊（私聊）+ 群聊里@机器人的消息（官方规则下群里只收得到
@机器人的消息，群友之间的普通聊天看不到）。

【群聊的记忆怎么算】用户2026-09-25定的方案（跟业界"短期上下文按对话场所、
长期记忆按人"一致）：画像按人、对话按群——
  user_id         = "qq_group:<群openid>:<成员openid>"（同一个人在不同群里算
                    不同的人，免得一个群的画像带到另一个群；私聊和群聊也不打通——
                    实测同一个人私聊的user_openid和群里的member_openid是同一个值，
                    不打通靠的是前缀不同，是有意的设计）
  conversation_id = "qq_group:<群openid>"（L0窗口/摘要/L2检索都按群：群友A刚问的
                    问题是群友B提问的上下文，群里别人问过的内容也能翻到）
具体实现见 ConversationSession 的类说明。

【四个设计点】
1. 用户隔离：QQ官方机器人拿不到QQ号，只给 user_openid（只对本机器人有效，
   实测同一个人私聊和群里的openid是同一个值）。user_id = "qq_c2c:" + openid，
   带前缀是为了跟 chat_cli.py 的 default_user、以后群聊的 member_openid
   分开命名空间。每个user_id一个 ConversationSession，第一次发消息时懒创建、
   之后复用；画像/历史/L0摘要本来就按user_id存在memory.db里，重启也记得。
2. 重资源只加载一次：知识库pipeline（bge-m3/精排模型）、MemoryStore、LLM
   client 启动时建一份，所有用户的session共用。
3. 不阻塞心跳：user_says() 是同步的、一轮可能十几秒（多次LLM调用+检索）。
   直接在异步回调里调会卡住botpy的事件循环，WebSocket心跳发不出去、连接被
   断开。所以丢进线程池执行。
   并发（2026-09-28用户定的"第一档"）：同一个对话串行、不同对话并行。
   "同一个对话"= 私聊按人、群聊按整个群（群里共用一份对话记录/摘要，两个
   群友同时写会打乱轮次顺序），用每个对话一把asyncio.Lock在事件循环里排队，
   排队时不占工作线程。不同对话之间真正共享的三样各有一把锁，都在内核里
   （一条消息内部的子问题并行也要用到）：SQLite连接（memory_store.py）、
   bge-m3的encode和精排模型的predict（pipeline.py）。这几样都是毫秒到几百
   毫秒级；一轮对话的大头是串行的几次LLM调用（网络等待），正好能并行起来，
   同时打到LLM接口的请求数由llm_client.py统一限流。Chroma本身支持多线程访问。
   一条消息内部的并行（推测执行、子问题按依赖图调度）见concurrency.py。
   之前（到2026-09-28为止）是全局只有1个工作线程、所有消息排队。
   对话摘要更新挪到回复发出之后（仍在这个对话的锁里），不拖慢这一条回复。
4. 被动回复：带上原消息的 msg_id 回复（官方对被动回复限制宽松，不带就算主动
   消息，额度/频率限制严得多）。被动回复有有效期，如果某个问题处理太久导致
   发送失败，日志里会看到，届时再考虑先回一句"正在查"。

【发送失败的兜底】回复里可能带网址（比如wiki引用 prts.wiki/...）。QQ官方
平台对消息里的链接有管控，未报备的域名可能导致整条消息被拒——具体规则以
实测为准。发送失败时去掉网址再试一次；还失败就只打日志。

运行：cd src && python qq_bot.py
（启动要先加载知识库和模型，比复读机慢，看到"已上线"再发消息）
"""

import asyncio
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))
except ImportError:
    pass

from runtime_logging import install_runtime_logging

if __name__ == "__main__":
    install_runtime_logging()

import botpy
from botpy import logging
from botpy.message import C2CMessage, GroupMessage

from pipeline import MemorySystemPipeline
from memory_store import MemoryStore
from conversation_memory import ConversationSession
from llm_client import build_llm_client
from chroma_index import DEGRADE_NOTICES
from bot_profile import intro_message
from timing import Trace
import admin_monitor

_log = logging.get_logger()

# qq-botpy 解析群消息时，发送者只留了 member_openid，其它字段丢掉了。这里把原始字段
# 整个留下来（message.author.raw）。2026-09-28群测确认原始字段里有 username——是QQ账号
# 昵称（不是群名片，改群名片后它不变），on_group_at_message_create 用它作为群里称呼
# 这个人的名字，见 memory_store.set_display_name()。
_orig_group_user_init = GroupMessage._User.__init__


def _group_user_init_keep_raw(self, data):
    _orig_group_user_init(self, data)
    self.raw = dict(data or {})


GroupMessage._User.__init__ = _group_user_init_keep_raw

# 回复开头的来源标题（【L3知识库——…】【L2情景记忆——…】【L1核心记忆——…】等）：测试
# 阶段用来看每条回答走的是哪条路，2026-09-28用户决定群测时不给群友看（回答末尾已经有
# "引用来源"）。用开关而不是删代码，需要时改成True就能恢复（chat_cli.py 不受影响，
# 终端里照样显示）。只去掉开头这类标题，不动【以上内容来自网络搜索…】这些用户定的
# 来源标注。
SHOW_SOURCE_HEADERS = False
_HEADER_RE = re.compile(r"^【(?:L\d[^】]*|本地知识库——[^】]*|未找到可靠信息)】[ \t]*\n?", re.M)

# 工作线程数：同时在处理的对话最多几个。瓶颈是LLM接口的并发额度，不是本机
MAX_WORKERS = 4


# /forget：群友要求机器人忘掉自己（用户2026-09-28定的指令名）
_FORGET_COMMANDS = ("/forget",)

_URL_RE = re.compile(r"(https?://)?[\w.-]+\.(?:wiki|com|cn|org|net|io)(?:/[^\s）)】\]]*)?")


class MemoryBotClient(botpy.Client):
    def __init__(self, kb_pipeline, store, llm_client, **kwargs):
        super().__init__(**kwargs)
        self.kb_pipeline = kb_pipeline
        self.store = store
        self.llm_client = llm_client
        self.sessions = {}  # user_id -> ConversationSession
        # 同一个对话串行、不同对话并行，见模块开头第3点
        self.worker = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        # 对话key（私聊是user_id，群聊是conversation_id）-> asyncio.Lock。只在事件循环
        # 线程里读写，不需要再加锁
        self._conv_locks = defaultdict(asyncio.Lock)
        # 对话key -> 回复之后在后台更新摘要的task（/forget前要先等它写完）
        self._summary_tasks = {}

    def _get_session(self, user_id: str, conversation_id: str = None) -> ConversationSession:
        session = self.sessions.get(user_id)
        if session is None:
            # 开场白：私聊是每个人第一次说话时；群聊的对话按群算，所以是这个群
            # 第一次有人@机器人时发一次，不会每个新群友都刷一遍
            session = ConversationSession(self.kb_pipeline, self.store, user_id=user_id,
                                          llm_client=self.llm_client, auto_intro=True,
                                          conversation_id=conversation_id)
            self.sessions[user_id] = session
        return session

    def _handle(self, user_id: str, text: str, conversation_id: str = None, display_name: str = None) -> str:
        """在工作线程里跑：取session + 调记忆系统内核。display_name（群消息发送者的
        QQ昵称）也在这里写库。对话摘要的更新推迟到回复发出之后（_after_reply）。"""
        started = time.time()
        if display_name:
            self.store.set_display_name(user_id, display_name)
        reply = self._get_session(user_id, conversation_id).user_says(text, defer_summary=True)
        # 每条消息的处理耗时（真机测过一次提问6分钟后才回复，当时日志没有时间，定位不了）。
        # 被动回复有效期5分钟，超过就发不出去了
        _log.info(f"[qq_bot] 处理耗时 {time.time() - started:.1f} 秒")
        return reply

    def _forget(self, user_id: str, conversation_id: str = None) -> str:
        """/forget：删掉这个人的画像和对话。私聊整段清空；群里只删这个人的部分
        （memory_store.forget_member()）。缓存的session一并丢掉，下次说话时重建、
        重建时跟Chroma对账，删掉对应的向量。"""
        self.sessions.pop(user_id, None)
        if conversation_id:
            n = self.store.forget_member(conversation_id, user_id)
            # 群的滚动摘要里可能写到了这个人：forget_member()删掉了摘要，这里按剩下的
            # 对话重新生成一份（一次LLM调用），其他群友更早的上下文不会因此丢掉
            ConversationSession(self.kb_pipeline, self.store, user_id=f"{conversation_id}:_summary",
                                llm_client=self.llm_client,
                                conversation_id=conversation_id)._maybe_update_l0_summary()
            # list(...)先拷一份key：别的线程可能正在往sessions里加别的对话的session
            for key in [k for k in list(self.sessions) if k.startswith(conversation_id + ":")]:
                self.sessions.pop(key)  # 同群其他人的session也重建，重建时跟Chroma对账删掉对应向量
            _log.info(f"[qq_bot] {user_id} 执行了/forget，删掉{n}条群对话记录和画像，群摘要已按剩下的对话重建")
            return "好的，我已经忘掉你的个人信息和你在这个群里跟我说过的话了。"
        self.store.reset_user(user_id)
        _log.info(f"[qq_bot] {user_id} 执行了/forget，清空了私聊记录和画像")
        return "好的，我已经忘掉你的个人信息和我们之前的全部对话了。"

    async def on_ready(self):
        admin_monitor.ready()
        _log.info(f"[qq_bot] 机器人「{self.robot.name}」已上线，可以私聊或在群里@它")

    def _after_reply(self, user_id: str) -> None:
        """回复发出之后：更新对话摘要（user_says(defer_summary=True)推迟下来的）。"""
        session = self.sessions.get(user_id)
        if session is not None:
            session.after_reply()

    async def _answer(self, post, user_id: str, text: str, conversation_id: str = None,
                      display_name: str = None) -> None:
        received_at = time.perf_counter()
        request_id = admin_monitor.begin(conversation_id or user_id, user_id)
        try:
            await self._answer_impl(post, user_id, text, conversation_id, display_name,
                                    request_id, received_at)
        except BaseException as exc:
            admin_monitor.finish(request_id, "interrupted" if isinstance(exc, asyncio.CancelledError) else "failed",
                                 (time.perf_counter() - received_at) * 1000, type(exc).__name__)
            raise

    async def _answer_impl(self, post, user_id, text, conversation_id, display_name,
                           request_id, received_at):
        """私聊和群聊共用：算回复、发出去、再做回复之后的收尾。post(content, msg_seq)
        是私聊/群聊各自的发送调用。/help由前端直接处理，其余交给记忆系统内核。"""
        # /help 跟 chat_cli.py 一样由前端直接处理：返回开场白，不交给记忆系统、
        # 不存进对话历史（真机测过：交给记忆系统会被当成闲聊现编一段介绍，
        # 还会作为一轮对话占掉L0窗口）
        if text.lower() in ("/help", "help", "/帮助"):
            admin_monitor.phase(request_id, "sending", 0)
            sent = await self._send(post, intro_message(self.llm_client.is_real))
            admin_monitor.finish(request_id, "sent" if sent else "send_failed",
                                 (time.perf_counter() - received_at) * 1000)
            return
        loop = asyncio.get_running_loop()
        conv_lock = self._conv_locks[conversation_id or user_id]
        if conv_lock.locked():
            _log.info(f"[qq_bot] {conversation_id or user_id} 上一条还在处理，这条排队")
        async with conv_lock:
            admin_monitor.phase(request_id, "processing", (time.perf_counter() - received_at) * 1000)
            _log.info(f"[qq_bot] 对话队列等待 {time.perf_counter() - received_at:.1f} 秒")
            if text.lower() in _FORGET_COMMANDS:
                # 后台还在写的摘要先等它写完，不然写完会把刚清掉的内容又写回去
                pending = self._summary_tasks.get(conversation_id or user_id)
                if pending is not None and not pending.done():
                    await pending
                # 也拿同一把对话锁：群里/forget要改整个群的对话记录和摘要
                reply = await loop.run_in_executor(self.worker, self._forget, user_id, conversation_id)
                admin_monitor.phase(request_id, "sending")
                sent = await self._send(post, reply)
                admin_monitor.finish(request_id, "sent" if sent else "send_failed",
                                     (time.perf_counter() - received_at) * 1000)
                return
            # 分步计时（timing.py）：从拿到对话锁开始算，到回复发出去为止
            trace = Trace(f"{conversation_id or user_id} 消息 {text[:30]!r}")
            processing_error = None
            try:
                reply = await loop.run_in_executor(self.worker, trace.run, self._handle, user_id, text,
                                                   conversation_id, display_name)
            except Exception as e:  # 内核异常不能让整个机器人挂掉
                processing_error = type(e).__name__
                _log.exception(f"[qq_bot] 处理消息出错：{e}")
                reply = "抱歉，我这边处理出错了，稍后再试试。"
            send_start = time.perf_counter()
            admin_monitor.phase(request_id, "sending")
            sent = await self._send(post, reply)
            trace.add(send_start, time.perf_counter() - send_start, "发送", "", None)
            trace.emit()
            _log.info(f"[qq_bot] 本条接收至发送 {time.perf_counter() - received_at:.1f} 秒（含排队）")
            admin_monitor.finish(request_id, "send_failed" if not sent else ("processing_failed" if processing_error else "sent"),
                                 (time.perf_counter() - received_at) * 1000, processing_error, trace.spans)
        # 【2026-10-01】摘要更新放到对话锁外面、在后台做：原来在锁里，同一对话的下一条
        # 要等它（真机13~31秒）。现在下一条用"现有摘要+还没合并的原文"做背景，不用等；
        # 摘要写完整体替换（一次数据库写入，不会读到一半），同一对话同时只跑一个摘要
        # 更新（ConversationSession._summary_lock）
        key = conversation_id or user_id
        prev = self._summary_tasks.get(key)
        if prev is None or prev.done():  # 上一次还在写就不另起，没合并的轮次留到下次
            self._summary_tasks[key] = asyncio.create_task(self._summarize_in_background(user_id))

    async def _summarize_in_background(self, user_id: str) -> None:
        loop = asyncio.get_running_loop()
        start = time.perf_counter()
        try:
            await loop.run_in_executor(self.worker, self._after_reply, user_id)
        except Exception as e:
            _log.exception(f"[qq_bot] 回复后更新对话摘要出错：{e}")
        _log.info(f"[qq_bot] 回复后的摘要检查/更新 {time.perf_counter() - start:.1f} 秒（后台进行，不影响回复，"
                  f"下一条也不用等；攒够4轮才真正重写一次）")

    async def on_c2c_message_create(self, message: C2CMessage):
        text = (message.content or "").strip()
        openid = message.author.user_openid
        user_id = f"qq_c2c:{openid}"
        if not text:
            return  # 纯图片/表情等，暂不处理
        _log.info(f"[qq_bot] 收到 {user_id}：{text!r}")

        async def post(content, seq):
            await message._api.post_c2c_message(openid=openid, msg_type=0, msg_id=message.id,
                                                content=content, msg_seq=seq)
        await self._answer(post, user_id, text)

    async def on_group_at_message_create(self, message: GroupMessage):
        # 群里@机器人的消息，content里已经不含@本身（可能带前导空格）
        text = (message.content or "").strip()
        group_openid = message.group_openid
        conversation_id = f"qq_group:{group_openid}"
        user_id = f"{conversation_id}:{message.author.member_openid}"
        if not text:
            return
        _log.info(f"[qq_bot] 收到群 {conversation_id} 成员 {user_id}：{text!r}")
        # 群消息发送者的username：SDK解析时丢掉了，靠上面的补丁从原始字段里拿
        display_name = (getattr(message.author, "raw", None) or {}).get("username") or None

        async def post(content, seq):
            await message._api.post_group_message(group_openid=group_openid, msg_type=0,
                                                  msg_id=message.id, content=content, msg_seq=seq)
        await self._answer(post, user_id, text, conversation_id, display_name)

    @staticmethod
    def _format_for_qq(reply: str) -> str:
        """只改"怎么显示"，不改内容——QQ私聊的纯文本消息不渲染Markdown。
        ①去掉加粗标记**（真机里原样显示成"**第一结局**"）；②引用来源里的
        "我（助手）的回答""我（助手）回答长安的话"对用户显示成"我的回答""我回答长安的话"——"（助手）"这个标签是给
        生成prompt区分"用户说的/我说的"用的（见llm_generate.py），内核里保留，
        只在QQ这一层显示时去掉（用户2026-09-25的反馈）。"""
        reply = reply.replace("**", "").replace("我（助手）的回答", "我的回答").replace("我（助手）回答", "我回答")
        if not SHOW_SOURCE_HEADERS:
            reply = _HEADER_RE.sub("", reply)
        return reply

    async def _send(self, post, reply: str):
        """post(content, msg_seq)：私聊/群聊各自的发送调用（被动回复，带msg_id）。
        同一条被动回复的第二次发送要换一个msg_seq，否则会被当成重复消息。"""
        reply = self._format_for_qq(reply)
        try:
            await post(reply, 1)
            _log.info(f"[qq_bot] 已回复（{len(reply)}字）")
            return True
        except Exception as e:
            _log.warning(f"[qq_bot] 发送失败：{e!r}；去掉网址后重试一次")
        stripped = _URL_RE.sub("", reply)
        if stripped == reply:
            _log.error("[qq_bot] 回复里没有网址可去，放弃发送（上面是失败原因）")
            return False
        try:
            await post(stripped, 2)
            _log.info("[qq_bot] 去掉网址后发送成功——说明是链接管控导致的，记下来")
            return True
        except Exception as e:
            _log.error(f"[qq_bot] 去掉网址后依然失败：{e!r}")
            return False


def main():
    appid = os.environ.get("QQ_BOT_APPID")
    secret = os.environ.get("QQ_BOT_SECRET")
    if not appid or not secret:
        print("缺少 QQ_BOT_APPID / QQ_BOT_SECRET：请在项目根目录的 .env 里填上")
        sys.exit(1)

    here = os.path.dirname(os.path.abspath(__file__))
    admin_monitor.configure()
    kb_dir = os.path.join(here, "..", "data", "knowledge_base")
    print("正在加载记忆系统知识库……")
    kb_pipeline = MemorySystemPipeline(kb_dir)
    store = MemoryStore()
    llm_client = build_llm_client()
    print(f"当前LLM后端：{llm_client.name}（is_real={llm_client.is_real}）")
    if not llm_client.is_real:
        print("警告：没有可用的LLM，机器人会退回2.1效果（只能查本地知识库）")
    for notice in DEGRADE_NOTICES:
        print(f"降级提示：{notice}")

    # 登录时要先请求QQ网关地址（/gateway/bot）。qq-botpy默认请求超时只有5秒，
    # 偶尔网络慢一点就超时，而它超时后不重试、直接在自己代码里崩溃
    # （TypeError: 'NoneType' object is not subscriptable）。真机撞到过一次，
    # 所以：超时放宽到20秒 + 登录失败时换一个新的事件循环和client重试。
    # 模型/知识库已经加载好，重试只是重新连QQ，成本很低。
    for attempt in range(1, 4):
        # Python 3.14 起 asyncio.get_event_loop() 在没有当前事件循环时直接报错，
        # qq-botpy 1.2.1 内部还在用它——先手动建一个循环兜住（同 qq_bot_echo.py）
        asyncio.set_event_loop(asyncio.new_event_loop())
        intents = botpy.Intents(public_messages=True)
        client = MemoryBotClient(kb_pipeline, store, llm_client, intents=intents, timeout=20)
        try:
            client.run(appid=appid, secret=secret)
            return
        except TypeError as e:
            print(f"[qq_bot] 第{attempt}次登录QQ失败（多半是请求网关地址超时）：{e}")
            if attempt < 3:
                print("[qq_bot] 5秒后重试……")
                import time
                time.sleep(5)
    print("[qq_bot] 连续3次登录失败，请检查网络后重新运行")

if __name__ == "__main__":
    main()
