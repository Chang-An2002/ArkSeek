# -*- coding: utf-8 -*-
"""
持久化存储层（L1 核心记忆 / L2 情景记忆）—— 阶段二 2.1（不接LLM）

对应方案文档 4.1 节的分层记忆设计：
    L0 会话上下文——本文件不管，是 conversation_memory.py 里的内存滑动窗口
    L1 用户画像/核心记忆——本文件的 user_profile 表
    L2 情景记忆（完整历史对话）——本文件的 conversation_turns 表
    L3 知识库——不归这里管，还是 chunking.py + pipeline.py 那一套

【为什么用 SQLite，而不是继续跟阶段一一样用文件】
阶段一的知识库是"一次性加载、只读"的静态语料，用 .txt 文件很自然。但 L1/L2
从阶段二开始要支持"运行时不断追加、按需查询"（每一轮对话都要写入、每一轮
提问都可能要按用户ID查历史），用一个真正的数据库比手搓JSON读写/加锁更省心，
也更贴近方案文档技术选型（第5节）里"结构化存储：SQLite"这条建议——这条建议
从阶段一开始就写在文档里，一直没真正用上，阶段二正好是它该登场的地方。

【表结构设计】
- user_profile：L1，用 (user_id, key) 做主键的键值表，而不是固定列的表——
  一个人的"稳定事实"是开放集合（今天多一条"研究方向"，明天可能多一条
  "常问话题"），键值表能不改表结构就加新的事实类型，这也是 fact_extractor.py
  能随时扩展识别规则、不用同步改数据库schema的原因。source_text 字段记录
  "这条事实是从哪句话抽出来的"，方便调试规则准不准、也符合"可验证性"的设计
  原则（跟L3知识库"每条结果都带来源"是同一个思路）。
- conversation_turns：L2，一行一轮对话（区分 user/assistant 两种角色），
  按 (user_id, turn_index) 排序即完整历史。这张表既是"事实的完整档案"
  （不物理删除，方案文档4.3节"只做检索权重上的时间衰减，不物理删除"），
  也是 conversation_memory.py 里做L2混合检索时的语料来源——检索时会把这张表
  的行现取出来包成 chunking.Chunk，临时建一份BM25/向量索引去搜，复用L3那套
  已经验证过的检索代码，而不是另起一套检索逻辑。
- l0_summary：阶段二2.2新增，L0会话窗口滑出去的内容的滚动摘要（每个用户
  一行，覆盖写）。只有真实LLM可用时才会写入，2.1完全不涉及这张表——详见
  下方 get_l0_summary()/set_l0_summary() 的说明。

【当前只有一个真实用户，为什么还要设计 user_id 字段】
虽然这个项目目前只有你一个人在用，但"L1/L2 记忆是按用户维度隔离的"是这类
系统的基本设计前提（也是多轮测试用例之间互不干扰的关键——每段测试对话可以
用不同的 user_id，天然隔离，不用在测试之间手动清空数据库）。加这一列不增加
实现复杂度，但让设计经得起"以后要不要支持多用户"这个问题，值得从一开始就
带上。
"""

import functools
import inspect
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "memory.db"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class MemoryStore:
    """L1（用户画像）+ L2（对话历史）的 SQLite 读写封装。"""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        # 所有方法共用这一个连接，2026-09-28起会被多个线程同时调用（qq_bot不同对话
        # 并行、一条消息内的子问题并行），每个方法整体在这把锁里执行，见文件末尾
        # _synchronized()。可重入：方法内部互相调用不会自己卡死自己
        self._lock = threading.RLock()
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id     TEXT NOT NULL,
                key         TEXT NOT NULL,
                value       TEXT NOT NULL,
                source_text TEXT,
                updated_at  TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );

            CREATE TABLE IF NOT EXISTS conversation_turns (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          TEXT NOT NULL,
                turn_index       INTEGER NOT NULL,
                role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                text             TEXT NOT NULL,
                created_at       TEXT NOT NULL,
                is_general_chat  INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_turns_user
                ON conversation_turns (user_id, turn_index);

            CREATE TABLE IF NOT EXISTS display_name (
                user_id     TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS l0_summary (
                user_id           TEXT PRIMARY KEY,
                summary           TEXT NOT NULL,
                summarized_up_to  INTEGER NOT NULL,
                updated_at        TEXT NOT NULL
            );
            """
        )
        self.conn.commit()
        self._migrate_add_is_general_chat_column()
        self._migrate_add_speaker_column()

    def _migrate_add_is_general_chat_column(self):
        """给已经存在的旧数据库（比如你本地那个已经聊了四十多轮的
        data/memory.db）补上 is_general_chat 这一列——`CREATE TABLE IF NOT
        EXISTS`只对全新数据库有效，已经建过表的旧库不会自动加新列，需要
        单独 `ALTER TABLE` 一次。这一列的用途见 conversation_memory.py
        模块开头"L2语料该不该收录route=general的发言"一节：标记"这一轮
        用户发言，最终是被route="general"当成寒暄/闲聊/建议请求直接回复的，
        不是什么值得被L2检索翻出来的历史事实"，默认0（不是），只有
        `mark_turn_general_chat()`会把它设成1。旧数据（这次修复之前就已经
        存在的历史轮次）统一还是默认值0——没办法回溯当时到底有没有命中
        route="general"，这是这个修复"只管往后、不回溯改历史"的诚实局限，
        完整讨论见 README。
        """
        cols = [row["name"] for row in self.conn.execute("PRAGMA table_info(conversation_turns)")]
        if "is_general_chat" not in cols:
            self.conn.execute(
                "ALTER TABLE conversation_turns ADD COLUMN is_general_chat INTEGER NOT NULL DEFAULT 0"
            )
            self.conn.commit()

    def _migrate_add_speaker_column(self):
        """阶段三群聊新增：conversation_turns 加一列 speaker（这句话是谁说的）。
        私聊/chat_cli 里一段对话只有一个用户，speaker 留空（NULL），行为跟加这
        一列之前完全一样；群聊里一个群是一段对话（user_id 列存的是群的
        conversation_id），用户发言的 speaker 存发言人自己的 user_id，见
        conversation_memory.py 模块开头"群聊"一节。旧库补列，旧数据留空。"""
        cols = [row["name"] for row in self.conn.execute("PRAGMA table_info(conversation_turns)")]
        if "speaker" not in cols:
            self.conn.execute("ALTER TABLE conversation_turns ADD COLUMN speaker TEXT")
            self.conn.commit()

    # ---------- L1：用户画像 ----------

    def upsert_profile_fact(self, user_id: str, key: str, value: str,
                             source_text: str = "") -> None:
        """写入/覆盖一条用户画像事实。同一个 (user_id, key) 再次写入会覆盖旧值
        （比如"研究方向"从"生物信息学"更新成更具体的说法），这对应方案文档
        4.3节 L1 的"增/改/删"里的"改"；"删"目前没有对应的规则触发场景，暂不实现。
        """
        self.conn.execute(
            """
            INSERT INTO user_profile (user_id, key, value, source_text, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (user_id, key) DO UPDATE SET
                value = excluded.value,
                source_text = excluded.source_text,
                updated_at = excluded.updated_at
            """,
            (user_id, key, value, source_text, _now_iso()),
        )
        self.conn.commit()

    def delete_profile_fact(self, user_id: str, key: str) -> None:
        """删掉一条画像（2026-10-01："不喜欢"里唯一的一项被移走等，画像抽取给value=
        "（无）"时调用）。"""
        self.conn.execute("DELETE FROM user_profile WHERE user_id = ? AND key = ?", (user_id, key))
        self.conn.commit()

    def get_profile(self, user_id: str) -> Dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value, source_text FROM user_profile WHERE user_id = ?", (user_id,)
        ).fetchall()
        from profile_facts import supported_profile_fact
        return {row["key"]: row["value"] for row in rows
                if supported_profile_fact(row["key"], row["value"], row["source_text"] or "")}

    # ---------- L2：对话历史 ----------

    def append_turn(self, user_id: str, role: str, text: str, speaker: Optional[str] = None) -> int:
        """追加一轮对话，返回这一轮的 turn_index（从0开始，按用户各自计数）。
        群聊时 user_id 是群的 conversation_id，speaker 是发言人，见
        _migrate_add_speaker_column()。"""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) AS max_idx FROM conversation_turns "
            "WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        next_idx = row["max_idx"] + 1
        self.conn.execute(
            "INSERT INTO conversation_turns (user_id, turn_index, role, text, created_at, speaker) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, next_idx, role, text, _now_iso(), speaker),
        )
        self.conn.commit()
        return next_idx

    def get_recent_turns(self, user_id: str, limit: int) -> List[sqlite3.Row]:
        """L0 用：最近 limit 轮（按 turn_index 升序返回，方便直接按顺序拼上下文）。"""
        rows = self.conn.execute(
            "SELECT turn_index, role, text, created_at, speaker FROM conversation_turns "
            "WHERE user_id = ? ORDER BY turn_index DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return list(reversed(rows))

    def get_all_turns(self, user_id: str) -> List[sqlite3.Row]:
        """L2 检索用语料：这个用户的完整历史，按 turn_index 升序。带上
        is_general_chat，方便调用方过滤掉route="general"命中的发言——见
        mark_turn_general_chat()的说明。"""
        rows = self.conn.execute(
            "SELECT turn_index, role, text, created_at, is_general_chat, speaker FROM conversation_turns "
            "WHERE user_id = ? ORDER BY turn_index ASC",
            (user_id,),
        ).fetchall()
        return rows

    def mark_turn_general_chat(self, user_id: str, turn_index: int) -> None:
        """事后把某一轮用户发言标记成"route='general'命中过"——之所以是"事后"
        标记而不是 append_turn() 时就知道，是因为存这句话的时候（conversation_
        memory.py 的 user_says() 一开始）还没跑完L1/L2/L3判断，不知道最终是
        谁接住了这句话；判断完之后如果是route="general"接住的，才回头补上
        这个标记。完整设计动机见 conversation_memory.py 模块开头。
        """
        self.conn.execute(
            "UPDATE conversation_turns SET is_general_chat = 1 "
            "WHERE user_id = ? AND turn_index = ? AND role = 'user'",
            (user_id, turn_index),
        )
        self.conn.commit()

    # ---------- L0：会话窗口溢出摘要（阶段二2.2新增，2.1不使用这两个方法）----------
    #
    # 2.1的L0只是"最近window_size轮"这个滑动窗口本身，没有"窗口之外的内容
    # 去哪了"这个问题——因为L2检索永远是对 conversation_turns 整张表现取现建
    # 索引，窗口滑出去的内容原文一直都在，随时能被L2语义检索翻出来，不存在
    # MemGPT那种"核心内存/归档内存"之间需要真正搬迁数据的"驱逐"过程。
    # 2.2要解决的是另一个问题：没有LLM时，滑出窗口的内容除了"能被L2检索到"，
    # 没有任何浓缩形式可以直接放进prompt上下文里；这张表存的就是这个浓缩
    # 结果——一份随着对话推进不断滚动更新的摘要，跟L2的完整原文检索是互补
    # 关系，不是替代关系（摘要负责"大致记得聊过什么"，L2负责"精确翻出原话"）。
    #
    # summarized_up_to：已经并入这份摘要的最后一个turn_index（含）。滚动更新
    # 只需要把"新滑出窗口、还没并入摘要"的那几轮追加进去，不用每次都从头
    # 重新摘要整个历史——这也是为什么每个用户只存一行、覆盖写，而不是像
    # conversation_turns那样每轮追加一行。

    def get_l0_summary(self, user_id: str) -> Optional[Dict[str, object]]:
        """没有摘要（还没触发过窗口溢出，或者一直没有真实LLM可用）时返回None。"""
        row = self.conn.execute(
            "SELECT summary, summarized_up_to FROM l0_summary WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return {"summary": row["summary"], "summarized_up_to": row["summarized_up_to"]}

    def set_l0_summary(self, user_id: str, summary: str, summarized_up_to: int) -> None:
        self.conn.execute(
            """
            INSERT INTO l0_summary (user_id, summary, summarized_up_to, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                summary = excluded.summary,
                summarized_up_to = excluded.summarized_up_to,
                updated_at = excluded.updated_at
            """,
            (user_id, summary, summarized_up_to, _now_iso()),
        )
        self.conn.commit()

    def reset_user(self, user_id: str) -> None:
        """清空某个用户的 L0摘要+L1+L2 数据。主要给批量测试脚本用——每段模拟
        对话开始前重置，保证测试用例之间互不污染（也可以给每段测试用例分配
        独立的 user_id，两种方式都行，reset更直接）。"""
        self.conn.execute("DELETE FROM user_profile WHERE user_id = ?", (user_id,))
        self.conn.execute("DELETE FROM display_name WHERE user_id = ?", (user_id,))
        self.conn.execute("DELETE FROM conversation_turns WHERE user_id = ?", (user_id,))
        self.conn.execute("DELETE FROM l0_summary WHERE user_id = ?", (user_id,))
        self.conn.commit()

    # ---------- 群聊：平台给的昵称 ----------

    def set_display_name(self, user_id: str, name: str) -> None:
        """QQ群消息里发送者的username（实测是QQ账号昵称，不是群名片——改群名片后它
        不变）。每条群消息都刷新一次，昵称改了也能跟上。这是平台元数据，不是用户
        说过的事实，所以单独一张表，不混进L1画像。"""
        self.conn.execute(
            "INSERT INTO display_name (user_id, name, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT (user_id) DO UPDATE SET name = excluded.name, updated_at = excluded.updated_at",
            (user_id, name, _now_iso()),
        )
        self.conn.commit()

    def get_display_name(self, user_id: str) -> Optional[str]:
        row = self.conn.execute("SELECT name FROM display_name WHERE user_id = ?", (user_id,)).fetchone()
        return row["name"] if row else None

    def forget_member(self, conversation_id: str, member_id: str) -> int:
        """群聊的 /forget（用户2026-09-28要求）：群里的对话是大家共用的，只删这个人
        的部分——画像、这个人说过的话、以及机器人紧接着对这些话的回复；群的L0滚动
        摘要里可能也写到了这个人，一并清掉（摘要之后会按剩下的对话重新累积）。
        返回删掉的对话条数。Chroma里对应的向量在下次检索前的对账里自动删掉
        （索引是SQLite的派生数据）。私聊的 /forget 直接用 reset_user()。"""
        rows = self.conn.execute(
            "SELECT turn_index, role, speaker FROM conversation_turns WHERE user_id = ? ORDER BY turn_index",
            (conversation_id,),
        ).fetchall()
        doomed = []
        for i, r in enumerate(rows):
            if r["role"] == "user" and r["speaker"] == member_id:
                doomed.append(r["turn_index"])
                if i + 1 < len(rows) and rows[i + 1]["role"] == "assistant":
                    doomed.append(rows[i + 1]["turn_index"])
        if doomed:
            self.conn.executemany(
                "DELETE FROM conversation_turns WHERE user_id = ? AND turn_index = ?",
                [(conversation_id, ti) for ti in doomed],
            )
        self.conn.execute("DELETE FROM user_profile WHERE user_id = ?", (member_id,))
        self.conn.execute("DELETE FROM display_name WHERE user_id = ?", (member_id,))
        self.conn.execute("DELETE FROM l0_summary WHERE user_id = ?", (conversation_id,))
        self.conn.commit()
        return len(doomed)

    def close(self):
        self.conn.close()


def _synchronized(cls):
    """给类里每个普通方法套上 self._lock（__init__ 除外，锁就是在里面建的）。
    比在每个方法里手写 with self._lock 不容易漏：以后新加的方法自动带锁。"""
    def wrap(fn):
        @functools.wraps(fn)
        def locked(self, *args, **kwargs):
            with self._lock:
                return fn(self, *args, **kwargs)
        return locked

    for name, fn in list(vars(cls).items()):
        if inspect.isfunction(fn) and name != "__init__":
            setattr(cls, name, wrap(fn))
    return cls


_synchronized(MemoryStore)


if __name__ == "__main__":
    # 自测：内存数据库，写几条数据验证读写逻辑
    store = MemoryStore(db_path=":memory:")

    store.upsert_profile_fact("demo_user", "研究方向", "生物信息学/单细胞谱系追踪",
                               source_text="我是做生物信息学的，具体方向是单细胞谱系追踪")
    store.upsert_profile_fact("demo_user", "研究方向", "单细胞谱系追踪（肝脏纤维化）",
                               source_text="补充一下，我做的具体是肝脏纤维化方向")
    print("画像：", store.get_profile("demo_user"))

    store.append_turn("demo_user", "user", "我是做生物信息学的")
    store.append_turn("demo_user", "assistant", "了解，请问有什么可以帮你的？")
    idx = store.append_turn("demo_user", "user", "我的研究方向是单细胞谱系追踪")
    print(f"第三轮的 turn_index = {idx}")

    print("最近2轮：")
    for row in store.get_recent_turns("demo_user", limit=2):
        print(f"  [{row['turn_index']}] {row['role']}: {row['text']}")

    print(f"完整历史共 {len(store.get_all_turns('demo_user'))} 轮")

    print(f"L0摘要（还没写入过）：{store.get_l0_summary('demo_user')}")
    store.set_l0_summary("demo_user", "用户是生物信息学研究生，做单细胞谱系追踪方向。", summarized_up_to=2)
    print(f"L0摘要（写入后）：{store.get_l0_summary('demo_user')}")

    store.reset_user("demo_user")
    print(f"reset后画像：{store.get_profile('demo_user')}，历史条数：{len(store.get_all_turns('demo_user'))}，"
          f"L0摘要：{store.get_l0_summary('demo_user')}")
