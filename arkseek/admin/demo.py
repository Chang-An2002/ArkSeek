"""Synthetic fixtures only. Never reads project databases, logs or .env."""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from admin_monitor import SCHEMA


def create_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = lambda seconds: (now - timedelta(seconds=seconds)).isoformat(timespec="milliseconds")
    with closing(sqlite3.connect(directory / "memory.db")) as conn:
        conn.executescript("""
        CREATE TABLE user_profile (user_id TEXT,key TEXT,value TEXT,source_text TEXT,updated_at TEXT,PRIMARY KEY(user_id,key));
        CREATE TABLE display_name (user_id TEXT PRIMARY KEY,name TEXT,updated_at TEXT);
        CREATE TABLE l0_summary (user_id TEXT PRIMARY KEY,summary TEXT,summarized_up_to INTEGER,updated_at TEXT);
        CREATE TABLE conversation_turns (id INTEGER PRIMARY KEY,user_id TEXT,turn_index INTEGER,role TEXT,text TEXT,created_at TEXT,is_general_chat INTEGER DEFAULT 0,speaker TEXT);
        """)
        group, a, b, private = "qq_group:demo", "qq_group:demo:a", "qq_group:demo:b", "qq_c2c:demo"
        conn.executemany("INSERT INTO display_name VALUES (?,?,?)", [(a, "成员甲", stamp(100)), (b, "成员乙", stamp(100)), (private, "演示用户", stamp(100))])
        conn.executemany("INSERT INTO user_profile VALUES (?,?,?,?,?)", [(a, "喜欢", "策略游戏", "我平时喜欢策略游戏。", stamp(120)),
                        (b, "称呼", "未提供", "", stamp(120)), (private, "回答偏好", "先给结论，再解释", "请先给结论，再解释。", stamp(120))])
        texts = [(group, "user", "我平时喜欢策略游戏。", a), (group, "assistant", "记下了，你喜欢策略游戏。", a),
                 (group, "user", "解释一下混合检索。", b), (group, "assistant", "关键词检索和语义检索各找一批材料，再融合排序，最后精排筛选证据。", b),
                 (group, "user", "太专业了，请通俗说一下。", b), (group, "assistant", "相当于两位图书管理员分别按关键词和意思找书，再一起筛选最有用的材料。\n引用来源：演示资料，不是真实知识库。", b),
                 (private, "user", "请先给结论，再解释。", None), (private, "assistant", "好的，之后按这个顺序组织回答。", None)]
        indexes = {}
        for n, (cid, role, text, speaker) in enumerate(texts):
            idx = indexes.get(cid, 0)
            conn.execute("INSERT INTO conversation_turns(user_id,turn_index,role,text,created_at,speaker) VALUES (?,?,?,?,?,?)", (cid, idx, role, text, stamp(90 - n * 8), speaker))
            indexes[cid] = idx + 1
        conn.execute("INSERT INTO l0_summary VALUES (?,?,?,?)", (group, "成员甲分享了游戏偏好；成员乙询问混合检索并要求通俗解释。此摘要为演示话题索引。", 5, stamp(10)))
        conn.commit()
    with closing(sqlite3.connect(directory / "admin_monitor.db")) as conn:
        conn.executescript(SCHEMA)
        conn.execute("INSERT INTO runtime VALUES ('demo',0,?,?,'stopped',0)", (stamp(300), stamp(10)))
        for i, (status, wait, duration) in enumerate([("sent", 0, 5200), ("sent", 2100, 18200), ("send_failed", 0, 7400), ("processing_failed", 300, 9200)]):
            received = now - timedelta(seconds=80 - i * 12)
            started = received + timedelta(milliseconds=wait)
            finished = received + timedelta(milliseconds=duration)
            spans = [{"start_ms": 0, "duration_ms": 1000, "name": "召回", "task": None}, {"start_ms": 1000, "duration_ms": 3500, "name": "LLM", "task": "q1"}, {"start_ms": 1600, "duration_ms": 2400, "name": "LLM", "task": "画像判断"}, {"start_ms": 4500, "duration_ms": 400, "name": "发送", "task": None}]
            conn.execute("INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (f"demo-{i}", "demo", group, a if i % 2 == 0 else b, received.isoformat(), started.isoformat(), finished.isoformat(), status, wait, duration, "DemoError" if status != "sent" else None, json.dumps(spans)))
        conn.commit()
    logs = directory / "runtime_logs"
    logs.mkdir()
    (logs / "bot_demo.log").write_text("[演示] 以下是虚构运行日志，不对应真实QQ群测试。\n[qq_bot] 对话队列等待 2.1 秒\n[qq_bot] 已回复\n[qq_bot] 本条接收至发送 18.2 秒（含排队）\n[qq_bot] 发送失败：DemoError\n", encoding="utf-8")
