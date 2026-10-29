"""Small local administration helpers; no imports of models, .env or real databases."""
import functools
import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
SETTINGS = {
    "ARKSEEK_LLM_MAX_CONCURRENCY": ("4", 1, 8),
    "ARKSEEK_LLM_STALL_TIMEOUT": ("15", 5, 120),
    "ARKSEEK_LLM_TIMEOUT": ("120", 30, 600),
    "ARKSEEK_LLM_MAX_DURATION": ("240", 60, 900),
    "ARKSEEK_RERANKER": ("auto", ("auto", "openvino-gpu", "openvino-cpu", "torch")),
}


def validate_settings(values):
    if not isinstance(values, dict) or set(values) - SETTINGS.keys():
        raise ValueError("包含不支持的配置项；密钥不在后台编辑")
    result = {}
    for key, spec in SETTINGS.items():
        value = str(values.get(key, spec[0]))
        if len(spec) == 3:
            number = int(value)
            if not spec[1] <= number <= spec[2]:
                raise ValueError(f"{key} 超出允许范围 {spec[1]}–{spec[2]}")
            value = str(number)
        elif value not in spec[1]:
            raise ValueError("精排后端不正确")
        result[key] = value
    if int(result["ARKSEEK_LLM_MAX_DURATION"]) < int(result["ARKSEEK_LLM_TIMEOUT"]):
        raise ValueError("安全上限不能低于慢请求提示时间")
    return result


def read_settings(directory=DATA):
    path = (Path(directory) / "admin_control.db").resolve()
    if not path.is_file():
        return None
    with closing(sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='settings'").fetchone():
            return None
        row = conn.execute("SELECT body FROM settings WHERE id=1").fetchone()
    return validate_settings(json.loads(row[0])) if row else None


def apply_settings(directory=DATA):
    values = read_settings(directory)
    if values:
        os.environ.update(values)
    return values


class RuntimeBusy(RuntimeError):
    pass


class RunLease:
    """OS releases the lock after a crash. Bot/CLI/index jobs share one lease."""
    def __init__(self, directory=DATA):
        self.directory = Path(directory)
        self.file = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.file = (self.directory / "admin_runtime.lock").open("a+b")
        try:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                if self.file.read(1) == b"":
                    self.file.write(b"0"); self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close(); self.file = None
            raise RuntimeBusy("机器人、CLI或索引任务仍在运行；请先停止后再维护") from exc
        return self

    def __exit__(self, *exc):
        if self.file:
            self.file.close(); self.file = None


def exclusive_runtime(fn):
    @functools.wraps(fn)
    def run(*args, **kwargs):
        with RunLease():
            return fn(*args, **kwargs)
    return run
