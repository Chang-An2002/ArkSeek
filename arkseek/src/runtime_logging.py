"""将机器人控制台输出同步保存到本机滚动日志，不改变终端显示。"""

import atexit
import logging
import os
import re
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path


class ConsoleTee:
    def __init__(self, original, handler, secrets=()):
        self.original = original
        self.handler = handler
        self.secrets = tuple(secret for secret in secrets if secret)
        self.lock = threading.RLock()
        self.pending = {}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def _save(self, line):
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line)
        for secret in self.secrets:
            line = line.replace(secret, "[REDACTED]")
        self.handler.handle(logging.LogRecord("console", logging.INFO, "", 0, line, (), None))

    def write(self, text):
        with self.lock:
            result = self.original.write(text)
            thread_id = threading.get_ident()
            lines = (self.pending.get(thread_id, "") + text).split("\n")
            self.pending[thread_id] = lines.pop()
            for line in lines:
                self._save(line.rstrip("\r"))
            return result

    def flush(self):
        with self.lock:
            self.original.flush()
            self.handler.flush()

    def finish(self):
        with self.lock:
            for line in self.pending.values():
                if line:
                    self._save(line)
            self.pending.clear()
            self.flush()


def install_runtime_logging(directory=None):
    if isinstance(sys.stdout, ConsoleTee):
        return sys.stdout.handler.baseFilename
    directory = Path(directory) if directory else Path(__file__).resolve().parent.parent / "data" / "runtime_logs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"bot_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}.log"
    handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    secrets = [value for key, value in os.environ.items()
               if any(marker in key.upper() for marker in ("API_KEY", "SECRET", "TOKEN")) and len(value) >= 8]
    original_out, original_err = sys.stdout, sys.stderr
    out, err = ConsoleTee(original_out, handler, secrets), ConsoleTee(original_err, handler, secrets)
    sys.stdout, sys.stderr = out, err

    def close():
        out.finish()
        err.finish()
        if sys.stdout is out:
            sys.stdout = original_out
        if sys.stderr is err:
            sys.stderr = original_err
        handler.close()

    atexit.register(close)
    print(f"[runtime_logging] 完整控制台日志：{path}")
    return str(path)
