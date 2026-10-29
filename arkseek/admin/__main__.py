import argparse
import sys
import tempfile
from pathlib import Path

from aiohttp import web

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))

from .demo import create_demo
from .server import create_app


def main():
    parser = argparse.ArgumentParser(description="ArkSeek 本机管理后台（真实模式默认只读）")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--demo", action="store_true", help="虚构演示数据（默认）")
    modes.add_argument("--live", action="store_true", help="只读接入本机真实记忆和运行日志")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-dir", type=Path, help="--live 使用的数据目录，默认项目 data")
    parser.add_argument("--manage", action="store_true", help="显式启用真实管理；所有写入仍需预览确认。演示自动启用虚构管理")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or (args.data_dir and not args.live):
        parser.error("端口须为1–65535；自定义数据目录只用于 --live")
    with tempfile.TemporaryDirectory(prefix="arkseek-admin-") as tmp:
        demo = not args.live
        directory = Path(tmp) if demo else (args.data_dir or BASE / "data").resolve()
        if demo:
            create_demo(directory)
        print("ArkSeek 管理后台 · " + ("虚构演示数据" if demo else "真实本机数据（只读）"))
        print("关闭此进程只关闭后台页面服务，QQ机器人独立运行。")
        web.run_app(create_app(directory, demo=demo, manage=demo or args.manage), host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
