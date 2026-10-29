"""Fixed child entry points. Demo never imports the bot, models, .env or sources."""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "src"))
from runtime_settings import RunLease, apply_settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("bot", "index"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch("[0-9a-f]{32}", args.job_id):
        parser.error("invalid job identifier")
    root = args.data_dir.resolve()
    target = root / "admin_jobs" / (args.job_id+".json")

    def report(**data):
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps({"demo": args.demo, **data}, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)

    try:
        report(stage="starting")
        if args.demo:
            with RunLease(root):
                if args.kind == "bot":
                    report(stage="demo_running", note="虚构工作进程，无QQ连接")
                    while True:
                        time.sleep(.2)
                from admin_knowledge import load_admin_chunks
                report(stage="chunking")
                chunks = load_admin_chunks(root)
                report(stage="done", chunks=len(chunks), note="演示仅验证虚构文本切分，不加载模型或真实向量索引")
        else:
            if root != (BASE / "data").resolve():
                raise ValueError("Real workers require the default project data directory")
            apply_settings(root)
            if args.kind == "bot":
                from runtime_logging import install_runtime_logging
                install_runtime_logging()
                from qq_bot import main as bot_main
                report(stage="starting_bot")
                bot_main()  # Bot main owns the shared lease for its entire lifetime.
            else:
                with RunLease(root):
                    report(stage="loading_sources_models")
                    from pipeline import MemorySystemPipeline
                    from chroma_index import DEGRADE_NOTICES
                    pipeline = MemorySystemPipeline(str(root / "knowledge_base"))
                    report(stage="done", chunks=len(pipeline.chunks), dense=type(pipeline.dense).__name__,
                           reranker=type(pipeline.reranker).__name__, degraded=bool(DEGRADE_NOTICES),
                           note="L3已加载并同步派生索引；BM25在本进程内验证，下次启动时重建")
    except Exception as exc:
        report(stage="failed", error_kind=type(exc).__name__)
        print("Worker failed:", type(exc).__name__, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
