"""Read administrator-owned text copies as L3 inputs. No access to source originals."""
import sqlite3
from contextlib import closing
from pathlib import Path


def load_admin_chunks(directory, start_chunk_id=0):
    path = (Path(directory) / "admin_control.db").resolve()
    if not path.is_file():
        return []
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("PRAGMA query_only=ON")
        if not conn.execute("SELECT 1 FROM sqlite_master WHERE name='documents'").fetchone():
            return []
        rows = conn.execute("SELECT id,title,body,revision FROM documents WHERE active=1 ORDER BY id").fetchall()
    from chunking import chunk_document
    chunks = []
    for doc_id, title, body, revision in rows:
        chunks.extend(chunk_document(body, f"管理员资料：{title}", {"doc_id": "admin:"+doc_id, "doc_type": "管理员资料", "title": title, "revision": revision}, start_chunk_id=start_chunk_id+len(chunks)))
    return chunks
