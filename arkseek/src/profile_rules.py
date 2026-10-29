"""Optional administrator decisions on L1. Original quotes and L2 are preserved."""

SCHEMA = """
CREATE TABLE IF NOT EXISTS admin_profile_rules (
 user_id TEXT NOT NULL,key TEXT NOT NULL,mode TEXT NOT NULL,value TEXT,
 reason TEXT NOT NULL,updated_at TEXT NOT NULL,PRIMARY KEY(user_id,key));
CREATE TABLE IF NOT EXISTS admin_profile_history (
 id TEXT PRIMARY KEY,user_id TEXT NOT NULL,key TEXT NOT NULL,before_json TEXT NOT NULL,
 after_json TEXT NOT NULL,reason TEXT NOT NULL,created_at TEXT NOT NULL);
"""


def has_rules(conn):
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='admin_profile_rules'").fetchone() is not None


def rules_for(conn, user_id):
    if not has_rules(conn):
        return {}
    return {r["key"]: dict(r) for r in conn.execute("SELECT * FROM admin_profile_rules WHERE user_id=?", (user_id,))}


def effective_fact(row, rule=None):
    from profile_facts import supported_profile_fact
    item = dict(row)
    item.update(stored_value=row["value"], management=rule)
    if rule and rule["mode"] == "override":
        item["value"] = rule["value"]
    item["effective"] = not (rule and rule["mode"] == "disabled") and supported_profile_fact(item["key"], item["value"], row["source_text"] or "")
    return item
