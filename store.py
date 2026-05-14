"""
store.py  —  SQLite-backed API key store, usage log, and job queue for ConnectCircuits API

Schema
------
api_keys   : hashed keys, user labels, tiers, status, timestamps
usage_log  : per-request log (key_hash, endpoint, job_id, status, ts)
jobs       : async job state (lifecycle: pending → running → complete | failed)

All raw keys are hashed with SHA-256 before storage.
The raw key is returned ONCE at creation and never stored.
"""

import hashlib
import secrets
import sqlite3
import os
import time
import uuid
import json
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv("DB_PATH", "/app/data/store.db"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/app/data/results"))
RESULT_TTL_SEC = int(os.getenv("RESULT_TTL_SEC", str(60 * 60 * 6)))
JOB_TTL_SEC = int(os.getenv("JOB_TTL_SEC", str(60 * 60 * 6)))

# Per-user concurrency cap
CONCURRENCY_CAP = int(os.getenv("USER_CONCURRENCY_CAP", "3"))


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    _ensure_schema(conn)
    return conn



def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Idempotent ALTER TABLE — only adds the column if it doesn't already exist."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash    TEXT    NOT NULL UNIQUE,
            user_label  TEXT    NOT NULL,
            tier        TEXT    NOT NULL DEFAULT 'standard',
            status      TEXT    NOT NULL DEFAULT 'active',
            created_at  TEXT    NOT NULL DEFAULT (datetime('now')),
            last_seen   TEXT
        );

        CREATE TABLE IF NOT EXISTS usage_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash    TEXT,
            endpoint    TEXT,
            job_id      TEXT,
            status      TEXT,
            ts          TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS jobs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id          TEXT    NOT NULL UNIQUE,
            endpoint        TEXT    NOT NULL,
            user_key_hash   TEXT    NOT NULL,
            webhook_url     TEXT    NOT NULL DEFAULT '',
            status          TEXT    NOT NULL DEFAULT 'pending',
            created_at      INTEGER NOT NULL,
            started_at      INTEGER,
            completed_at    INTEGER,
            error           TEXT    NOT NULL DEFAULT '',
            result_url      TEXT    NOT NULL DEFAULT '',
            result_headers  TEXT    NOT NULL DEFAULT '{}',
            request_meta    TEXT    NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_api_keys_hash    ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_usage_key_hash   ON usage_log(key_hash);
        CREATE INDEX IF NOT EXISTS idx_usage_ts         ON usage_log(ts);
        CREATE INDEX IF NOT EXISTS idx_jobs_job_id      ON jobs(job_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_user_hash   ON jobs(user_key_hash);
        CREATE INDEX IF NOT EXISTS idx_jobs_status      ON jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_created_at  ON jobs(created_at);
    """)
    # ── Safe migrations — idempotent, run on every startup ───────────────────
    _add_column_if_missing(conn, "jobs", "queue_position", "INTEGER")
    _add_column_if_missing(conn, "jobs", "request_meta",   "TEXT NOT NULL DEFAULT '{}'")
    conn.commit()


def _hash_key(raw_key: str) -> str:
    """SHA-256 hex digest — used identically at creation and verification."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ── Result Storage ─────────────────────────────────────────────────────────────

def save_result(job_id: str, data: bytes, content_type: str, filename: str) -> None:
    """Save a job result to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ext_map = {
        "video/mp4": ".mp4",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/opus": ".opus",
        "audio/flac": ".flac",
        "image/png": ".png",
        "image/jpeg": ".jpg",
    }
    ext = ext_map.get(content_type, ".bin")
    result_path = RESULTS_DIR / f"{job_id}{ext}"
    with open(result_path, "wb") as f:
        f.write(data)


def load_result(job_id: str):
    """Load a completed job result from disk.
    Returns (bytes, content_type, filename) or None if not found."""
    import glob
    pattern = str(RESULTS_DIR / f"{job_id}.*")
    matches = glob.glob(pattern)
    if not matches:
        return None
    result_path = matches[0]
    ext = os.path.splitext(result_path)[1].lower()
    mime_map = {
        ".mp4":  ("video/mp4",   "output.mp4"),
        ".mp3":  ("audio/mpeg",  "audio.mp3"),
        ".wav":  ("audio/wav",   "audio.wav"),
        ".opus": ("audio/opus",  "audio.opus"),
        ".flac": ("audio/flac",  "audio.flac"),
        ".png":  ("image/png",   "image.png"),
        ".jpg":  ("image/jpeg",  "image.jpg"),
        ".jpeg": ("image/jpeg",  "image.jpg"),
    }
    content_type, filename = mime_map.get(ext, ("application/octet-stream", f"result{ext}"))
    with open(result_path, "rb") as f:
        return f.read(), content_type, filename


def cleanup_expired_results() -> int:
    """Remove result files older than RESULT_TTL_SEC. Returns count removed."""
    if not RESULTS_DIR.exists():
        return 0
    now = time.time()
    removed = 0
    for f in RESULTS_DIR.iterdir():
        if f.is_file():
            age = now - f.stat().st_mtime
            if age > RESULT_TTL_SEC:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


def cleanup_old_usage_logs(days: int = 30) -> int:
    """Delete usage_log entries older than N days. Returns count removed."""
    conn = _get_db()
    cursor = conn.execute(
        "DELETE FROM usage_log WHERE ts < datetime('now', ?)",
        (f"-{days} days",)
    )
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def cleanup_old_jobs(days: int = 7) -> int:
    """Delete completed/failed jobs older than N days. Returns count removed."""
    conn = _get_db()
    cutoff = int(time.time()) - (days * 86400)
    cursor = conn.execute(
        "DELETE FROM jobs WHERE status IN ('complete', 'failed') AND completed_at < ?",
        (cutoff,)
    )
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


# ── Key Management ─────────────────────────────────────────────────────────────

def create_api_key(user_label: str, tier: str = "standard") -> str:
    """
    Generate a new API key, store its hash with status='active', return raw key.
    The raw key is NEVER stored — caller must persist it immediately.
    """
    raw_key  = "cc-" + secrets.token_urlsafe(32)
    key_hash = _hash_key(raw_key)
    conn = _get_db()
    conn.execute(
        """INSERT INTO api_keys (key_hash, user_label, tier, status, created_at)
           VALUES (?, ?, ?, 'active', datetime('now'))""",
        (key_hash, user_label, tier),
    )
    conn.commit()
    conn.close()
    return raw_key


def revoke_api_key(raw_key: str) -> None:
    """Mark a key as revoked by its raw value."""
    key_hash = _hash_key(raw_key)
    conn = _get_db()
    conn.execute(
        "UPDATE api_keys SET status = 'revoked' WHERE key_hash = ?",
        (key_hash,),
    )
    conn.commit()
    conn.close()


def list_api_keys() -> list:
    """Return all keys as a list of dicts (no raw key — hashes only)."""
    conn = _get_db()
    rows = conn.execute(
        """SELECT id, user_label, tier, status, created_at, last_seen
           FROM api_keys
           ORDER BY created_at DESC"""
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def verify_api_key(raw_key: str) -> Optional[dict]:
    """
    Validate a raw key.
    Returns the key row dict if active, None if missing or revoked.
    Also updates last_seen on success.
    """
    if not raw_key:
        return None
    key_hash = _hash_key(raw_key)
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM api_keys WHERE key_hash = ? AND status = 'active'",
        (key_hash,),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE api_keys SET last_seen = datetime('now') WHERE key_hash = ?",
            (key_hash,),
        )
        conn.commit()
    conn.close()
    return dict(row) if row else None


# ── Job Queue ──────────────────────────────────────────────────────────────────

def create_job(
    endpoint: str,
    user_key_hash: str,
    webhook_url: Optional[str],
    request_payload: dict,
) -> str:
    """Create a PENDING job in SQLite. Returns job_id."""
    # Validate webhook URL if provided
    if webhook_url and not is_safe_webhook_url(webhook_url):
        raise ValueError(f"Invalid webhook URL: {webhook_url}")

    job_id = str(uuid.uuid4())
    now = int(time.time())

    request_meta = json.dumps({
        k: v for k, v in request_payload.items()
        if isinstance(v, (str, int, float, bool, type(None)))
    })

    conn = _get_db()
    conn.execute(
        """INSERT INTO jobs
           (job_id, endpoint, user_key_hash, webhook_url, status,
            created_at, result_headers, request_meta)
           VALUES (?, ?, ?, ?, 'pending', ?, '{}', ?)""",
        (job_id, endpoint, user_key_hash, webhook_url or "", now, request_meta),
    )
    conn.commit()
    conn.close()
    return job_id


def get_job(job_id: str) -> Optional[dict]:
    """Retrieve a job by its ID."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    try:
        result["result_headers"] = json.loads(result.get("result_headers", "{}"))
    except (json.JSONDecodeError, TypeError):
        result["result_headers"] = {}
    return result


def get_job_by_user(job_id: str, user_key_hash: str) -> Optional[dict]:
    """Retrieve a job only if it belongs to the given user."""
    conn = _get_db()
    row = conn.execute(
        "SELECT * FROM jobs WHERE job_id = ? AND user_key_hash = ?",
        (job_id, user_key_hash),
    ).fetchone()
    conn.close()
    if not row:
        return None
    result = dict(row)
    try:
        result["result_headers"] = json.loads(result.get("result_headers", "{}"))
    except (json.JSONDecodeError, TypeError):
        result["result_headers"] = {}
    return result


def update_job(job_id: str, **fields) -> None:
    """Update arbitrary fields on a job."""
    if not fields:
        return
    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [job_id]
    conn = _get_db()
    conn.execute(f"UPDATE jobs SET {set_clause} WHERE job_id = ?", values)
    conn.commit()
    conn.close()


def get_user_active_job_count(user_key_hash: str) -> int:
    """Count how many jobs a user has in pending or running state."""
    conn = _get_db()
    row = conn.execute(
        """SELECT COUNT(*) as cnt FROM jobs
           WHERE user_key_hash = ? AND status IN ('pending', 'running')""",
        (user_key_hash,),
    ).fetchone()
    conn.close()
    return row["cnt"] if row else 0


def check_user_concurrency(user_key_hash: str) -> bool:
    """Returns True if user is under the concurrency cap."""
    count = get_user_active_job_count(user_key_hash)
    return count < CONCURRENCY_CAP


# ── Webhook URL Validation ─────────────────────────────────────────────────────

def is_safe_webhook_url(url: str) -> bool:
    """Validate webhook URL to prevent SSRF attacks."""
    from urllib.parse import urlparse
    import ipaddress
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname
        if not hostname:
            return False
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_reserved or ip.is_loopback or ip.is_link_local:
                return False
        except ValueError:
            pass
        return True
    except Exception:
        return False


# ── Usage Logging ──────────────────────────────────────────────────────────────

def log_usage(
    raw_key: Optional[str],
    endpoint: str,
    job_id:   Optional[str] = None,
    status:   str = "queued",
) -> None:
    """
    Record a single API request.  key_hash may be None for unauthenticated
    requests that were rejected (so we still count 401s).
    """
    if raw_key and not isinstance(raw_key, str):
        return
    key_hash = _hash_key(raw_key) if raw_key else None
    conn = _get_db()
    conn.execute(
        """INSERT INTO usage_log (key_hash, endpoint, job_id, status, ts)
           VALUES (?, ?, ?, ?, datetime('now'))""",
        (key_hash, endpoint, job_id, status),
    )
    conn.commit()
    conn.close()


def update_usage_status(job_id: str, status: str) -> None:
    """Update the status of a usage_log entry when a job completes or fails."""
    conn = _get_db()
    conn.execute(
        "UPDATE usage_log SET status = ? WHERE job_id = ?",
        (status, job_id),
    )
    conn.commit()
    conn.close()


# ── Periodic Cleanup ───────────────────────────────────────────────────────────

def run_cleanup() -> dict:
    """Run all cleanup tasks. Returns a summary dict."""
    results = {}
    results["expired_results"] = cleanup_expired_results()
    results["old_usage_logs"] = cleanup_old_usage_logs(days=30)
    results["old_jobs"] = cleanup_old_jobs(days=7)
    return results
