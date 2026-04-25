"""
store.py  —  SQLite-backed API key store + usage log for ConnectCircuits API

Schema
------
api_keys   : hashed keys, user labels, tiers, status, timestamps
usage_log  : per-request log (key_hash, endpoint, job_id, status, ts)

All raw keys are hashed with SHA-256 before storage.
The raw key is returned ONCE at creation and never stored.
"""

import hashlib
import secrets
import sqlite3
import os
import time
from pathlib import Path
from typing import Optional

DB_PATH = Path(os.getenv("DB_PATH", "/app/data/store.db"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", "/app/data/results"))
RESULT_TTL_SEC = int(os.getenv("RESULT_TTL_SEC", str(60 * 60 * 6)))


def _get_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


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

        CREATE INDEX IF NOT EXISTS idx_api_keys_hash   ON api_keys(key_hash);
        CREATE INDEX IF NOT EXISTS idx_usage_key_hash  ON usage_log(key_hash);
        CREATE INDEX IF NOT EXISTS idx_usage_ts        ON usage_log(ts);
    """)
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
