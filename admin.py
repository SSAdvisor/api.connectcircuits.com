"""
admin.py  —  Internal admin endpoints + GUI for ConnectCircuits API

API routes:   POST/GET/DELETE /admin/keys
              GET /admin/usage
              GET /admin/usage/summary
              GET /admin/jobs
GUI route:    GET /admin/ui

Protect /admin/* behind a reverse-proxy allow-list or VPN in production.
"""

import os
import hmac
import pathlib
from fastapi import APIRouter, HTTPException, Header, Depends
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from store import (
    create_api_key,
    revoke_api_key,
    list_api_keys,
    _get_db,
    _hash_key,
)

ADMIN_SECRET    = os.getenv("ADMIN_SECRET", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://api.connectcircuits.com")

router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(x_admin_secret: str = Header(...)):
    if not ADMIN_SECRET:
        raise HTTPException(status_code=500, detail="ADMIN_SECRET not configured.")
    if not hmac.compare_digest(x_admin_secret.encode(), ADMIN_SECRET.encode()):
        raise HTTPException(status_code=403, detail="Forbidden.")
    return True


class CreateKeyRequest(BaseModel):
    user_label: str
    tier: Optional[str] = "standard"


class RevokeKeyRequest(BaseModel):
    raw_key: str


@router.post("/keys")
def admin_create_key(req: CreateKeyRequest, _=Depends(_require_admin)):
    """Create a new API key. Returns the raw key ONCE."""
    raw_key = create_api_key(user_label=req.user_label, tier=req.tier or "standard")
    return {
        "raw_key":    raw_key,
        "user_label": req.user_label,
        "tier":       req.tier,
        "note":       "This is the only time the raw key is shown. Store it securely.",
    }


@router.get("/keys")
def admin_list_keys(_=Depends(_require_admin)):
    keys = list_api_keys()
    redacted = []
    for k in keys:
        entry = dict(k)
        entry.pop("key_hash", None)
        redacted.append(entry)
    return redacted


@router.delete("/keys")
def admin_revoke_key(req: RevokeKeyRequest, _=Depends(_require_admin)):
    revoke_api_key(req.raw_key)
    return {"status": "revoked"}


@router.get("/usage")
def admin_usage(
    user_label: Optional[str] = None,
    limit: int = 100,
    _=Depends(_require_admin),
):
    conn = _get_db()
    if user_label:
        rows = conn.execute("""
            SELECT u.ts, u.endpoint, u.job_id, u.status, k.user_label, k.tier
            FROM usage_log u
            JOIN api_keys k ON u.key_hash = k.key_hash
            WHERE k.user_label = ?
            ORDER BY u.ts DESC LIMIT ?
        """, (user_label, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT u.ts, u.endpoint, u.job_id, u.status, k.user_label, k.tier
            FROM usage_log u
            LEFT JOIN api_keys k ON u.key_hash = k.key_hash
            ORDER BY u.ts DESC LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/usage/summary")
def admin_usage_summary(_=Depends(_require_admin)):
    conn = _get_db()
    rows = conn.execute("""
        SELECT k.user_label, k.tier, u.endpoint, u.status, COUNT(*) as count
        FROM usage_log u
        LEFT JOIN api_keys k ON u.key_hash = k.key_hash
        GROUP BY k.user_label, k.tier, u.endpoint, u.status
        ORDER BY k.user_label, u.endpoint
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/jobs")
def admin_jobs(
    status: Optional[str] = None,
    user_label: Optional[str] = None,
    limit: int = 50,
    _=Depends(_require_admin),
):
    """List recent jobs with optional filters."""
    conn = _get_db()
    conditions = []
    params = []

    if status:
        conditions.append("j.status = ?")
        params.append(status)
    if user_label:
        conditions.append("k.user_label = ?")
        params.append(user_label)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    params.append(limit)

    rows = conn.execute(f"""
        SELECT j.job_id, j.endpoint, j.status, j.created_at, j.started_at,
               j.completed_at, j.error, k.user_label, k.tier
        FROM jobs j
        LEFT JOIN api_keys k ON j.user_key_hash = k.key_hash
        {where}
        ORDER BY j.created_at DESC
        LIMIT ?
    """, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def admin_ui():
    """
    Serves the admin panel HTML from admin_ui.html.
    __API_BASE__ tokens are replaced at request time with PUBLIC_BASE_URL.
    """
    html_path = pathlib.Path(__file__).parent / "admin_ui.html"
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("__API_BASE__", PUBLIC_BASE_URL.rstrip("/"))
    return HTMLResponse(content=html)
