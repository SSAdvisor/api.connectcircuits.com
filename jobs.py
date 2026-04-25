"""
jobs.py — Redis-backed async job queue for ConnectCircuits API

Job lifecycle:
  PENDING → RUNNING → COMPLETE | FAILED

Each job stored as a Redis hash with TTL.
Results (binary files) stored on disk via store.py.
Webhook delivery is fire-and-forget via httpx after job completes.
"""

import os
import uuid
import time
import json
import asyncio
import logging
import ipaddress
from urllib.parse import urlparse
from typing import Optional, Callable, Awaitable, Any

import httpx
import redis.asyncio as aioredis

from store import save_result, log_usage, cleanup_expired_results, cleanup_old_usage_logs

logger = logging.getLogger("jobs")

REDIS_URL      = os.getenv("REDIS_URL", "redis://redis:6379/0")
JOB_TTL_SEC    = int(os.getenv("JOB_TTL_SEC", str(60 * 60 * 6)))    # 6 hours
WEBHOOK_TIMEOUT= float(os.getenv("WEBHOOK_TIMEOUT_SEC", "10"))
WEBHOOK_RETRIES= int(os.getenv("WEBHOOK_RETRIES", "3"))

# Per-user concurrency cap (per API key, across all endpoints)
CONCURRENCY_CAP = int(os.getenv("USER_CONCURRENCY_CAP", "3"))

# Global worker semaphore — limits total parallel heavy jobs on this node
GLOBAL_WORKER_SEM = asyncio.Semaphore(int(os.getenv("GLOBAL_WORKER_CONCURRENCY", "5")))

_redis: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = await aioredis.from_url(
            REDIS_URL,
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=5,
            socket_keepalive=True,
        )
    return _redis


async def close_redis() -> None:
    """Close the Redis connection pool. Call on application shutdown."""
    global _redis
    if _redis is not None:
        await _redis.close()
        _redis = None
        logger.info("Redis connection closed.")


def is_safe_webhook_url(url: str) -> bool:
    """Validate webhook URL to prevent SSRF attacks.
    Rejects private/internal IPs and non-HTTP(S) schemes."""
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
            # hostname is a domain name, not an IP — allow it
            pass
        return True
    except Exception:
        return False


async def check_redis_health() -> bool:
    """Returns True if Redis is reachable and responsive."""
    try:
        r = await get_redis()
        result = await r.ping()
        return result is True
    except Exception:
        return False


async def periodic_cleanup(interval_sec: int = 3600):
    """Run cleanup tasks periodically. Spawn as a background task."""
    while True:
        try:
            removed = cleanup_expired_results()
            if removed > 0:
                logger.info(f"Cleanup: removed {removed} expired result files.")
            logs_deleted = cleanup_old_usage_logs(days=30)
            if logs_deleted > 0:
                logger.info(f"Cleanup: removed {logs_deleted} old usage log entries.")
        except Exception as e:
            logger.exception(f"Cleanup task error: {e}")
        await asyncio.sleep(interval_sec)


# -------------------------------------------------------
# Job key helpers
# -------------------------------------------------------

def _job_key(job_id: str) -> str:
    return f"cc:job:{job_id}"

def _user_active_key(key_hash: str) -> str:
    return f"cc:active:{key_hash}"


# -------------------------------------------------------
# Job CRUD
# -------------------------------------------------------

async def create_job(
    endpoint: str,
    user_key_hash: str,
    webhook_url: Optional[str],
    request_payload: dict,
) -> str:
    """Create a PENDING job in Redis. Returns job_id."""
    if webhook_url and not is_safe_webhook_url(webhook_url):
        raise ValueError(f"Invalid webhook URL: {webhook_url}")

    r       = await get_redis()
    job_id  = str(uuid.uuid4())
    now     = int(time.time())

    job_data = {
        "job_id":         job_id,
        "endpoint":       endpoint,
        "user_key_hash":  user_key_hash,
        "webhook_url":    webhook_url or "",
        "status":         "pending",
        "created_at":     now,
        "started_at":     "",
        "completed_at":   "",
        "error":          "",
        "result_url":     "",
        "result_headers": json.dumps({}),
        "request_meta":   json.dumps({k: v for k, v in request_payload.items()
                                      if isinstance(v, (str, int, float, bool, type(None)))}),
    }

    pipe = r.pipeline()
    pipe.hset(_job_key(job_id), mapping=job_data)
    pipe.expire(_job_key(job_id), JOB_TTL_SEC)
    pipe.incr(_user_active_key(user_key_hash))
    pipe.expire(_user_active_key(user_key_hash), JOB_TTL_SEC)
    await pipe.execute()

    return job_id


async def get_job(job_id: str) -> Optional[dict]:
    r   = await get_redis()
    raw = await r.hgetall(_job_key(job_id))
    if not raw:
        return None
    raw["result_headers"] = json.loads(raw.get("result_headers", "{}"))
    return raw


async def _update_job(job_id: str, **fields):
    r = await get_redis()
    await r.hset(_job_key(job_id), mapping={k: str(v) for k, v in fields.items()})


async def _decrement_user_active(user_key_hash: str):
    r   = await get_redis()
    val = await r.decr(_user_active_key(user_key_hash))
    if int(val) < 0:
        await r.set(_user_active_key(user_key_hash), 0)


async def get_user_active_count(user_key_hash: str) -> int:
    r   = await get_redis()
    val = await r.get(_user_active_key(user_key_hash))
    return int(val) if val else 0


# -------------------------------------------------------
# Concurrency guard
# -------------------------------------------------------

async def check_user_concurrency(user_key_hash: str) -> bool:
    """Returns True if user is under the cap, False if at/over it."""
    count = await get_user_active_count(user_key_hash)
    return count < CONCURRENCY_CAP


# -------------------------------------------------------
# Webhook delivery
# -------------------------------------------------------

async def _fire_webhook(job_id: str, webhook_url: str, payload: dict):
    if not is_safe_webhook_url(webhook_url):
        logger.error(f"Webhook URL validation failed for job={job_id}: {webhook_url}")
        return

    for attempt in range(1, WEBHOOK_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
                resp = await client.post(webhook_url, json=payload)
                if resp.status_code < 400:
                    logger.info(f"Webhook delivered job={job_id} attempt={attempt} status={resp.status_code}")
                    return
                logger.warning(f"Webhook job={job_id} attempt={attempt} HTTP {resp.status_code}")
        except Exception as e:
            logger.warning(f"Webhook job={job_id} attempt={attempt} error: {e}")
        if attempt < WEBHOOK_RETRIES:
            await asyncio.sleep(2 ** attempt)

    logger.error(f"Webhook FAILED for job={job_id} after {WEBHOOK_RETRIES} attempts")


# -------------------------------------------------------
# Job runner — wraps any async task function
# -------------------------------------------------------

async def run_job(
    job_id: str,
    user_key_hash: str,
    raw_key: str,
    endpoint: str,
    task_fn: Callable[[], Awaitable[tuple]],
    result_filename: str,
    public_base_url: str,
):
    """
    Execute task_fn() under the global semaphore.

    task_fn must return (bytes, content_type, extra_headers_dict).

    On completion:
      - Saves result to disk
      - Updates job status in Redis
      - Fires webhook if configured
      - Logs usage
    """
    r          = await get_redis()
    webhook_url= (await r.hget(_job_key(job_id), "webhook_url")) or ""

    async with GLOBAL_WORKER_SEM:
        await _update_job(job_id, status="running", started_at=int(time.time()))
        log_usage(raw_key, endpoint, job_id, "running")

        try:
            result_bytes, content_type, extra_headers = await task_fn()

            save_result(job_id, result_bytes, content_type, result_filename)
            result_url     = f"{public_base_url.rstrip('/')}/v1/jobs/{job_id}/result"
            now            = int(time.time())

            await _update_job(
                job_id,
                status="complete",
                completed_at=now,
                result_url=result_url,
                result_headers=json.dumps(extra_headers),
            )
            log_usage(raw_key, endpoint, job_id, "complete")

            if webhook_url:
                job_snapshot = await get_job(job_id)
                webhook_payload = {
                    "job_id":       job_id,
                    "status":       "complete",
                    "endpoint":     endpoint,
                    "result_url":   result_url,
                    "completed_at": now,
                    "headers":      extra_headers,
                }
                asyncio.create_task(_fire_webhook(job_id, webhook_url, webhook_payload))

        except Exception as exc:
            logger.exception(f"Job {job_id} failed: {exc}")
            await _update_job(
                job_id,
                status="failed",
                completed_at=int(time.time()),
                error=str(exc)[:500],
            )
            log_usage(raw_key, endpoint, job_id, "failed")

            if webhook_url:
                asyncio.create_task(_fire_webhook(job_id, webhook_url, {
                    "job_id":   job_id,
                    "status":   "failed",
                    "endpoint": endpoint,
                    "error":    str(exc)[:500],
                }))

        finally:
            await _decrement_user_active(user_key_hash)
