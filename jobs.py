"""
jobs.py — SQLite-backed async job runner for ConnectCircuits API

Job lifecycle:  PENDING → RUNNING → COMPLETE | FAILED

Job state lives in SQLite (store.py). This module provides the async
execution wrapper, webhook delivery, and background cleanup.

No Redis dependency.
"""

import os
import time
import json
import asyncio
import logging
from typing import Optional, Callable, Awaitable

import httpx

from store import (
    save_result,
    log_usage,
    get_job,
    update_job,
    run_cleanup,
    is_safe_webhook_url,
)

logger = logging.getLogger("jobs")

WEBHOOK_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT_SEC", "10"))
WEBHOOK_RETRIES = int(os.getenv("WEBHOOK_RETRIES", "3"))

# Global worker semaphore — limits total parallel heavy jobs on this node
GLOBAL_WORKER_CONCURRENCY = int(os.getenv("GLOBAL_WORKER_CONCURRENCY", "5"))
GLOBAL_WORKER_SEM = asyncio.Semaphore(GLOBAL_WORKER_CONCURRENCY)


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
                    logger.info(
                        f"Webhook delivered job={job_id} "
                        f"attempt={attempt} status={resp.status_code}"
                    )
                    return
                logger.warning(
                    f"Webhook job={job_id} attempt={attempt} HTTP {resp.status_code}"
                )
        except Exception as e:
            logger.warning(f"Webhook job={job_id} attempt={attempt} error: {e}")
        if attempt < WEBHOOK_RETRIES:
            await asyncio.sleep(2 ** attempt)  # exponential back-off: 2s, 4s

    logger.error(f"Webhook FAILED for job={job_id} after {WEBHOOK_RETRIES} attempts")


# -------------------------------------------------------
# Periodic cleanup
# -------------------------------------------------------

async def periodic_cleanup(interval_sec: int = 3600):
    """Run cleanup tasks periodically. Spawn as a background task."""
    while True:
        try:
            results = run_cleanup()
            parts = [f"{k}={v}" for k, v in results.items() if v > 0]
            if parts:
                logger.info(f"Cleanup: {', '.join(parts)}")
        except Exception as e:
            logger.exception(f"Cleanup task error: {e}")
        await asyncio.sleep(interval_sec)


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
      - Updates job status in SQLite
      - Fires webhook if configured
      - Logs usage
    """
    job = get_job(job_id)
    webhook_url = (job.get("webhook_url") if job else "") or ""

    async with GLOBAL_WORKER_SEM:
        update_job(job_id, status="running", started_at=int(time.time()))
        log_usage(raw_key, endpoint, job_id, "running")

        try:
            result_bytes, content_type, extra_headers = await task_fn()

            save_result(job_id, result_bytes, content_type, result_filename)
            result_url = f"{public_base_url.rstrip('/')}/v1/jobs/{job_id}/result"
            now = int(time.time())

            update_job(
                job_id,
                status="complete",
                completed_at=now,
                result_url=result_url,
                result_headers=json.dumps(extra_headers),
            )
            log_usage(raw_key, endpoint, job_id, "complete")

            if webhook_url:
                webhook_payload = {
                    "job_id":       job_id,
                    "status":       "complete",
                    "endpoint":     endpoint,
                    "result_url":   result_url,
                    "completed_at": now,
                    "headers":      extra_headers,
                }
                asyncio.create_task(
                    _fire_webhook(job_id, webhook_url, webhook_payload)
                )

        except Exception as exc:
            logger.exception(f"Job {job_id} failed: {exc}")
            update_job(
                job_id,
                status="failed",
                completed_at=int(time.time()),
                error=str(exc)[:500],
            )
            log_usage(raw_key, endpoint, job_id, "failed")

            if webhook_url:
                asyncio.create_task(
                    _fire_webhook(job_id, webhook_url, {
                        "job_id":   job_id,
                        "status":   "failed",
                        "endpoint": endpoint,
                        "error":    str(exc)[:500],
                    })
                )
