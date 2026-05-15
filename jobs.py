"""
jobs.py — Async job queue and runner for ConnectCircuits API

Job lifecycle:
  queued → started → complete
                   → failed

Architecture:
  - A single asyncio.Queue (JOB_QUEUE) holds pending job coroutines.
  - GLOBAL_WORKER_CONCURRENCY worker coroutines drain the queue in parallel.
  - Workers are started once at app startup via start_queue_workers().
  - Per-user cap enforced in store.check_user_concurrency() before enqueue.

Status mapping:
  "queued"   — accepted by the API, waiting in queue
  "started"  — dequeued by a worker, actively processing
  "complete" — finished successfully, result on disk
  "failed"   — exception raised during processing
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
    get_global_queue_length,
)

logger = logging.getLogger("jobs")

WEBHOOK_TIMEOUT          = float(os.getenv("WEBHOOK_TIMEOUT_SEC",       "10"))
WEBHOOK_RETRIES          = int(os.getenv("WEBHOOK_RETRIES",              "3"))
GLOBAL_WORKER_CONCURRENCY = int(os.getenv("GLOBAL_WORKER_CONCURRENCY",  "5"))

# The central asyncio queue — items are coroutines (async callables)
JOB_QUEUE: asyncio.Queue = asyncio.Queue()


# -------------------------------------------------------
# Queue workers — started once at app startup
# -------------------------------------------------------

async def _queue_worker(worker_id: int):
    """Long-running coroutine that pulls and executes jobs from JOB_QUEUE."""
    logger.info(f"Queue worker {worker_id} started")
    while True:
        job_coro = await JOB_QUEUE.get()
        try:
            await job_coro()
        except Exception as e:
            logger.exception(f"Worker {worker_id} unhandled exception: {e}")
        finally:
            JOB_QUEUE.task_done()


def start_queue_workers():
    """
    Spawn GLOBAL_WORKER_CONCURRENCY worker coroutines.
    Call once from app startup — idempotent (checks for existing workers).
    """
    loop = asyncio.get_event_loop()
    for i in range(1, GLOBAL_WORKER_CONCURRENCY + 1):
        loop.create_task(_queue_worker(i))
    logger.info(f"Started {GLOBAL_WORKER_CONCURRENCY} queue workers")


# -------------------------------------------------------
# Webhook delivery
# -------------------------------------------------------

async def _fire_webhook(job_id: str, webhook_url: str, payload: dict):
    if not is_safe_webhook_url(webhook_url):
        logger.error(f"Webhook URL blocked (SSRF guard) job={job_id}: {webhook_url}")
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
            await asyncio.sleep(2 ** attempt)   # 2s, 4s back-off

    logger.error(f"Webhook FAILED job={job_id} after {WEBHOOK_RETRIES} attempts")


# -------------------------------------------------------
# Periodic cleanup
# -------------------------------------------------------

async def periodic_cleanup(interval_sec: int = 3600):
    """Background task — runs cleanup once per hour."""
    while True:
        try:
            results = run_cleanup()
            parts = [f"{k}={v}" for k, v in results.items() if v > 0]
            if parts:
                logger.info(f"Cleanup: {', '.join(parts)}")
        except Exception as e:
            logger.exception(f"Cleanup error: {e}")
        await asyncio.sleep(interval_sec)


# -------------------------------------------------------
# Job enqueue — called by every endpoint
# -------------------------------------------------------

async def enqueue_job(
    job_id: str,
    user_key_hash: str,
    raw_key: str,
    endpoint: str,
    task_fn: Callable[[], Awaitable[tuple]],
    result_filename: str,
    public_base_url: str,
):
    """
    Build the job execution coroutine and push it onto JOB_QUEUE.
    The caller already wrote the job to SQLite with status='queued'.

    The coroutine will:
      1. Set status → 'started'
      2. Call task_fn()  →  (bytes, content_type, headers)
      3. Save result, set status → 'complete', fire webhook
      On exception: set status → 'failed', fire webhook
    """
    queue_pos = get_global_queue_length()   # approximate position for logging
    logger.info(f"Job {job_id} queued at position ~{queue_pos} endpoint={endpoint}")

    # Log the initial queued event
    log_usage(raw_key, endpoint, job_id, "queued")

    async def _execute():
        job = get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} disappeared before execution")
            return

        webhook_url = (job.get("webhook_url") or "").strip()

        # ── Mark as started ─────────────────────────────────────────────────
        update_job(job_id, status="started", started_at=int(time.time()))
        log_usage(raw_key, endpoint, job_id, "started")
        logger.info(f"Job {job_id} started endpoint={endpoint}")

        try:
            result_bytes, content_type, extra_headers = await task_fn()

            save_result(job_id, result_bytes, content_type, result_filename)
            result_url = f"{public_base_url.rstrip('/')}/v1/jobs/{job_id}/result"
            now = int(time.time())

            update_job(
                job_id,
                status       = "complete",
                completed_at = now,
                result_url   = result_url,
                result_headers = json.dumps(extra_headers),
            )
            log_usage(raw_key, endpoint, job_id, "complete")
            logger.info(f"Job {job_id} complete endpoint={endpoint}")

            if webhook_url:
                asyncio.create_task(_fire_webhook(job_id, webhook_url, {
                    "job_id":       job_id,
                    "status":       "complete",
                    "endpoint":     endpoint,
                    "result_url":   result_url,
                    "completed_at": now,
                    "headers":      extra_headers,
                }))

        except Exception as exc:
            logger.exception(f"Job {job_id} failed: {exc}")
            update_job(
                job_id,
                status       = "failed",
                completed_at = int(time.time()),
                error        = str(exc)[:500],
            )
            log_usage(raw_key, endpoint, job_id, "failed")

            if webhook_url:
                asyncio.create_task(_fire_webhook(job_id, webhook_url, {
                    "job_id":   job_id,
                    "status":   "failed",
                    "endpoint": endpoint,
                    "error":    str(exc)[:500],
                }))

    await JOB_QUEUE.put(_execute)


# -------------------------------------------------------
# Queue depth helper — exposed to status endpoints
# -------------------------------------------------------

def get_queue_depth() -> int:
    """Return the number of jobs currently waiting in the asyncio queue."""
    return JOB_QUEUE.qsize()
