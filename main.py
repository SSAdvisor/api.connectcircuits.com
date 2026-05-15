import os
import uuid
import asyncio
import hashlib
import tempfile
import httpx
import logging

from fastapi import FastAPI, HTTPException, Security, Header
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv

from store import (
    verify_api_key as _db_verify_api_key,
    _hash_key,
    log_usage,
    update_usage_status,
    _get_db,
    load_result as _load_result,
    create_job,
    get_job,
    get_job_by_user,
    check_user_concurrency,
)
from jobs import (
    enqueue_job,
    start_queue_workers,
    get_queue_depth,
    periodic_cleanup,
    GLOBAL_WORKER_CONCURRENCY,
)
from admin import router as admin_router
from helpers import (
    chunk_text,
    generate_srt_from_whisper,
    build_overlay_filtergraph,
    build_cover_scale_crop_filter,
    get_video_format_dimensions,
    generate_overlay_timed_segments,
    build_timed_segments_from_audio_chunks,
    build_timed_segments_from_audio_file,
    probe_video_dimensions,
    probe_audio_duration,
    generate_image_fal,
    generate_image_together,
    build_slide_segment,
    generate_text_thumbnail,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="ConnectCircuits API",
    description="ConnectCircuits multi-purpose media generation API",
    version="2.0.0",
)
app.include_router(admin_router)

KOKORO_BASE_URL  = os.getenv("KOKORO_BASE_URL", "http://kokoro:8880")
FAL_API_KEY      = os.getenv("FAL_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
FLUX_PROVIDER    = os.getenv("FLUX_PROVIDER", "fal").lower().strip()
PUBLIC_BASE_URL  = os.getenv("PUBLIC_BASE_URL", "https://api.connectcircuits.com")

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

COLOR_MAP = {
    "white":  "&H00FFFFFF",
    "yellow": "&H0000FFFF",
    "cyan":   "&H00FFFF00",
    "green":  "&H0000FF00",
    "red":    "&H000000FF",
    "black":  "&H00000000",
}


# -------------------------------------------------------
# Startup
# -------------------------------------------------------

@app.on_event("startup")
async def startup():
    conn = _get_db()
    conn.close()
    start_queue_workers()
    asyncio.create_task(periodic_cleanup(interval_sec=3600))
    logger.info(f"API ready. {GLOBAL_WORKER_CONCURRENCY} queue workers started.")


# -------------------------------------------------------
# Auth — per-user key validation
# -------------------------------------------------------

def verify_api_key(raw_key: str = Security(api_key_header)):
    if not raw_key:
        raise HTTPException(status_code=401, detail="Missing x-api-key header.")
    row = _db_verify_api_key(raw_key)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key.")
    return raw_key


# -------------------------------------------------------
# Shared helpers
# -------------------------------------------------------

def _build_subtitle_vf(srt_path, font_color, outline_color, font_size, position):
    primary_color = COLOR_MAP.get(font_color.lower(), "&H00FFFFFF")
    out_color     = COLOR_MAP.get(outline_color.lower(), "&H00000000")
    alignment     = 2 if position == "bottom" else 8
    force_style   = (
        f"FontName=Liberation Sans,FontSize={font_size},"
        f"PrimaryColour={primary_color},OutlineColour={out_color},"
        f"Outline=2,Shadow=1,Alignment={alignment},MarginV=20"
    )
    safe_srt = srt_path.replace("\\", "/").replace(":", "\\:")
    return f"subtitles='{safe_srt}':force_style='{force_style}'"


async def _resolve_timed_segments(caption_text, chunk_paths, audio_path, video_path, words_per_caption, loop):
    if caption_text and chunk_paths:
        return build_timed_segments_from_audio_chunks(caption_text, chunk_paths, words_per_caption)
    elif caption_text and audio_path and os.path.exists(audio_path):
        return build_timed_segments_from_audio_file(caption_text, audio_path, words_per_caption)
    else:
        return await loop.run_in_executor(
            None, generate_overlay_timed_segments, video_path, caption_text, words_per_caption,
        )


def _timing_method_label(chunk_paths, audio_path, do_captions, caption_text):
    if not do_captions:
        return "none"
    if chunk_paths:
        return "chunk-audio"
    if caption_text and audio_path and os.path.exists(audio_path):
        return "merged-audio"
    return "whisper"


def _job_response(job_id: str, webhook_url: Optional[str]) -> dict:
    return {
        "job_id":      job_id,
        "status":      "queued",
        "webhook_url": webhook_url or None,
        "status_url":  f"{PUBLIC_BASE_URL}/v1/jobs/{job_id}",
        "result_url":  f"{PUBLIC_BASE_URL}/v1/jobs/{job_id}/result",
        "note": (
            "Webhook will be called when complete."
            if webhook_url
            else "Poll status_url for updates, then fetch result_url when complete."
        ),
    }


async def _guard_concurrency(raw_key: str, endpoint: str):
    key_hash = _hash_key(raw_key)
    if not check_user_concurrency(key_hash):
        raise HTTPException(
            status_code=429,
            detail="Concurrency limit reached. Wait for existing jobs to complete before submitting more.",
            headers={"Retry-After": "30"},
        )
    return key_hash


# -------------------------------------------------------
# Job status + result endpoints
# -------------------------------------------------------

@app.get("/v1/jobs/{job_id}")
async def job_status(job_id: str, raw_key: str = Security(verify_api_key)):
    key_hash = _hash_key(raw_key)
    job = get_job_by_user(job_id, key_hash)
    if not job:
        # Check if job exists at all (wrong user) vs not found
        any_job = get_job(job_id)
        if any_job:
            raise HTTPException(status_code=403, detail="Access denied.")
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    return {
        "job_id":       job["job_id"],
        "endpoint":     job["endpoint"],
        "status":         job["status"],
        "queue_position": job.get("queue_position"),
        "queue_depth":    get_queue_depth(),
        "created_at":   job.get("created_at"),
        "started_at":   job.get("started_at") or None,
        "completed_at": job.get("completed_at") or None,
        "result_url":   job.get("result_url") or None,
        "error":        job.get("error") or None,
    }


@app.get("/v1/jobs/{job_id}/result")
async def job_result(job_id: str, raw_key: str = Security(verify_api_key)):
    key_hash = _hash_key(raw_key)
    job = get_job_by_user(job_id, key_hash)
    if not job:
        any_job = get_job(job_id)
        if any_job:
            raise HTTPException(status_code=403, detail="Access denied.")
        raise HTTPException(status_code=404, detail="Job not found or expired.")
    if job["status"] != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Job is not complete yet (status: {job['status']}).",
        )
    result = _load_result(job_id)
    if not result:
        raise HTTPException(status_code=410, detail="Result has expired or been cleaned up.")
    data, content_type, filename = result
    extra_headers = job.get("result_headers") or {}
    return StreamingResponse(
        content=iter([data]),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}", **extra_headers},
    )


# -------------------------------------------------------
# Models
# -------------------------------------------------------

class AsyncBase(BaseModel):
    webhook_url: Optional[str] = Field(
        None,
        description="Optional callback URL. POST with JSON job result on completion. "
                    "If omitted, poll GET /v1/jobs/{job_id} for status.",
    )


class AudioGenerateRequest(AsyncBase):
    text: str
    voice: Optional[str] = "af_bella"
    speed: Optional[float] = 1.0
    response_format: Optional[str] = "mp3"


class VideoGenerateRequest(AsyncBase):
    video_url: str
    audio_url: str
    loop_video: Optional[bool] = True
    caption_text: Optional[str] = None
    audio_chunk_urls: Optional[List[str]] = None
    caption_style: Optional[str] = "overlay"
    font_size: Optional[int] = 52
    font_color: Optional[str] = "white"
    outline_color: Optional[str] = "black"
    position: Optional[str] = "bottom"
    words_per_line: Optional[int] = 5
    overlay_bar_color: Optional[str] = "yellow"
    words_per_caption: Optional[int] = 12
    video_format: Optional[str] = "full"


class VideoCaptionRequest(AsyncBase):
    video_url: str
    caption_text: Optional[str] = None
    audio_url: Optional[str] = None
    audio_chunk_urls: Optional[List[str]] = None
    font_size: Optional[int] = 18
    font_color: Optional[str] = "white"
    outline_color: Optional[str] = "black"
    position: Optional[str] = "bottom"
    words_per_line: Optional[int] = 5
    style: Optional[str] = "subtitle"
    overlay_bar_color: Optional[str] = "yellow"
    words_per_caption: Optional[int] = 12


class ImageGenerateRequest(AsyncBase):
    prompt: str
    width:  int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    steps:  Optional[int] = Field(4, ge=1, le=8)
    seed:   Optional[int] = None
    provider: Optional[str] = None



class TextThumbnailRequest(AsyncBase):
    top_text: str = Field(..., description="Main heading — large bold text on the left panel.")
    bottom_text: Optional[str] = Field(None, description="Bottom banner text. Omit for no banner text.")
    avatar_url: Optional[str] = Field(None, description="URL of avatar/character image for the right panel.")
    width: int = Field(1280, ge=256, le=3840)
    height: int = Field(720, ge=144, le=2160)
    bg_color: Optional[str] = Field("#000000", description="Left panel background hex color.")
    top_text_color: Optional[str] = Field("#ffffff", description="Top text hex color.")
    bottom_bg_color: Optional[str] = Field("#FFD700", description="Bottom banner background hex color.")
    bottom_text_color: Optional[str] = Field("#000000", description="Bottom banner text hex color.")
    top_font_size: Optional[int] = Field(95, ge=12, le=300)
    bottom_font_size: Optional[int] = Field(44, ge=12, le=200)
    font_path: Optional[str] = Field(None, description="Absolute path to a .ttf font inside the container.")


class SlideItem(BaseModel):
    image_url: str
    text: str


class SlideshowRequest(AsyncBase):
    slides: List[SlideItem]
    voice: Optional[str] = "af_bella"
    speed: Optional[float] = 1.0
    out_w: Optional[int] = 720
    out_h: Optional[int] = 1280
    caption_position: Optional[str] = None
    font_size: Optional[int] = 52
    font_color: Optional[str] = "white"
    words_per_caption: Optional[int] = 1
    pad_start: Optional[float] = 0.3
    pad_end: Optional[float] = 0.5


# -------------------------------------------------------
# Health
# -------------------------------------------------------

@app.get("/health")
async def health():
    try:
        conn = _get_db()
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False
    return {
        "status":              "ok" if db_ok else "degraded",
        "database":            "connected" if db_ok else "error",
        "tts_backend":         KOKORO_BASE_URL,
        "flux_provider":       FLUX_PROVIDER,
        "fal_configured":      bool(FAL_API_KEY),
        "together_configured": bool(TOGETHER_API_KEY),
        "public_base_url":     PUBLIC_BASE_URL,
    }


# -------------------------------------------------------
# POST /v1/generate/audio
# -------------------------------------------------------

@app.post("/v1/generate/audio", status_code=202)
async def generate_audio(request: AudioGenerateRequest, raw_key: str = Security(verify_api_key)):
    key_hash = await _guard_concurrency(raw_key, "/v1/generate/audio")

    async def _task():
        mime_map  = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/opus", "flac": "audio/flac"}
        mime_type = mime_map.get(request.response_format, "audio/mpeg")
        chunks    = chunk_text(request.text)
        if not chunks:
            raise ValueError("No text provided.")

        tmp_dir     = tempfile.mkdtemp()
        chunk_paths = []
        fmt         = request.response_format or "mp3"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                for i, chunk in enumerate(chunks):
                    resp = await client.post(
                        f"{KOKORO_BASE_URL}/v1/audio/speech",
                        json={"model": "kokoro", "input": chunk,
                              "voice": request.voice, "speed": request.speed,
                              "response_format": fmt},
                        headers={"Content-Type": "application/json"},
                    )
                    resp.raise_for_status()
                    cp = os.path.join(tmp_dir, f"chunk_{i:04d}.{fmt}")
                    with open(cp, "wb") as f:
                        f.write(resp.content)
                    chunk_paths.append(cp)

            if len(chunk_paths) == 1:
                with open(chunk_paths[0], "rb") as f:
                    return f.read(), mime_type, {"X-Voice": request.voice, "X-Chunks": "1"}

            concat_list = os.path.join(tmp_dir, "concat.txt")
            output_path = os.path.join(tmp_dir, f"output.{fmt}")
            with open(concat_list, "w") as f:
                for p in chunk_paths:
                    f.write(f"file '{p}'\n")
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                "-i", concat_list, "-c", "copy", output_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg concat error: {stderr.decode()}")
            with open(output_path, "rb") as f:
                return f.read(), mime_type, {"X-Voice": request.voice, "X-Chunks": str(len(chunks))}
        finally:
            for p in chunk_paths:
                if os.path.exists(p):
                    os.remove(p)
            for n in ["concat.txt", f"output.{fmt}"]:
                pp = os.path.join(tmp_dir, n)
                if os.path.exists(pp):
                    os.remove(pp)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    job_id = create_job("/v1/generate/audio", key_hash, request.webhook_url, request.dict())
    asyncio.create_task(enqueue_job(
        job_id=job_id, user_key_hash=key_hash, raw_key=raw_key,
        endpoint="/v1/generate/audio", task_fn=_task,
        result_filename=f"audio.{request.response_format or 'mp3'}",
        public_base_url=PUBLIC_BASE_URL,
    ))
    return _job_response(job_id, request.webhook_url)


@app.get("/v1/generate/audio/voices")
async def list_voices(raw_key: str = Security(verify_api_key)):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{KOKORO_BASE_URL}/v1/audio/voices")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


# -------------------------------------------------------
# POST /v1/generate/image
# -------------------------------------------------------

@app.post("/v1/generate/image", status_code=202)
async def generate_image(request: ImageGenerateRequest, raw_key: str = Security(verify_api_key)):
    key_hash = await _guard_concurrency(raw_key, "/v1/generate/image")
    if request.width % 16 != 0 or request.height % 16 != 0:
        raise HTTPException(status_code=400, detail="width and height must be multiples of 16.")

    async def _task():
        provider  = (request.provider or FLUX_PROVIDER).lower().strip()
        providers = {
            "fal":          ["fal"],
            "together":     ["together"],
            "fal+together": ["fal", "together"],
            "together+fal": ["together", "fal"],
        }.get(provider)
        if not providers:
            raise ValueError(f"Unknown provider '{provider}'.")

        last_error = None
        for p in providers:
            try:
                if p == "fal":
                    img_bytes, ctype = await generate_image_fal(
                        api_key=FAL_API_KEY, prompt=request.prompt,
                        width=request.width, height=request.height,
                        steps=request.steps or 4, seed=request.seed,
                    )
                    return img_bytes, ctype, {"X-Provider": "fal",
                                              "X-Width": str(request.width),
                                              "X-Height": str(request.height)}
                elif p == "together":
                    img_bytes, ctype = await generate_image_together(
                        api_key=TOGETHER_API_KEY, prompt=request.prompt,
                        width=request.width, height=request.height,
                        steps=request.steps or 4, seed=request.seed,
                    )
                    return img_bytes, ctype, {"X-Provider": "together",
                                              "X-Width": str(request.width),
                                              "X-Height": str(request.height)}
            except Exception as e:
                last_error = f"{p}: {e}"
        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    job_id = create_job("/v1/generate/image", key_hash, request.webhook_url, request.dict())
    asyncio.create_task(enqueue_job(
        job_id=job_id, user_key_hash=key_hash, raw_key=raw_key,
        endpoint="/v1/generate/image", task_fn=_task,
        result_filename="image.png",
        public_base_url=PUBLIC_BASE_URL,
    ))
    return _job_response(job_id, request.webhook_url)



# -------------------------------------------------------
# POST /v1/generate/text-thumbnail
# -------------------------------------------------------

@app.post("/v1/generate/text-thumbnail", status_code=202)
async def generate_text_thumbnail_endpoint(
    request: TextThumbnailRequest,
    raw_key: str = Security(verify_api_key),
):
    key_hash = await _guard_concurrency(raw_key, "/v1/generate/text-thumbnail")

    async def _task():
        img_bytes, ctype = await generate_text_thumbnail(
            top_text          = request.top_text,
            bottom_text       = request.bottom_text,
            avatar_url        = request.avatar_url,
            width             = request.width,
            height            = request.height,
            bg_color          = request.bg_color          or "#000000",
            top_text_color    = request.top_text_color    or "#ffffff",
            bottom_bg_color   = request.bottom_bg_color   or "#FFD700",
            bottom_text_color = request.bottom_text_color or "#000000",
            top_font_size     = request.top_font_size     or 95,
            bottom_font_size  = request.bottom_font_size  or 44,
            font_path         = request.font_path,
        )
        return img_bytes, ctype, {
            "X-Width":  str(request.width),
            "X-Height": str(request.height),
        }


    job_id = create_job("/v1/generate/text-thumbnail", key_hash, request.webhook_url, request.dict())
    asyncio.create_task(enqueue_job(
        job_id=job_id, user_key_hash=key_hash, raw_key=raw_key,
        endpoint="/v1/generate/text-thumbnail", task_fn=_task,
        result_filename="thumbnail.png",
        public_base_url=PUBLIC_BASE_URL,
    ))
    return _job_response(job_id, request.webhook_url)


# -------------------------------------------------------
# POST /v1/generate/slideshow
# -------------------------------------------------------

@app.post("/v1/generate/slideshow", status_code=202)
async def generate_slideshow(request: SlideshowRequest, raw_key: str = Security(verify_api_key)):
    if not request.slides:
        raise HTTPException(status_code=400, detail="slides array is empty.")
    out_w = request.out_w or 720
    out_h = request.out_h or 1280
    if out_w < 256 or out_w > 3840 or out_h < 256 or out_h > 3840:
        raise HTTPException(status_code=400, detail="out_w and out_h must be between 256 and 3840.")
    if out_w % 2 != 0 or out_h % 2 != 0:
        raise HTTPException(status_code=400, detail="out_w and out_h must be divisible by 2.")
    cap_pos = None
    if request.caption_position:
        cap_pos = request.caption_position.lower().strip()
        if cap_pos not in ("top", "center", "bottom"):
            raise HTTPException(status_code=400, detail="caption_position must be 'top', 'center', or 'bottom'.")

    key_hash = await _guard_concurrency(raw_key, "/v1/generate/slideshow")

    async def _task():
        tmp_dir       = tempfile.mkdtemp()
        segment_paths = []
        all_tmp_files = []
        try:
            tasks = [
                build_slide_segment(
                    slide=slide, index=i, tmp_dir=tmp_dir,
                    kokoro_base_url=KOKORO_BASE_URL,
                    voice=request.voice or "af_bella",
                    speed=request.speed or 1.0,
                    font_size=request.font_size or 52,
                    font_color=request.font_color or "white",
                    caption_position=cap_pos,
                    words_per_caption=request.words_per_caption or 1,
                    pad_start=request.pad_start if request.pad_start is not None else 0.3,
                    pad_end=request.pad_end if request.pad_end is not None else 0.5,
                    out_w=out_w, out_h=out_h,
                )
                for i, slide in enumerate(request.slides)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    raise RuntimeError(f"Slide {i+1} failed: {result}")
                seg_path, tmp_files = result
                segment_paths.append(seg_path)
                all_tmp_files.extend(tmp_files)

            concat_list = os.path.join(tmp_dir, "slideshow_concat.txt")
            output_path = os.path.join(tmp_dir, f"slideshow_{uuid.uuid4().hex}.mp4")
            all_tmp_files.extend([concat_list, output_path])
            with open(concat_list, "w") as f:
                for p in segment_paths:
                    f.write(f"file '{p}'\n")

            proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
                "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                output_path,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg concat error: {stderr.decode()}")

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            return video_bytes, "video/mp4", {
                "X-Slide-Count":      str(len(request.slides)),
                "X-Resolution":       f"{out_w}x{out_h}",
                "X-Caption-Position": cap_pos or "none",
                "X-Voice":            request.voice or "af_bella",
            }
        finally:
            for p in all_tmp_files + segment_paths:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    job_id = create_job("/v1/generate/slideshow", key_hash, request.webhook_url, {"slide_count": len(request.slides)})
    asyncio.create_task(enqueue_job(
        job_id=job_id, user_key_hash=key_hash, raw_key=raw_key,
        endpoint="/v1/generate/slideshow", task_fn=_task,
        result_filename="slideshow.mp4",
        public_base_url=PUBLIC_BASE_URL,
    ))
    return _job_response(job_id, request.webhook_url)


# -------------------------------------------------------
# POST /v1/generate/video
# -------------------------------------------------------

@app.post("/v1/generate/video", status_code=202)
async def generate_video(request: VideoGenerateRequest, raw_key: str = Security(verify_api_key)):
    key_hash      = await _guard_concurrency(raw_key, "/v1/generate/video")
    do_captions   = bool(request.caption_text or request.audio_chunk_urls)
    caption_style = (request.caption_style or "overlay").lower().strip()
    if caption_style not in ("overlay", "subtitle"):
        caption_style = "overlay"

    async def _task():
        tmp_dir     = tempfile.mkdtemp()
        video_path  = os.path.join(tmp_dir, f"video_{uuid.uuid4().hex}.mp4")
        audio_path  = os.path.join(tmp_dir, f"audio_{uuid.uuid4().hex}.mp3")
        output_path = os.path.join(tmp_dir, f"output_{uuid.uuid4().hex}.mp4")
        srt_path    = os.path.join(tmp_dir, "captions.srt")
        fg_path     = os.path.join(tmp_dir, "overlay.filtergraph")
        chunk_paths = []

        try:
            urls = [request.video_url, request.audio_url]
            if request.audio_chunk_urls:
                urls.extend(request.audio_chunk_urls)
            async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
                responses = await asyncio.gather(*[client.get(u) for u in urls])
            for r in responses:
                r.raise_for_status()
            with open(video_path, "wb") as f:
                f.write(responses[0].content)
            with open(audio_path, "wb") as f:
                f.write(responses[1].content)
            if request.audio_chunk_urls:
                for i, resp in enumerate(responses[2:]):
                    cp = os.path.join(tmp_dir, f"chunk_{i:04d}.mp3")
                    with open(cp, "wb") as f:
                        f.write(resp.content)
                    chunk_paths.append(cp)

            loop       = asyncio.get_event_loop()
            loop_flags = ["-stream_loop", "-1"] if request.loop_video else []
            target_w, target_h, resolved_format = get_video_format_dimensions(request.video_format)
            base_vf = build_cover_scale_crop_filter(target_w, target_h)

            if not do_captions:
                cmd = [
                    "ffmpeg", "-y", *loop_flags, "-i", video_path, "-i", audio_path,
                    "-vf", base_vf,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", output_path,
                ]
            elif caption_style == "overlay":
                overlay_font_size = request.font_size or (64 if resolved_format == "shorts" else 52)
                timed_segments = await _resolve_timed_segments(
                    request.caption_text, chunk_paths, audio_path,
                    video_path, request.words_per_caption or 12, loop,
                )
                if not timed_segments:
                    raise RuntimeError("No caption segments could be generated.")
                overlay_fg = build_overlay_filtergraph(
                    timed_segments, overlay_font_size,
                    request.overlay_bar_color or "yellow",
                    request.font_color or "white", target_w, target_h,
                )
                filtergraph = f"{base_vf},{overlay_fg}"
                with open(fg_path, "w", encoding="utf-8") as f:
                    f.write(filtergraph)
                cmd = [
                    "ffmpeg", "-y", *loop_flags, "-i", video_path, "-i", audio_path,
                    "-filter_script:v", fg_path,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", output_path,
                ]
            else:
                srt_content = await loop.run_in_executor(
                    None, generate_srt_from_whisper,
                    video_path, request.caption_text, request.words_per_line or 5,
                )
                if not srt_content.strip():
                    raise RuntimeError("No speech detected in video.")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                subtitle_font_size = request.font_size or (24 if resolved_format == "shorts" else 18)
                subtitle_vf = _build_subtitle_vf(
                    srt_path, request.font_color or "white",
                    request.outline_color or "black",
                    subtitle_font_size, request.position or "bottom",
                )
                vf = f"{base_vf},{subtitle_vf}"
                cmd = [
                    "ffmpeg", "-y", *loop_flags, "-i", video_path, "-i", audio_path,
                    "-vf", vf,
                    "-map", "0:v", "-map", "1:a",
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", output_path,
                ]

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {stderr.decode()}")

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            return video_bytes, "video/mp4", {
                "X-Caption-Style": caption_style if do_captions else "none",
                "X-Timing-Method": _timing_method_label(chunk_paths, audio_path, do_captions, request.caption_text),
                "X-Video-Format": resolved_format,
                "X-Width": str(target_w),
                "X-Height": str(target_h),
            }
        finally:
            for p in [video_path, audio_path, output_path, srt_path, fg_path] + chunk_paths:
                if os.path.exists(p):
                    os.remove(p)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    job_id = create_job("/v1/generate/video", key_hash, request.webhook_url, {"video_url": request.video_url})
    asyncio.create_task(enqueue_job(
        job_id=job_id, user_key_hash=key_hash, raw_key=raw_key,
        endpoint="/v1/generate/video", task_fn=_task,
        result_filename="output.mp4",
        public_base_url=PUBLIC_BASE_URL,
    ))
    return _job_response(job_id, request.webhook_url)


# -------------------------------------------------------
# POST /v1/generate/video/captions
# -------------------------------------------------------

@app.post("/v1/generate/video/captions", status_code=202)
async def generate_video_captions(request: VideoCaptionRequest, raw_key: str = Security(verify_api_key)):
    key_hash = await _guard_concurrency(raw_key, "/v1/generate/video/captions")
    style    = (request.style or "subtitle").lower().strip()
    if style not in ("overlay", "subtitle"):
        style = "subtitle"

    async def _task():
        tmp_dir     = tempfile.mkdtemp()
        video_path  = os.path.join(tmp_dir, f"video_{uuid.uuid4().hex}.mp4")
        output_path = os.path.join(tmp_dir, f"output_{uuid.uuid4().hex}.mp4")
        srt_path    = os.path.join(tmp_dir, "captions.srt")
        fg_path     = os.path.join(tmp_dir, "overlay.filtergraph")
        audio_path  = os.path.join(tmp_dir, f"audio_{uuid.uuid4().hex}.mp3")
        chunk_paths = []

        try:
            urls = [request.video_url]
            if request.audio_url:
                urls.append(request.audio_url)
            if request.audio_chunk_urls:
                urls.extend(request.audio_chunk_urls)
            async with httpx.AsyncClient(timeout=180.0, follow_redirects=True) as client:
                responses = await asyncio.gather(*[client.get(u) for u in urls])
            for r in responses:
                r.raise_for_status()
            with open(video_path, "wb") as f:
                f.write(responses[0].content)
            idx = 1
            if request.audio_url:
                with open(audio_path, "wb") as f:
                    f.write(responses[idx].content)
                idx += 1
            if request.audio_chunk_urls:
                for i, resp in enumerate(responses[idx:]):
                    cp = os.path.join(tmp_dir, f"chunk_{i:04d}.mp3")
                    with open(cp, "wb") as f:
                        f.write(resp.content)
                    chunk_paths.append(cp)

            loop = asyncio.get_event_loop()
            words_per_caption = request.words_per_caption or 12

            if style == "overlay":
                video_width, video_height = probe_video_dimensions(video_path)
                timed_segments = await _resolve_timed_segments(
                    request.caption_text, chunk_paths,
                    audio_path if request.audio_url else None,
                    video_path, words_per_caption, loop,
                )
                if not timed_segments:
                    raise RuntimeError("No caption segments could be generated.")
                filtergraph = build_overlay_filtergraph(
                    timed_segments,
                    request.font_size if (request.font_size and request.font_size != 18) else 52,
                    request.overlay_bar_color or "yellow",
                    request.font_color or "white", video_width, video_height,
                )
                with open(fg_path, "w", encoding="utf-8") as f:
                    f.write(filtergraph)
                cmd = [
                    "ffmpeg", "-y", "-i", video_path,
                    "-filter_script:v", fg_path,
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-c:a", "copy", output_path,
                ]
            else:
                srt_content = await loop.run_in_executor(
                    None, generate_srt_from_whisper,
                    video_path, request.caption_text, request.words_per_line or 5,
                )
                if not srt_content.strip():
                    raise RuntimeError("No speech detected in video.")
                with open(srt_path, "w", encoding="utf-8") as f:
                    f.write(srt_content)
                vf = _build_subtitle_vf(
                    srt_path, request.font_color or "white",
                    request.outline_color or "black",
                    request.font_size or 18, request.position or "bottom",
                )
                cmd = [
                    "ffmpeg", "-y", "-i", video_path, "-vf", vf,
                    "-c:v", "libx264", "-crf", "23", "-preset", "fast",
                    "-c:a", "copy", output_path,
                ]

            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg error: {stderr.decode()}")

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            return video_bytes, "video/mp4", {
                "X-Caption-Style": style,
                "X-Timing-Method": _timing_method_label(chunk_paths, audio_path, True, request.caption_text),
            }
        finally:
            for p in [video_path, output_path, srt_path, fg_path, audio_path] + chunk_paths:
                if os.path.exists(p):
                    os.remove(p)
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass

    job_id = create_job("/v1/generate/video/captions", key_hash, request.webhook_url, {"video_url": request.video_url})
    asyncio.create_task(enqueue_job(
        job_id=job_id, user_key_hash=key_hash, raw_key=raw_key,
        endpoint="/v1/generate/video/captions", task_fn=_task,
        result_filename="captioned.mp4",
        public_base_url=PUBLIC_BASE_URL,
    ))
    return _job_response(job_id, request.webhook_url)
