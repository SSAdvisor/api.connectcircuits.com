# ConnectCircuits API

A self-hosted media generation API powering audio synthesis, image generation, video overlay, and slideshow creation — built on FastAPI, Kokoro TTS, and FLUX image models. All endpoints are **async-first**: every job returns a `job_id` immediately and results are fetched after completion, or delivered via webhook.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [Async Job Pattern](#async-job-pattern)
- [Endpoints](#endpoints)
  - [GET /health](#get-health)
  - [POST /v1/generate/audio](#post-v1generateaudio)
  - [GET /v1/generate/audio/voices](#get-v1generateaudiovoices)
  - [POST /v1/generate/image](#post-v1generateimage)
  - [POST /v1/generate/text-thumbnail](#post-v1generatetext-thumbnail)
  - [POST /v1/generate/slideshow](#post-v1generateslideshow)
  - [POST /v1/generate/video](#post-v1generatevideo)
  - [POST /v1/generate/video/captions](#post-v1generatevideocaptions)
  - [GET /v1/jobs/{job_id}](#get-v1jobsjob_id)
  - [GET /v1/jobs/{job_id}/result](#get-v1jobsjob_idresult)
- [Admin Endpoints](#admin-endpoints)
- [Error Codes](#error-codes)
- [Rate Limits & Concurrency](#rate-limits--concurrency)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)

---

## Quick Start

```bash
# 1. Submit a job
curl -X POST https://api.connectcircuits.com/v1/generate/audio \
  -H "x-api-key: cc-your-key-here" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello from ConnectCircuits!", "voice": "af_bella"}'

# → {"job_id": "abc-123", "status": "queued", "status_url": "...", "result_url": "..."}

# 2. Poll until complete
curl -H "x-api-key: cc-your-key-here" \
  https://api.connectcircuits.com/v1/jobs/abc-123

# → {"status": "complete", "result_url": "..."}

# 3. Download result
curl -H "x-api-key: cc-your-key-here" \
  https://api.connectcircuits.com/v1/jobs/abc-123/result \
  --output audio.mp3
```

---

## Authentication

All endpoints (except `/health`) require an API key passed as a request header:

```
x-api-key: cc-your-key-here
```

Keys are issued by the admin panel at `https://api.connectcircuits.com/admin/ui`. Keys are hashed at rest and never stored in plaintext. A revoked key returns `401`.

---

## Async Job Pattern

Every generation endpoint returns `HTTP 202 Accepted` immediately with a job payload:

```json
{
  "job_id": "3f4a1b2c-...",
  "status": "queued",
  "status_url": "https://api.connectcircuits.com/v1/jobs/3f4a1b2c-...",
  "result_url": "https://api.connectcircuits.com/v1/jobs/3f4a1b2c-.../result",
  "webhook_url": null,
  "note": "Poll status_url for updates, then fetch result_url when complete."
}
```

### Polling Flow

```
POST /v1/generate/*  →  202 + job_id
    ↓
GET /v1/jobs/{job_id}  →  { status: "queued" | "started" | "complete" | "failed" }
    ↓ (when complete)
GET /v1/jobs/{job_id}/result  →  binary file download
```

### Webhook Flow

Include `"webhook_url"` in any request body. When the job finishes, the API POSTs to your URL:

```json
{
  "job_id": "3f4a1b2c-...",
  "status": "complete",
  "endpoint": "/v1/generate/audio",
  "result_url": "https://api.connectcircuits.com/v1/jobs/3f4a1b2c-.../result",
  "completed_at": 1716041234,
  "headers": { "X-Voice": "af_bella", "X-Chunks": "1" }
}
```

Failed jobs deliver:
```json
{
  "job_id": "3f4a1b2c-...",
  "status": "failed",
  "endpoint": "/v1/generate/audio",
  "error": "No text provided."
}
```

---

## Endpoints

### GET /health

Returns API health status. No authentication required.

**Response**
```json
{
  "status": "ok",
  "database": "connected",
  "tts_backend": "http://kokoro:8880",
  "flux_provider": "fal",
  "fal_configured": true,
  "together_configured": false,
  "public_base_url": "https://api.connectcircuits.com"
}
```

---

### POST /v1/generate/audio

Synthesizes speech from text using Kokoro TTS. Long texts are automatically chunked and concatenated.

**Request Body**

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | string | **required** | Text to synthesize. No hard length limit; chunked automatically. |
| `voice` | string | `"af_bella"` | Voice ID. See `GET /v1/generate/audio/voices` for available voices. |
| `speed` | float | `1.0` | Playback speed multiplier. Range: `0.5` – `2.0`. |
| `response_format` | string | `"mp3"` | Output format. One of: `mp3`, `wav`, `opus`, `flac`. |
| `webhook_url` | string | `null` | Optional. POST target for job completion notification. |

**Example**

```json
{
  "text": "The quick brown fox jumped over the lazy dog.",
  "voice": "af_bella",
  "speed": 1.0,
  "response_format": "mp3"
}
```

**Result** — `audio/mpeg` binary (or format specified)

**Response Headers**

| Header | Description |
|---|---|
| `X-Voice` | Voice used |
| `X-Chunks` | Number of text chunks processed |

---

### GET /v1/generate/audio/voices

Returns the list of available TTS voices from the Kokoro backend.

**Response** — JSON array of voice objects (passthrough from Kokoro).

---

### POST /v1/generate/image

Generates an image from a text prompt using FLUX (via fal.ai or Together AI).

**Request Body**

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | string | **required** | Image generation prompt. |
| `width` | int | `1024` | Output width in pixels. Must be a multiple of 16. Range: `256`–`2048`. |
| `height` | int | `1024` | Output height in pixels. Must be a multiple of 16. Range: `256`–`2048`. |
| `steps` | int | `4` | Inference steps. Range: `1`–`8`. |
| `seed` | int | `null` | Optional seed for reproducibility. |
| `provider` | string | *(env default)* | Override provider: `fal`, `together`, `fal+together`, `together+fal`. |
| `webhook_url` | string | `null` | Optional. |

**Example**

```json
{
  "prompt": "A photorealistic mountain landscape at sunset",
  "width": 1024,
  "height": 576,
  "steps": 4
}
```

**Result** — `image/png` or `image/jpeg` binary

---

### POST /v1/generate/text-thumbnail

Generates a styled two-panel YouTube-style thumbnail image with heading text and optional avatar.

**Request Body**

| Field | Type | Default | Description |
|---|---|---|---|
| `top_text` | string | **required** | Main heading on the left panel. |
| `bottom_text` | string | `null` | Bottom banner text. Omit for no banner. |
| `avatar_url` | string | `null` | URL of avatar/character image for the right panel. |
| `width` | int | `1280` | Output width. Range: `256`–`3840`. |
| `height` | int | `720` | Output height. Range: `144`–`2160`. |
| `bg_color` | string | `"#000000"` | Left panel background hex color. |
| `top_text_color` | string | `"#ffffff"` | Heading text hex color. |
| `bottom_bg_color` | string | `"#FFD700"` | Bottom banner background hex color. |
| `bottom_text_color` | string | `"#000000"` | Bottom banner text hex color. |
| `top_font_size` | int | `95` | Heading font size. Range: `12`–`300`. |
| `bottom_font_size` | int | `44` | Banner font size. Range: `12`–`200`. |
| `font_path` | string | `null` | Absolute path to a `.ttf` font inside the container. |
| `webhook_url` | string | `null` | Optional. |

**Result** — `image/png` binary

---

### POST /v1/generate/slideshow

Generates a narrated slideshow video from an array of slides, each with an image and text. Audio is synthesized via Kokoro TTS and captions are overlaid on each slide.

**Request Body**

| Field | Type | Default | Description |
|---|---|---|---|
| `slides` | array | **required** | Array of slide objects (see below). Max recommended: 50. |
| `voice` | string | `"af_bella"` | Kokoro TTS voice for narration. |
| `speed` | float | `1.0` | TTS playback speed. |
| `out_w` | int | `720` | Output width in pixels. Must be even. Range: `256`–`3840`. |
| `out_h` | int | `1280` | Output height in pixels. Must be even. Range: `256`–`3840`. |
| `caption_position` | string | `null` | Caption vertical position: `top`, `center`, `bottom`. |
| `font_size` | int | `52` | Caption font size in pixels. |
| `font_color` | string | `"white"` | Caption font color: `white`, `yellow`, `cyan`, `green`, `red`, `black`. |
| `words_per_caption` | int | `1` | Words to display per caption segment. |
| `pad_start` | float | `0.3` | Seconds of silence before narration on each slide. |
| `pad_end` | float | `0.5` | Seconds of silence after narration on each slide. |
| `webhook_url` | string | `null` | Optional. |

**Slide Object**

| Field | Type | Description |
|---|---|---|
| `image_url` | string | Publicly accessible URL to the slide image. |
| `text` | string | Narration text for this slide (also used as caption). |

**Example**

```json
{
  "slides": [
    {
      "image_url": "https://example.com/slide1.jpg",
      "text": "Welcome to our story. Today we explore the unknown."
    },
    {
      "image_url": "https://example.com/slide2.jpg",
      "text": "In 2024, researchers made a breakthrough discovery."
    }
  ],
  "voice": "af_bella",
  "out_w": 720,
  "out_h": 1280,
  "caption_position": "bottom",
  "font_size": 52,
  "words_per_caption": 3
}
```

**Result** — `video/mp4` binary

**Response Headers**

| Header | Description |
|---|---|
| `X-Slide-Count` | Number of slides rendered |
| `X-Resolution` | Output resolution (e.g. `720x1280`) |
| `X-Caption-Position` | Caption position used |
| `X-Voice` | Voice used |

---

### POST /v1/generate/video

Combines a background video with an audio track and optional synchronized captions. Always outputs at a fixed resolution regardless of source video size.

**Request Body**

| Field | Type | Default | Description |
|---|---|---|---|
| `video_url` | string | **required** | URL to the background video. Any resolution accepted; scaled and cropped to target. |
| `audio_url` | string | **required** | URL to the audio track (MP3). |
| `video_format` | string | `"full"` | Output format: `full` (1280×720) or `shorts` / `vertical` / `portrait` (720×1280). |
| `loop_video` | bool | `true` | Loop the video if shorter than the audio. |
| `caption_text` | string | `null` | Text for captions. If omitted with no `audio_chunk_urls`, captions are auto-transcribed via Whisper. |
| `audio_chunk_urls` | array | `null` | Ordered array of audio chunk URLs for precise per-chunk caption timing. |
| `caption_style` | string | `"overlay"` | `overlay` (bar + text drawn on video) or `subtitle` (SRT-style burned-in captions). |
| `font_size` | int | `52` (`64` for shorts) | Caption font size. |
| `font_color` | string | `"white"` | Caption font color. |
| `outline_color` | string | `"black"` | Subtitle outline color (subtitle style only). |
| `overlay_bar_color` | string | `"yellow"` | Accent bar color for overlay style. |
| `position` | string | `"bottom"` | Subtitle position: `top` or `bottom` (subtitle style only). |
| `words_per_caption` | int | `12` | Words per caption segment. |
| `words_per_line` | int | `5` | Words per line (subtitle style only). |
| `webhook_url` | string | `null` | Optional. |

**Output Resolutions**

| `video_format` | Width | Height | Use case |
|---|---|---|---|
| `full` (default) | 1280 | 720 | YouTube, standard widescreen |
| `shorts` / `vertical` / `portrait` | 720 | 1280 | YouTube Shorts, TikTok, Reels |

**Example — Full with overlay captions**

```json
{
  "video_url": "https://example.com/background.mp4",
  "audio_url": "https://example.com/narration.mp3",
  "video_format": "full",
  "caption_text": "The school board voted eleven to one for improvements.",
  "caption_style": "overlay",
  "overlay_bar_color": "yellow",
  "font_color": "white",
  "words_per_caption": 8
}
```

**Example — Shorts with audio chunks**

```json
{
  "video_url": "https://example.com/background.mp4",
  "audio_url": "https://example.com/merged.mp3",
  "video_format": "shorts",
  "caption_text": "Breaking news from Ohio today.",
  "audio_chunk_urls": [
    "https://example.com/chunk_0001.mp3",
    "https://example.com/chunk_0002.mp3"
  ],
  "caption_style": "overlay",
  "words_per_caption": 5
}
```

**Result** — `video/mp4` binary (H.264, AAC 192k, yuv420p)

**Response Headers**

| Header | Description |
|---|---|
| `X-Caption-Style` | Caption style used |
| `X-Timing-Method` | How caption timing was derived: `chunk-audio`, `merged-audio`, `whisper`, or `none` |
| `X-Video-Format` | `full` or `shorts` |
| `X-Width` | Output width in pixels |
| `X-Height` | Output height in pixels |

---

### POST /v1/generate/video/captions

Adds captions to an existing video that already has audio. No separate audio track required; timing can be derived from Whisper transcription, a provided audio track, or pre-timed audio chunks.

**Request Body**

| Field | Type | Default | Description |
|---|---|---|---|
| `video_url` | string | **required** | URL to the video file (must already contain audio for Whisper mode). |
| `caption_text` | string | `null` | Override text. If omitted, Whisper transcribes the video's audio. |
| `audio_url` | string | `null` | Optional separate audio file for caption timing. |
| `audio_chunk_urls` | array | `null` | Ordered audio chunks for precise timing. |
| `style` | string | `"subtitle"` | `subtitle` (SRT burned-in) or `overlay` (bar + text). |
| `font_size` | int | `18` | Font size. |
| `font_color` | string | `"white"` | Font color. |
| `outline_color` | string | `"black"` | Subtitle outline color. |
| `position` | string | `"bottom"` | `top` or `bottom` (subtitle style only). |
| `overlay_bar_color` | string | `"yellow"` | Overlay accent bar color (overlay style only). |
| `words_per_caption` | int | `12` | Words per caption segment. |
| `words_per_line` | int | `5` | Words per line (subtitle style only). |
| `webhook_url` | string | `null` | Optional. |

**Result** — `video/mp4` binary

---

### GET /v1/jobs/{job_id}

Returns the current status of a job. Results are retained for 6 hours after completion.

**Response**

```json
{
  "job_id": "3f4a1b2c-...",
  "endpoint": "/v1/generate/audio",
  "status": "complete",
  "queue_position": null,
  "queue_depth": 0,
  "created_at": 1716041100,
  "started_at": 1716041102,
  "completed_at": 1716041108,
  "result_url": "https://api.connectcircuits.com/v1/jobs/3f4a1b2c-.../result",
  "error": null
}
```

**Job Statuses**

| Status | Meaning |
|---|---|
| `queued` | Accepted, waiting for a worker slot |
| `started` | Actively processing |
| `complete` | Finished — result available at `result_url` |
| `failed` | Error during processing — see `error` field |

---

### GET /v1/jobs/{job_id}/result

Downloads the binary result of a completed job. Returns `409` if the job is not yet complete, `410` if the result has expired.

**Response** — Binary file with appropriate `Content-Type` header and `Content-Disposition: attachment`.

---

## Admin Endpoints

Admin endpoints require the `x-admin-secret` header (set via `ADMIN_SECRET` env var). A browser-based admin panel is available at `https://api.connectcircuits.com/admin/ui`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/admin/ui` | Browser admin panel |
| `GET` | `/admin/keys` | List all API keys |
| `POST` | `/admin/keys` | Create a new API key |
| `DELETE` | `/admin/keys` | Revoke an API key |
| `GET` | `/admin/usage/summary` | Usage summary per user and endpoint |

**Create Key**

```bash
curl -X POST https://api.connectcircuits.com/admin/keys \
  -H "x-admin-secret: your-admin-secret" \
  -H "Content-Type: application/json" \
  -d '{"user_label": "customer_acme", "tier": "pro"}'
```

```json
{
  "raw_key": "cc-xxxxxxxxxxxxxxxx",
  "note": "Store this securely — it will not be shown again."
}
```

---

## Error Codes

| Code | Meaning |
|---|---|
| `400` | Bad request — invalid parameters |
| `401` | Missing or invalid `x-api-key` |
| `403` | Job belongs to a different API key |
| `404` | Job not found or expired |
| `409` | Job not complete yet |
| `410` | Job result has expired (>6 hours) |
| `422` | Request body validation error |
| `429` | Concurrency limit reached — wait for active jobs to finish |
| `500` | Internal server error |
| `503` | Upstream service unavailable (TTS backend, image provider) |

---

## Rate Limits & Concurrency

Each API key may have up to **3 concurrent active jobs** by default (configurable via `USER_CONCURRENCY_CAP`). The global worker pool processes up to **5 jobs simultaneously** across all users (configurable via `GLOBAL_WORKER_CONCURRENCY`).

When the per-user cap is reached, the API returns `429` with a `Retry-After: 30` header. Poll your existing jobs and retry once one completes.

---

## Docker Deployment

```yaml
# docker-compose.yml (relevant service)
services:
  tts-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FAL_API_KEY=${FAL_API_KEY}
      - TOGETHER_API_KEY=${TOGETHER_API_KEY}
      - ADMIN_SECRET=${ADMIN_SECRET}
      - PUBLIC_BASE_URL=https://api.connectcircuits.com
      - KOKORO_BASE_URL=http://kokoro:8880
      - FLUX_PROVIDER=fal
      - USER_CONCURRENCY_CAP=3
      - GLOBAL_WORKER_CONCURRENCY=5
      - RESULT_TTL_SEC=21600
    volumes:
      - api_data:/app/data
    depends_on:
      - kokoro
      - redis
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FAL_API_KEY` | — | fal.ai API key for FLUX image generation |
| `TOGETHER_API_KEY` | — | Together AI API key for FLUX image generation |
| `ADMIN_SECRET` | — | Secret for admin endpoint access |
| `PUBLIC_BASE_URL` | `https://api.connectcircuits.com` | Base URL returned in job response URLs |
| `KOKORO_BASE_URL` | `http://kokoro:8880` | Internal Kokoro TTS service URL |
| `FLUX_PROVIDER` | `fal` | Default image provider: `fal`, `together`, `fal+together` |
| `USER_CONCURRENCY_CAP` | `3` | Max simultaneous active jobs per API key |
| `GLOBAL_WORKER_CONCURRENCY` | `5` | Total worker slots across all users |
| `RESULT_TTL_SEC` | `21600` | Seconds to retain completed job results (default 6 hours) |
| `WEBHOOK_TIMEOUT_SEC` | `10` | Timeout per webhook delivery attempt |
| `WEBHOOK_RETRIES` | `3` | Max webhook delivery attempts with exponential backoff |
