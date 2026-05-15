from typing import Optional, List
import re
import asyncio
import subprocess
import base64
import os
import threading
import httpx
from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


# -------------------------------------------------------
# Text chunking
# -------------------------------------------------------

def chunk_text(text: str, max_chars: int = 400) -> list:
    chunks = []
    current_chunk = ""
    sentences = re.split(r'(?<=[.!?;])\s+', text.strip())
    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(sentence) > max_chars:
            parts = sentence.split(",")
            for part in parts:
                if len(current_chunk) + len(part) <= max_chars:
                    current_chunk += part + ","
                else:
                    if len(part) > max_chars:
                        words = part.split()
                        for word in words:
                            if len(current_chunk) + len(word) + 1 > max_chars:
                                if current_chunk.strip():
                                    chunks.append(current_chunk.strip())
                                current_chunk = word + " "
                            else:
                                current_chunk += word + " "
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = part + ","
        else:
            if len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += " " + sentence
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return [c for c in chunks if c]


# -------------------------------------------------------
# SRT helpers
# -------------------------------------------------------

def seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def group_words_into_lines(words: list, words_per_line: int = 5) -> list:
    lines = []
    for i in range(0, len(words), words_per_line):
        group = words[i:i + words_per_line]
        start = group[0]["start"]
        end = group[-1]["end"]
        text = " ".join(w["word"].strip() for w in group)
        lines.append({"start": start, "end": end, "text": text})
    return lines


def generate_srt_from_whisper(video_path: str, caption_text: str = None, words_per_line: int = 5) -> str:
    model = get_whisper_model()
    segments, _ = model.transcribe(
        video_path, beam_size=5, language="en",
        word_timestamps=True, vad_filter=True,
    )
    all_words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                all_words.append({"word": w.word, "start": w.start, "end": w.end})

    if not all_words:
        return ""

    if not caption_text:
        lines = group_words_into_lines(all_words, words_per_line)
    else:
        caption_words = caption_text.split()
        n_caption = len(caption_words)
        n_whisper = len(all_words)
        timed = []
        for i, cap_word in enumerate(caption_words):
            idx = min(int(i * n_whisper / n_caption), n_whisper - 1)
            timed.append({"word": cap_word, "start": all_words[idx]["start"], "end": all_words[idx]["end"]})
        for i in range(len(timed) - 1):
            timed[i]["end"] = timed[i + 1]["start"]
        lines = group_words_into_lines(timed, words_per_line)

    srt_lines = []
    for i, line in enumerate(lines, 1):
        srt_lines.append(str(i))
        srt_lines.append(f"{seconds_to_srt_time(line['start'])} --> {seconds_to_srt_time(line['end'])}")
        srt_lines.append(line["text"])
        srt_lines.append("")
    return "\n".join(srt_lines)


# -------------------------------------------------------
# Audio-anchored caption timing
# -------------------------------------------------------

def probe_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except Exception:
        return 0.0


async def async_probe_audio_duration(audio_path: str) -> float:
    """Async wrapper around ffprobe for audio duration."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except (ValueError, AttributeError):
        return 0.0


async def async_probe_video_dimensions(video_path: str) -> tuple:
    """Async wrapper around ffprobe for video dimensions."""
    proc = await asyncio.create_subprocess_exec(
        "ffprobe", "-v", "quiet", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        parts = stdout.decode().strip().split(",")
        return int(parts[0]), int(parts[1])
    except (ValueError, IndexError, AttributeError):
        return 640, 360


def build_timed_segments_from_audio_chunks(caption_text, chunk_audio_paths, words_per_caption=12):
    text_chunks = chunk_text(caption_text)
    if len(text_chunks) != len(chunk_audio_paths):
        raise ValueError(
            f"Mismatch: {len(text_chunks)} text chunks vs {len(chunk_audio_paths)} audio files."
        )
    all_timed_words = []
    cursor = 0.0
    for chunk_text_str, audio_path in zip(text_chunks, chunk_audio_paths):
        duration = probe_audio_duration(audio_path)
        words = chunk_text_str.split()
        n = len(words)
        if n == 0 or duration <= 0:
            cursor += duration
            continue
        time_per_word = duration / n
        for i, word in enumerate(words):
            all_timed_words.append({
                "word": word,
                "start": cursor + i * time_per_word,
                "end":   cursor + (i + 1) * time_per_word,
            })
        cursor += duration

    timed_segments = []
    for i in range(0, len(all_timed_words), words_per_caption):
        group = all_timed_words[i:i + words_per_caption]
        timed_segments.append({
            "start": group[0]["start"],
            "end":   group[-1]["end"],
            "text":  " ".join(w["word"] for w in group),
        })
    return timed_segments


def build_timed_segments_from_audio_file(caption_text, audio_path, words_per_caption=12):
    total_duration = probe_audio_duration(audio_path)
    if total_duration <= 0:
        raise ValueError(f"Could not determine duration of {audio_path}")
    words = caption_text.split()
    n = len(words)
    if n == 0:
        return []
    time_per_word = total_duration / n
    all_timed_words = [
        {"word": w, "start": i * time_per_word, "end": (i + 1) * time_per_word}
        for i, w in enumerate(words)
    ]
    timed_segments = []
    for i in range(0, len(all_timed_words), words_per_caption):
        group = all_timed_words[i:i + words_per_caption]
        timed_segments.append({
            "start": group[0]["start"],
            "end":   group[-1]["end"],
            "text":  " ".join(w["word"] for w in group),
        })
    return timed_segments


# -------------------------------------------------------
# Video probe
# -------------------------------------------------------

def probe_video_dimensions(video_path: str) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", video_path],
        capture_output=True, text=True,
    )
    try:
        parts = result.stdout.strip().split(",")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 640, 360


# -------------------------------------------------------
# Overlay caption filtergraph
# -------------------------------------------------------

CHAR_WIDTH_RATIO = 0.55


def compute_max_chars_per_line(font_size, video_width, usable_fraction=0.94):
    usable_px = video_width * usable_fraction
    avg_char_px = font_size * CHAR_WIDTH_RATIO
    return max(10, int(usable_px / avg_char_px))


def wrap_text_to_lines(text, max_chars_per_line):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        if len(word) > max_chars_per_line:
            if current:
                lines.append(current)
                current = ""
            lines.append(word[:max_chars_per_line])
            continue
        if current and len(current) + 1 + len(word) > max_chars_per_line:
            lines.append(current)
            current = word
        else:
            current = (current + " " + word).strip() if current else word
    if current:
        lines.append(current)
    return lines


def build_overlay_filtergraph(timed_segments, font_size, overlay_bar_color, text_color, video_width, video_height):
    def esc(s):
        return (
            s.replace("\\", "\\\\")
             .replace("'", "\u2019")
             .replace(":", "\\:")
             .replace(",", "\\,")
             .replace("[", "\\[")
             .replace("]", "\\]")
        )

    bar_thickness = max(4, font_size // 6)
    line_height   = int(font_size * 1.4)
    v_padding     = int(font_size * 0.35)
    font_path     = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    vid_center_y  = video_height // 2
    max_chars     = compute_max_chars_per_line(font_size, video_width)
    parts         = []

    for seg in timed_segments:
        lines = wrap_text_to_lines(seg["text"], max_chars_per_line=max_chars)
        n_lines         = len(lines)
        rendered_height = (n_lines - 1) * line_height + font_size
        block_top_y     = vid_center_y - (rendered_height // 2)
        block_bot_y     = block_top_y + rendered_height
        box_y           = max(0, block_top_y - v_padding)
        box_h           = rendered_height + v_padding * 2
        bar_top_y       = max(0, box_y - bar_thickness)
        bar_bot_y       = min(video_height - bar_thickness, block_bot_y + v_padding)
        enable_expr     = f"between(t\\,{seg['start']:.3f}\\,{seg['end']:.3f})"

        parts.append(f"drawbox=x=0:y={box_y}:w=iw:h={box_h}:color=black@0.55:t=fill:enable='{enable_expr}'")
        parts.append(f"drawbox=x=0:y={bar_top_y}:w=iw:h={bar_thickness}:color={overlay_bar_color}@1.0:t=fill:enable='{enable_expr}'")
        parts.append(f"drawbox=x=0:y={bar_bot_y}:w=iw:h={bar_thickness}:color={overlay_bar_color}@1.0:t=fill:enable='{enable_expr}'")

        for li, line_text in enumerate(lines):
            y_px = block_top_y + (li * line_height)
            parts.append(
                f"drawtext=text='{esc(line_text)}':fontfile='{font_path}':"
                f"fontsize={font_size}:fontcolor={text_color}:"
                f"x=(w-text_w)/2:y={y_px}:enable='{enable_expr}'"
            )

    return ",\n".join(parts)


def generate_overlay_timed_segments(video_path, caption_text=None, words_per_caption=12):
    model = get_whisper_model()
    segments, _ = model.transcribe(
        video_path, beam_size=5, language="en",
        word_timestamps=True, vad_filter=True,
    )
    all_words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                all_words.append({"word": w.word, "start": w.start, "end": w.end})

    if not all_words:
        return []

    if not caption_text:
        display_words = all_words
    else:
        caption_words = caption_text.split()
        n_caption = len(caption_words)
        n_whisper = len(all_words)
        display_words = []
        for i, cap_word in enumerate(caption_words):
            idx = min(int(i * n_whisper / n_caption), n_whisper - 1)
            display_words.append({
                "word": cap_word,
                "start": all_words[idx]["start"],
                "end":   all_words[idx]["end"],
            })
        for i in range(len(display_words) - 1):
            display_words[i]["end"] = display_words[i + 1]["start"]

    timed_segments = []
    for i in range(0, len(display_words), words_per_caption):
        group = display_words[i:i + words_per_caption]
        timed_segments.append({
            "start": group[0]["start"],
            "end":   group[-1]["end"],
            "text":  " ".join(w["word"].strip() for w in group),
        })
    return timed_segments


async def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path,
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    try:
        return float(stdout.decode().strip())
    except ValueError:
        return 0.0


# -------------------------------------------------------
# FLUX image generation providers
# -------------------------------------------------------

async def _fetch_image_bytes(url: str) -> tuple:
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        r = await client.get(url)
        r.raise_for_status()
        ctype = r.headers.get("content-type", "image/png").split(";")[0].strip()
        return r.content, ctype


def _decode_data_uri(data_uri: str) -> tuple:
    header, payload = data_uri.split(",", 1)
    ctype = header.split(":", 1)[1].split(";", 1)[0] or "image/png"
    return base64.b64decode(payload), ctype


async def generate_image_fal(api_key, prompt, width, height, steps=4, seed=None):
    url = "https://fal.run/fal-ai/flux/schnell"
    payload = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_inference_steps": steps,
        "num_images": 1,
        "enable_safety_checker": True,
    }
    if seed is not None:
        payload["seed"] = seed

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            url, json=payload,
            headers={
                "Authorization": f"Key {api_key}",
                "Content-Type": "application/json",
            },
        )
        if r.status_code >= 400:
            raise RuntimeError(f"fal.ai HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()

    images = data.get("images") or []
    if not images:
        raise RuntimeError(f"fal.ai returned no images: {data}")

    first   = images[0]
    img_url = first.get("url") if isinstance(first, dict) else first
    if not img_url:
        raise RuntimeError(f"fal.ai image entry missing url: {first}")

    if img_url.startswith("data:"):
        return _decode_data_uri(img_url)
    return await _fetch_image_bytes(img_url)


async def generate_image_together(api_key, prompt, width, height, steps=4, seed=None):
    url = "https://api.together.xyz/v1/images/generations"
    payload = {
        "model":  "black-forest-labs/FLUX.1-schnell-Free",
        "prompt": prompt,
        "width":  width,
        "height": height,
        "steps":  steps,
        "n": 1,
        "response_format": "b64_json",
    }
    if seed is not None:
        payload["seed"] = seed

    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            url, json=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        if r.status_code >= 400:
            raise RuntimeError(f"Together.ai HTTP {r.status_code}: {r.text[:400]}")
        data = r.json()

    entries = data.get("data") or []
    if not entries:
        raise RuntimeError(f"Together.ai returned no data: {data}")

    entry = entries[0]
    if "b64_json" in entry and entry["b64_json"]:
        return base64.b64decode(entry["b64_json"]), "image/png"
    if "url" in entry and entry["url"]:
        return await _fetch_image_bytes(entry["url"])

    raise RuntimeError(f"Together.ai entry missing image data: {entry}")


# -------------------------------------------------------
# Slideshow: word-by-word caption filtergraph
# -------------------------------------------------------

def build_slideshow_caption_filtergraph(
    text: str,
    audio_dur: float,
    pad_start: float,
    font_size: int,
    font_color: str,
    caption_position: str,
    out_w: int,
    out_h: int,
    words_per_caption: int,
) -> str:
    """
    Build a drawtext filtergraph that cycles caption groups word-by-word
    across the slide duration.
    """

    def esc(s):
        return (
            s.replace("\\", "\\\\")
             .replace("'", "\u2019")
             .replace(":", "\\:")
             .replace(",", "\\,")
             .replace("[", "\\[")
             .replace("]", "\\]")
        )

    font_path    = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    shadow_color = "black@0.8"
    words        = text.split()
    n_words      = len(words)

    if n_words == 0:
        return "null"

    groups = []
    for i in range(0, n_words, words_per_caption):
        groups.append(" ".join(words[i:i + words_per_caption]))

    n_groups       = len(groups)
    time_per_group = audio_dur / n_groups if n_groups > 0 else audio_dur

    margin = int(out_h * 0.08)

    if caption_position == "top":
        y_expr = str(margin)
    elif caption_position == "center":
        y_expr = "(h-text_h)/2"
    else:
        y_expr = f"h-text_h-{margin}"

    parts = []
    for gi, group_text in enumerate(groups):
        t_start = pad_start + gi * time_per_group
        t_end   = pad_start + (gi + 1) * time_per_group
        enable  = f"between(t\\,{t_start:.3f}\\,{t_end:.3f})"
        escaped = esc(group_text)

        parts.append(
            f"drawtext=text='{escaped}':"
            f"fontfile='{font_path}':"
            f"fontsize={font_size}:"
            f"fontcolor={shadow_color}:"
            f"x=(w-text_w)/2+3:y={y_expr}+3:"
            f"enable='{enable}'"
        )
        parts.append(
            f"drawtext=text='{escaped}':"
            f"fontfile='{font_path}':"
            f"fontsize={font_size}:"
            f"fontcolor={font_color}:"
            f"x=(w-text_w)/2:y={y_expr}:"
            f"enable='{enable}'"
        )

    return ",\n".join(parts)


# -------------------------------------------------------
# Slideshow: per-slide segment builder
# -------------------------------------------------------

async def _tts_bytes(kokoro_base_url: str, text: str, voice: str, speed: float) -> bytes:
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": voice,
        "speed": speed,
        "response_format": "mp3",
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(
            f"{kokoro_base_url}/v1/audio/speech",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        r.raise_for_status()
        return r.content


async def build_slide_segment(
    slide,
    index: int,
    tmp_dir: str,
    kokoro_base_url: str,
    voice: str,
    speed: float,
    font_size: int,
    font_color: str,
    caption_position: str,
    words_per_caption: int,
    pad_start: float,
    pad_end: float,
    out_w: int = 720,
    out_h: int = 1280,
) -> tuple:
    """Build one slide MP4 segment."""
    prefix      = f"slide_{index:04d}"
    img_path    = os.path.join(tmp_dir, f"{prefix}_img")
    audio_path  = os.path.join(tmp_dir, f"{prefix}_audio.mp3")
    padded_audio = os.path.join(tmp_dir, f"{prefix}_padded.mp3")
    seg_path    = os.path.join(tmp_dir, f"{prefix}_seg.mp4")
    tmp_files   = [img_path, audio_path, padded_audio]

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        img_resp, audio_bytes = await asyncio.gather(
            client.get(slide.image_url),
            _tts_bytes(kokoro_base_url, slide.text, voice, speed),
        )
    img_resp.raise_for_status()

    ctype    = img_resp.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    ext      = "png" if "png" in ctype else "webp" if "webp" in ctype else "jpg"
    img_path += f".{ext}"
    tmp_files[0] = img_path

    with open(img_path, "wb") as f:
        f.write(img_resp.content)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)

    audio_dur      = probe_audio_duration(audio_path)
    if audio_dur <= 0:
        raise RuntimeError(f"Slide {index}: could not determine TTS audio duration.")
    slide_duration = pad_start + audio_dur + pad_end

    pad_start_ms = int(pad_start * 1000)
    pad_cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-af", f"adelay={pad_start_ms}|{pad_start_ms},apad=pad_dur={pad_end:.3f}",
        "-t", f"{slide_duration:.3f}",
        padded_audio,
    ]
    proc = await asyncio.create_subprocess_exec(
        *pad_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Slide {index} audio pad error: {stderr.decode()[:300]}")

    scale_pad_vf = (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:color=black"
    )

    if caption_position and caption_position.lower() in ("top", "center", "bottom"):
        caption_vf = build_slideshow_caption_filtergraph(
            text=slide.text,
            audio_dur=audio_dur,
            pad_start=pad_start,
            font_size=font_size,
            font_color=font_color,
            caption_position=caption_position.lower(),
            out_w=out_w,
            out_h=out_h,
            words_per_caption=words_per_caption,
        )
        full_vf = f"{scale_pad_vf},{caption_vf}"
    else:
        full_vf = scale_pad_vf

    compose_cmd = [
        "ffmpeg", "-y",
        "-loop", "1", "-framerate", "25", "-i", img_path,
        "-i", padded_audio,
        "-vf", full_vf,
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-c:a", "aac", "-b:a", "128k",
        "-t", f"{slide_duration:.3f}",
        "-pix_fmt", "yuv420p",
        seg_path,
    ]
    proc2 = await asyncio.create_subprocess_exec(
        *compose_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr2 = await proc2.communicate()
    if proc2.returncode != 0:
        raise RuntimeError(f"Slide {index} compose error: {stderr2.decode()[:400]}")

    return seg_path, tmp_files

# -------------------------------------------------------
# Text Thumbnail Generator  (YouTube-style layout)
# -------------------------------------------------------

async def generate_text_thumbnail(
    top_text: str,
    bottom_text: Optional[str] = None,
    avatar_url: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    bg_color: str = "#000000",
    top_text_color: str = "#ffffff",
    bottom_bg_color: str = "#FFD700",
    bottom_text_color: str = "#000000",
    top_font_size: int = 89,
    bottom_font_size: int = 44,
    font_path: Optional[str] = None,
) -> tuple:
    """
    Layout:
      - Right panel (~40%): avatar image, full 720px height, cover-cropped, no truncation
      - Left panel (~60%): black bg
          - top section: large bold white ALL-CAPS top_text, vertically centred above banner
          - bottom banner: yellow strip spanning left panel only, bold black bottom_text
    """
    from PIL import Image, ImageDraw, ImageFont
    import io
    import httpx as _httpx

    AVATAR_W     = 380                    # fixed avatar column width (px)
    BANNER_RATIO = 0.155
    TEXT_PADDING = 52
    LINE_GAP     = 12

    banner_h = int(height * BANNER_RATIO)
    text_h   = height - banner_h          # usable height above banner
    avatar_w = AVATAR_W                   # fixed 380px regardless of canvas width
    text_w   = width - avatar_w           # left panel = 900px at 1280 wide

    def _parse_hex(h: str):
        h = h.lstrip("#")
        if len(h) == 3:
            h = "".join(c * 2 for c in h)
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _load_font(size: int):
        if font_path:
            try:
                return ImageFont.truetype(font_path, size)
            except (IOError, OSError):
                pass
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
        ):
            try:
                return ImageFont.truetype(candidate, size)
            except (IOError, OSError):
                pass
        return ImageFont.load_default()

    # ── Base canvas ───────────────────────────────────────────────────────────
    img  = Image.new("RGB", (width, height), color=_parse_hex(bg_color))
    draw = ImageDraw.Draw(img)

    # ── Avatar panel (right) — full height, cover crop ───────────────────────
    if avatar_url:
        try:
            async with _httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                resp = await client.get(avatar_url)
                resp.raise_for_status()
            avatar_img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            orig_w, orig_h = avatar_img.size

            # Cover scale: fill full height AND avatar_w, no black bars
            scale = max(height / orig_h, avatar_w / orig_w)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            avatar_img = avatar_img.resize((new_w, new_h), Image.LANCZOS)

            # Centre-crop to (avatar_w x height)
            left = (new_w - avatar_w) // 2
            top  = (new_h - height)   // 2
            avatar_img = avatar_img.crop((left, top, left + avatar_w, top + height))
            # Paste at full height — avatar is NOT masked by banner
            img.paste(avatar_img, (text_w, 0))
        except Exception:
            pass

    # ── Bottom banner — LEFT PANEL ONLY ──────────────────────────────────────
    banner_y = height - banner_h
    draw.rectangle(
        [(0, banner_y), (text_w, height)],          # only spans left panel width
        fill=_parse_hex(bottom_bg_color)
    )

    # ── Word-wrap helper ─────────────────────────────────────────────────────
    def _wrap(text: str, font, max_w: int) -> list:
        words, lines, current = text.split(), [], ""
        for word in words:
            test = (current + " " + word).strip()
            bb = draw.textbbox((0, 0), test, font=font)
            if bb[2] > max_w and current:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)
        return lines

    def _block_height(lines, font, gap):
        total = 0
        for ln in lines:
            bb = draw.textbbox((0, 0), ln, font=font)
            total += (bb[3] - bb[1]) + gap
        return max(0, total - gap)

    # ── Auto-shrink top font until text fits above banner ────────────────────
    usable_w = text_w - TEXT_PADDING * 2
    usable_h = text_h - (TEXT_PADDING // 2) - TEXT_PADDING  # top margin halved
    font_size = top_font_size
    MIN_FONT  = 28

    while font_size >= MIN_FONT:
        top_font  = _load_font(font_size)
        top_lines = _wrap(top_text.upper(), top_font, usable_w)
        if _block_height(top_lines, top_font, LINE_GAP) <= usable_h:
            break
        font_size -= 4

    # ── Draw top text — vertically centred in text_h ─────────────────────────
    blk_h = _block_height(top_lines, top_font, LINE_GAP)
    y = (TEXT_PADDING // 2) + ((text_h - (TEXT_PADDING // 2) - TEXT_PADDING - blk_h) // 2)

    for ln in top_lines:
        bb = draw.textbbox((0, 0), ln, font=top_font)
        lw, lh = bb[2] - bb[0], bb[3] - bb[1]
        x = TEXT_PADDING + (usable_w - lw) // 2
        draw.text((x, y), ln, font=top_font, fill=_parse_hex(top_text_color))
        y += lh + LINE_GAP

    # ── Bottom banner text — constrained to left panel width, auto-shrink ──────
    if bottom_text:
        bot_usable_w  = text_w - TEXT_PADDING * 2
        bot_usable_h  = banner_h - 12           # small vertical margin
        bfs           = bottom_font_size
        BOT_MIN_FONT  = 18
        BOT_LINE_GAP  = 6
        while bfs >= BOT_MIN_FONT:
            bot_font  = _load_font(bfs)
            bot_lines = _wrap(bottom_text.upper(), bot_font, bot_usable_w)
            if _block_height(bot_lines, bot_font, BOT_LINE_GAP) <= bot_usable_h:
                break
            bfs -= 2
        bot_blk_h = _block_height(bot_lines, bot_font, BOT_LINE_GAP)
        BOT_PAD_TOP = 8
        BOT_PAD_BOT = 24
        by = banner_y + BOT_PAD_TOP + ((banner_h - BOT_PAD_TOP - BOT_PAD_BOT - bot_blk_h) // 2)
        for ln in bot_lines:
            bb  = draw.textbbox((0, 0), ln, font=bot_font)
            lw, lh = bb[2] - bb[0], bb[3] - bb[1]
            bx  = TEXT_PADDING + (bot_usable_w - lw) // 2
            draw.text((bx, by), ln, font=bot_font, fill=_parse_hex(bottom_text_color))
            by += lh + BOT_LINE_GAP

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read(), "image/png"
