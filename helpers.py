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
            headers={"Authorization": f"Key {api_key}", "Content
