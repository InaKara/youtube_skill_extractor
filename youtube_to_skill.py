#!/usr/bin/env python3
"""
youtube_to_markdown_full.py

Convert a YouTube video into a Markdown notes bundle.

Pipeline:
1. Source acquisition: yt-dlp metadata, captions, optional video
2. Transcript extraction: captions, fallback Whisper/faster-whisper
3. Keyframe extraction: PySceneDetect, fallback ffmpeg fixed interval
4. Frame filtering: simple dedupe/ranking, fallback even sampling
5. Visual analysis: OpenAI-compatible vision model, fallback Tesseract OCR
6. Chart/table extraction: Docling if available, fallback visual descriptions
7. Timeline merge: align transcript + visuals by timestamp
8. Markdown composition: OpenAI-compatible LLM, fallback deterministic template

Required for useful runs:
- yt-dlp
- ffmpeg

Optional:
- faster-whisper or whisper
- scenedetect
- pillow + pytesseract
- docling
- OPENAI_API_KEY for vision/composition
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------

YTDLP_COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "")
YTDLP_COOKIES = os.environ.get("YTDLP_COOKIES", "")
YTDLP_RETRIES = os.environ.get("YTDLP_RETRIES", "10")
YTDLP_SLEEP_INTERVAL = os.environ.get("YTDLP_SLEEP_INTERVAL", "1")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_TEXT_MODEL = os.environ.get("OPENAI_TEXT_MODEL", os.environ.get("OPENAI_MODEL", "gpt-5.5"))
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", OPENAI_TEXT_MODEL)

DEFAULT_LANGS = os.environ.get("YOUTUBE_TRANSCRIPT_LANGS", "en,de,tr")
DEFAULT_MAX_FRAMES = int(os.environ.get("YOUTUBE_MAX_FRAMES", "24"))
DEFAULT_SCENE_THRESHOLD = float(os.environ.get("YOUTUBE_SCENE_THRESHOLD", "27.0"))
DEFAULT_VIDEO_HEIGHT = int(os.environ.get("YOUTUBE_VIDEO_HEIGHT", "720"))


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------


def which_or_none(command: str) -> str | None:
    return shutil.which(command)


def resolve_ffmpeg() -> str:
    ffmpeg_location = os.environ.get("FFMPEG_LOCATION", "")
    if ffmpeg_location:
        p = Path(ffmpeg_location)
        if p.is_file():
            return str(p)
        candidate = p / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if candidate.exists():
            return str(candidate)

    found = shutil.which("ffmpeg")
    if found:
        return found

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


FFMPEG_EXE = resolve_ffmpeg()


def run_cmd(cmd: list[str], *, cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def ytdlp_flags() -> list[str]:
    flags = ["--retries", YTDLP_RETRIES, "--sleep-interval", YTDLP_SLEEP_INTERVAL]
    if YTDLP_COOKIES_FROM_BROWSER:
        flags += ["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER]
    if YTDLP_COOKIES:
        flags += ["--cookies", YTDLP_COOKIES]
    return flags


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower().strip())
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len] or "youtube-video"


def extract_video_id(url_or_id: str) -> str | None:
    raw = url_or_id.strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{8,20}", raw):
        return raw

    parsed = urlparse(raw)
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0] or None

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            q = parse_qs(parsed.query)
            if "v" in q:
                return q["v"][0]

        parts = [p for p in parsed.path.split("/") if p]
        for marker in ("shorts", "embed", "live"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1]

    return None


def prompt_bool(question: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "true", "1"}:
            return True
        if answer in {"n", "no", "false", "0"}:
            return False
        print("Please answer yes or no.")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def append_log(ctx: "PipelineContext", message: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    line = f"[{ts}] {message}\n"
    ctx.logs_dir.mkdir(parents=True, exist_ok=True)
    with (ctx.logs_dir / "pipeline.log").open("a", encoding="utf-8") as f:
        f.write(line)


def step(ctx: "PipelineContext", name: str) -> None:
    print(f"\n=== {name} ===")
    append_log(ctx, name)


def seconds_to_ts(seconds: float) -> str:
    seconds = max(0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def ts_to_seconds(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(ts)
    except Exception:
        return 0.0


def clean_vtt_text(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]+\}", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[...truncated...]\n"


# -----------------------------------------------------------------------------
# Pipeline context
# -----------------------------------------------------------------------------


@dataclass
class PipelineContext:
    url: str
    video_id: str | None
    visual_analysis: bool
    created_at: str
    root_dir: Path
    langs: list[str]
    max_frames: int
    scene_threshold: float
    video_height: int

    @property
    def input_dir(self) -> Path:
        return self.root_dir / "input"

    @property
    def raw_dir(self) -> Path:
        return self.root_dir / "raw"

    @property
    def transcript_dir(self) -> Path:
        return self.root_dir / "transcript"

    @property
    def visuals_dir(self) -> Path:
        return self.root_dir / "visuals"

    @property
    def frames_dir(self) -> Path:
        return self.visuals_dir / "frames"

    @property
    def selected_frames_dir(self) -> Path:
        return self.visuals_dir / "selected"

    @property
    def combined_dir(self) -> Path:
        return self.root_dir / "combined"

    @property
    def output_dir(self) -> Path:
        return self.root_dir / "output"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"


@dataclass
class PipelineState:
    metadata: dict[str, Any] | None = None
    captions_path: Path | None = None
    video_path: Path | None = None
    audio_path: Path | None = None
    transcript_segments: list[dict[str, Any]] | None = None
    frames: list[dict[str, Any]] | None = None
    selected_frames: list[dict[str, Any]] | None = None
    visual_observations: list[dict[str, Any]] | None = None


def create_context(args: argparse.Namespace) -> PipelineContext:
    url = args.url or input("YouTube URL or video ID: ").strip()
    if not url:
        raise ValueError("No URL provided.")

    if args.visual_analysis is None:
        visual_analysis = prompt_bool("Do you want visual frame analysis?", default=True)
    else:
        visual_analysis = args.visual_analysis == "yes"

    video_id = extract_video_id(url)
    created_at = datetime.now().isoformat(timespec="seconds")
    folder_name = f"youtube_{video_id}" if video_id else f"youtube_{slugify(url)}"
    root_dir = (args.out_dir or Path.cwd()) / folder_name

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    return PipelineContext(
        url=url,
        video_id=video_id,
        visual_analysis=visual_analysis,
        created_at=created_at,
        root_dir=root_dir.resolve(),
        langs=langs,
        max_frames=args.max_frames,
        scene_threshold=args.scene_threshold,
        video_height=args.video_height,
    )


def initialize_output_tree(ctx: PipelineContext) -> None:
    step(ctx, "0. Initialize output tree")

    for directory in [
        ctx.input_dir,
        ctx.raw_dir,
        ctx.transcript_dir,
        ctx.frames_dir,
        ctx.selected_frames_dir,
        ctx.combined_dir,
        ctx.output_dir,
        ctx.logs_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    write_text(ctx.input_dir / "source_url.txt", ctx.url)
    write_json(
        ctx.logs_dir / "run_config.json",
        {
            "created_at": ctx.created_at,
            "url": ctx.url,
            "video_id": ctx.video_id,
            "visual_analysis": ctx.visual_analysis,
            "langs": ctx.langs,
            "max_frames": ctx.max_frames,
            "scene_threshold": ctx.scene_threshold,
            "video_height": ctx.video_height,
            "root_dir": str(ctx.root_dir),
        },
    )
    print(f"Output folder: {ctx.root_dir}")


# -----------------------------------------------------------------------------
# Stage 1: Source acquisition
# -----------------------------------------------------------------------------


def fetch_metadata(ctx: PipelineContext) -> dict[str, Any]:
    if not which_or_none("yt-dlp"):
        raise RuntimeError("yt-dlp not found. Install it first: pip install yt-dlp")

    cmd = ["yt-dlp", "--dump-json", "--skip-download", *ytdlp_flags(), ctx.url]
    result = run_cmd(cmd, timeout=120)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"yt-dlp metadata failed:\n{result.stderr[:2000]}")

    meta = json.loads(result.stdout)
    chapters = [
        {
            "start_time": c.get("start_time"),
            "end_time": c.get("end_time"),
            "title": c.get("title"),
        }
        for c in (meta.get("chapters") or [])
    ]

    wanted = {
        "id": meta.get("id"),
        "title": meta.get("title"),
        "channel": meta.get("channel") or meta.get("uploader"),
        "duration": meta.get("duration"),
        "webpage_url": meta.get("webpage_url") or ctx.url,
        "upload_date": meta.get("upload_date"),
        "description": meta.get("description", ""),
        "chapters": chapters,
        "availability": meta.get("availability"),
        "language": meta.get("language"),
    }
    write_json(ctx.raw_dir / "metadata.json", wanted)
    write_json(ctx.raw_dir / "metadata_full.json", meta)
    return wanted


def download_captions(ctx: PipelineContext) -> Path | None:
    # yt-dlp writes captions with language in the filename. We find the newest .vtt afterwards.
    before = {p.resolve() for p in ctx.raw_dir.glob("*.vtt")}
    out_template = str(ctx.raw_dir / "captions.%(id)s.%(ext)s")
    sub_langs = ",".join(ctx.langs)

    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-langs",
        sub_langs,
        "--sub-format",
        "vtt/best",
        "-o",
        out_template,
        *ytdlp_flags(),
        ctx.url,
    ]
    result = run_cmd(cmd, timeout=180)
    if result.returncode != 0:
        append_log(ctx, f"Caption download failed: {result.stderr[:1000]}")

    candidates = [p for p in ctx.raw_dir.glob("*.vtt") if p.resolve() not in before]
    if not candidates:
        candidates = list(ctx.raw_dir.glob("*.vtt"))
    if not candidates:
        return None

    # Prefer first configured language appearing in name.
    for lang in ctx.langs:
        matching = [p for p in candidates if f".{lang}." in p.name or p.name.endswith(f".{lang}.vtt")]
        if matching:
            captions = max(matching, key=lambda p: p.stat().st_mtime)
            shutil.copy2(captions, ctx.raw_dir / "captions.vtt")
            return ctx.raw_dir / "captions.vtt"

    captions = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copy2(captions, ctx.raw_dir / "captions.vtt")
    return ctx.raw_dir / "captions.vtt"


def download_video(ctx: PipelineContext) -> Path | None:
    for old in ctx.raw_dir.glob("video.*"):
        if old.suffix.lower() not in {".placeholder", ".txt"}:
            old.unlink(missing_ok=True)

    fmt = f"best[ext=mp4][height<={ctx.video_height}]/best[height<={ctx.video_height}]/best"
    cmd = [
        "yt-dlp",
        "-f",
        fmt,
        "-o",
        str(ctx.raw_dir / "video.%(ext)s"),
        *ytdlp_flags(),
        ctx.url,
    ]
    result = run_cmd(cmd, timeout=900)
    if result.returncode != 0:
        append_log(ctx, f"Video download failed: {result.stderr[:1500]}")
        return None

    candidates = [
        p for p in ctx.raw_dir.glob("video.*")
        if p.suffix.lower() not in {".part", ".ytdl", ".json", ".txt"}
    ]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def acquire_source(ctx: PipelineContext, state: PipelineState) -> None:
    step(ctx, "1. Source acquisition")

    state.metadata = fetch_metadata(ctx)
    print(f"Metadata: {state.metadata.get('title') or 'Unknown title'}")

    state.captions_path = download_captions(ctx)
    if state.captions_path:
        print(f"Captions: {state.captions_path.name}")
    else:
        print("Captions: not found; transcript fallback may use Whisper.")

    if ctx.visual_analysis:
        state.video_path = download_video(ctx)
        if state.video_path:
            print(f"Video: {state.video_path.name}")
        else:
            print("Video download failed; visual analysis will be skipped unless a video is added manually.")
    else:
        print("Video download skipped because visual frame analysis is disabled.")


# -----------------------------------------------------------------------------
# Stage 2: Transcript extraction
# -----------------------------------------------------------------------------


def parse_vtt(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    segments: list[dict[str, Any]] = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if "-->" not in line:
            i += 1
            continue

        timing = line
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        parts = timing.split("-->")
        if len(parts) != 2:
            continue

        start_raw = parts[0].strip().split()[0]
        end_raw = parts[1].strip().split()[0]
        segment_text = clean_vtt_text(" ".join(text_lines))
        if not segment_text:
            continue

        # Avoid common duplicate caption lines.
        if segments and segments[-1]["text"] == segment_text:
            continue

        segments.append(
            {
                "start": ts_to_seconds(start_raw),
                "end": ts_to_seconds(end_raw),
                "text": segment_text,
                "source": "youtube_captions",
            }
        )

    return segments


def download_audio_for_whisper(ctx: PipelineContext) -> Path | None:
    audio_path = ctx.raw_dir / "audio.wav"
    if audio_path.exists() and audio_path.stat().st_size > 0:
        return audio_path

    cmd = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        str(ctx.raw_dir / "audio.%(ext)s"),
        *ytdlp_flags(),
        ctx.url,
    ]
    result = run_cmd(cmd, timeout=900)
    if result.returncode != 0:
        append_log(ctx, f"Audio download/extract failed: {result.stderr[:1500]}")
        return None

    candidates = list(ctx.raw_dir.glob("audio*.wav"))
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def transcribe_with_whisper(ctx: PipelineContext, audio_path: Path) -> list[dict[str, Any]]:
    # Primary optional package: faster-whisper.
    try:
        from faster_whisper import WhisperModel

        model_name = os.environ.get("WHISPER_MODEL", "base")
        device = os.environ.get("WHISPER_DEVICE", "auto")
        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "default")
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        segments_iter, _info = model.transcribe(str(audio_path), vad_filter=True)
        segments = []
        for s in segments_iter:
            text = str(s.text).strip()
            if text:
                segments.append(
                    {
                        "start": float(s.start),
                        "end": float(s.end),
                        "text": text,
                        "source": "faster_whisper",
                    }
                )
        return segments
    except Exception as e:
        append_log(ctx, f"faster-whisper unavailable/failed: {e}")

    # Same fallback step, alternate local Whisper package.
    try:
        import whisper

        model_name = os.environ.get("WHISPER_MODEL", "base")
        model = whisper.load_model(model_name)
        result = model.transcribe(str(audio_path))
        segments = []
        for s in result.get("segments", []):
            text = str(s.get("text", "")).strip()
            if text:
                segments.append(
                    {
                        "start": float(s.get("start", 0.0)),
                        "end": float(s.get("end", 0.0)),
                        "text": text,
                        "source": "whisper",
                    }
                )
        return segments
    except Exception as e:
        append_log(ctx, f"whisper unavailable/failed: {e}")
        return []


def render_transcript_md(segments: list[dict[str, Any]]) -> str:
    return "\n".join(f"[{seconds_to_ts(float(s['start']))}] {s['text']}" for s in segments if s.get("text"))


def extract_transcript(ctx: PipelineContext, state: PipelineState) -> None:
    step(ctx, "2. Transcript extraction")

    segments: list[dict[str, Any]] = []

    if state.captions_path and state.captions_path.exists():
        segments = parse_vtt(state.captions_path)
        print(f"Caption segments: {len(segments)}")

    if not segments:
        print("No caption transcript found. Trying Whisper fallback...")
        state.audio_path = download_audio_for_whisper(ctx)
        if state.audio_path:
            segments = transcribe_with_whisper(ctx, state.audio_path)
            print(f"Whisper segments: {len(segments)}")
        else:
            print("Audio unavailable; transcript will be empty.")

    if not segments:
        segments = [
            {
                "start": 0.0,
                "end": 0.0,
                "text": "No transcript available.",
                "source": "empty",
            }
        ]

    state.transcript_segments = segments
    write_json(ctx.transcript_dir / "transcript_raw.json", segments)
    write_text(ctx.transcript_dir / "transcript_clean.md", render_transcript_md(segments))


# -----------------------------------------------------------------------------
# Stage 3: Keyframe / visual extraction
# -----------------------------------------------------------------------------


def extract_frame_at(video_path: Path, timestamp: float, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_EXE,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        "scale='min(1280,iw)':-2",
        "-q:v",
        "3",
        str(out_path),
    ]
    result = run_cmd(cmd, timeout=60)
    return result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def detect_scenes_pyscenedetect(ctx: PipelineContext, video_path: Path) -> list[float]:
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        video = open_video(str(video_path))
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=ctx.scene_threshold))
        scene_manager.detect_scenes(video)
        scene_list = scene_manager.get_scene_list()
        timestamps = []
        for start, _end in scene_list:
            # Use a slight offset to avoid black transition frames.
            timestamps.append(max(0.0, float(start.get_seconds()) + 0.35))
        return timestamps
    except Exception as e:
        append_log(ctx, f"PySceneDetect unavailable/failed: {e}")
        return []


def fixed_interval_timestamps(duration: float | None, max_frames: int) -> list[float]:
    if not duration or duration <= 0:
        return [float(i * 30) for i in range(max_frames)]

    if max_frames <= 1:
        return [0.0]

    interval = max(10.0, duration / max_frames)
    ts = []
    cur = 0.0
    while cur < duration and len(ts) < max_frames:
        ts.append(cur)
        cur += interval
    return ts


def extract_keyframes(ctx: PipelineContext, state: PipelineState) -> None:
    step(ctx, "3. Keyframe / visual extraction")

    if not ctx.visual_analysis:
        manifest = {"enabled": False, "reason": "visual frame analysis disabled", "frames": []}
        write_json(ctx.visuals_dir / "frame_manifest.json", manifest)
        state.frames = []
        print("Skipped.")
        return

    if not state.video_path or not state.video_path.exists():
        manifest = {"enabled": False, "reason": "video unavailable", "frames": []}
        write_json(ctx.visuals_dir / "frame_manifest.json", manifest)
        state.frames = []
        print("Skipped because video is unavailable.")
        return

    metadata = state.metadata or read_json(ctx.raw_dir / "metadata.json", {})
    duration = metadata.get("duration")

    timestamps = detect_scenes_pyscenedetect(ctx, state.video_path)
    method = "pyscenedetect"

    if not timestamps:
        timestamps = fixed_interval_timestamps(float(duration or 0), ctx.max_frames)
        method = "ffmpeg_fixed_interval"

    # Limit early to avoid extracting too many frames.
    if ctx.max_frames > 0 and len(timestamps) > ctx.max_frames:
        indices = {round(i * (len(timestamps) - 1) / (ctx.max_frames - 1)) for i in range(ctx.max_frames)}
        timestamps = [t for i, t in enumerate(timestamps) if i in indices]

    frames = []
    for idx, timestamp in enumerate(timestamps, 1):
        out_path = ctx.frames_dir / f"frame_{idx:04d}.jpg"
        ok = extract_frame_at(state.video_path, timestamp, out_path)
        if not ok:
            append_log(ctx, f"Could not extract frame at {timestamp:.3f}s")
            continue
        frames.append(
            {
                "frame_id": f"frame_{idx:04d}",
                "file": out_path.relative_to(ctx.root_dir).as_posix(),
                "timestamp_seconds": float(timestamp),
                "timestamp": seconds_to_ts(timestamp),
                "scene_number": idx,
                "extraction_reason": method,
            }
        )

    manifest = {"enabled": True, "method": method, "frames": frames}
    write_json(ctx.visuals_dir / "frame_manifest.json", manifest)
    state.frames = frames
    print(f"Extracted frames: {len(frames)} using {method}")


# -----------------------------------------------------------------------------
# Stage 4: Frame filtering / deduplication
# -----------------------------------------------------------------------------


def image_fingerprint(path: Path) -> tuple[int, ...] | None:
    try:
        from PIL import Image

        img = Image.open(path).convert("L").resize((8, 8))
        values = list(img.getdata())
        avg = sum(values) / len(values)
        return tuple(1 if v >= avg else 0 for v in values)
    except Exception:
        return None


def hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(x != y for x, y in zip(a, b))


def ocr_text(path: Path) -> str:
    try:
        from PIL import Image
        import pytesseract

        text = pytesseract.image_to_string(Image.open(path))
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
    except Exception:
        return ""


def filter_frames(ctx: PipelineContext, state: PipelineState) -> None:
    step(ctx, "4. Frame filtering / deduplication")

    if not ctx.visual_analysis or not state.frames:
        selected = {"enabled": False, "reason": "no frames", "selected_frames": []}
        write_json(ctx.visuals_dir / "selected_frames.json", selected)
        state.selected_frames = []
        print("Skipped.")
        return

    scored = []
    fingerprints: list[tuple[int, ...]] = []
    for frame in state.frames:
        source_path = ctx.root_dir / frame["file"]
        fp = image_fingerprint(source_path)
        is_duplicate = False
        if fp is not None:
            for prev in fingerprints:
                if hamming(fp, prev) <= 5:
                    is_duplicate = True
                    break
            if not is_duplicate:
                fingerprints.append(fp)

        text = ocr_text(source_path)
        text_score = min(len(text), 1000)
        uniqueness_score = 0 if is_duplicate else 100
        score = text_score + uniqueness_score
        scored.append((score, is_duplicate, text, frame))

    # Keep non-duplicates first. If OCR/PIL not available, this still keeps original order.
    scored.sort(key=lambda x: (x[1], -x[0], x[3]["timestamp_seconds"]))
    kept = scored[: ctx.max_frames] if ctx.max_frames > 0 else scored
    kept.sort(key=lambda x: x[3]["timestamp_seconds"])

    selected_frames = []
    for idx, (_score, is_duplicate, text, frame) in enumerate(kept, 1):
        source_path = ctx.root_dir / frame["file"]
        dst = ctx.selected_frames_dir / f"selected_{idx:04d}.jpg"
        shutil.copy2(source_path, dst)
        selected_frames.append(
            {
                "selected_id": f"selected_{idx:04d}",
                "frame_id": frame["frame_id"],
                "source_file": frame["file"],
                "selected_file": dst.relative_to(ctx.root_dir).as_posix(),
                "timestamp_seconds": frame["timestamp_seconds"],
                "timestamp": frame["timestamp"],
                "selection_reason": "dedupe_ranked" if not is_duplicate else "kept_despite_similarity",
                "ocr_preview": text[:500],
            }
        )

    payload = {"enabled": True, "method": "dedupe_rank_or_even_sample", "selected_frames": selected_frames}
    write_json(ctx.visuals_dir / "selected_frames.json", payload)
    state.selected_frames = selected_frames
    print(f"Selected frames: {len(selected_frames)}")


# -----------------------------------------------------------------------------
# Stage 5: Visual analysis
# -----------------------------------------------------------------------------


def encode_image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def call_openai_responses(payload: dict[str, Any], timeout: int = 120) -> dict[str, Any] | None:
    if not OPENAI_API_KEY:
        return None

    try:
        import requests

        resp = requests.post(
            f"{OPENAI_BASE_URL}/responses",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        if resp.status_code >= 400:
            return {"_error": f"HTTP {resp.status_code}: {resp.text[:1000]}"}
        return resp.json()
    except Exception as e:
        return {"_error": str(e)}


def extract_response_text(resp: dict[str, Any] | None) -> str:
    if not resp:
        return ""
    if "_error" in resp:
        return ""

    # Responses API normal shape.
    try:
        parts = []
        for item in resp.get("output", []):
            for content in item.get("content", []):
                if content.get("type") in {"output_text", "text"} and content.get("text"):
                    parts.append(content["text"])
        if parts:
            return "\n".join(parts).strip()
    except Exception:
        pass

    # Fallback for compatible APIs.
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def analyze_frame_with_vision(ctx: PipelineContext, frame: dict[str, Any], image_path: Path, ocr: str) -> dict[str, Any] | None:
    prompt = f"""Analyze this YouTube video frame for Markdown note generation.

Return strict JSON with these keys:
- type: one of slide, chart, table, diagram, code, ui_screenshot, demo, person, other
- description: concise explanation of what is visible
- visible_text: important visible text, not every OCR artifact
- extracted_data: array of simple objects for visible chart/table data if obvious, otherwise []
- importance: high, medium, or low
- requires_chart_or_table_extraction: boolean

Timestamp: {frame.get('timestamp')}
OCR preview:
{ocr[:2000]}
"""

    payload = {
        "model": OPENAI_VISION_MODEL,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": encode_image_data_url(image_path)},
                ],
            }
        ],
        "temperature": 0.1,
    }

    resp = call_openai_responses(payload, timeout=120)
    text = extract_response_text(resp)
    if not text:
        return None

    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        return {"type": "other", "description": text, "visible_text": ocr, "extracted_data": [], "importance": "medium", "requires_chart_or_table_extraction": False}


def analyze_visuals(ctx: PipelineContext, state: PipelineState) -> None:
    step(ctx, "5. Visual analysis")

    if not ctx.visual_analysis or not state.selected_frames:
        payload = {"enabled": False, "reason": "visual frame analysis disabled or no selected frames", "observations": []}
        write_json(ctx.visuals_dir / "visual_observations.json", payload)
        write_text(ctx.visuals_dir / "visual_notes.md", "# Visual Notes\n\nVisual frame analysis was disabled or no frames were selected.")
        state.visual_observations = []
        print("Skipped.")
        return

    observations = []
    used_method = "vision_model" if OPENAI_API_KEY else "tesseract_ocr_only"

    for frame in state.selected_frames:
        image_path = ctx.root_dir / frame["selected_file"]
        ocr = ocr_text(image_path)
        vision = analyze_frame_with_vision(ctx, frame, image_path, ocr) if OPENAI_API_KEY else None

        if vision is None:
            vision = {
                "type": "other",
                "description": "OCR-only visual observation. No vision model was available.",
                "visible_text": ocr,
                "extracted_data": [],
                "importance": "medium" if ocr else "low",
                "requires_chart_or_table_extraction": False,
            }

        observations.append(
            {
                "selected_id": frame["selected_id"],
                "frame_id": frame["frame_id"],
                "timestamp": frame["timestamp"],
                "timestamp_seconds": frame["timestamp_seconds"],
                "image": frame["selected_file"],
                "type": vision.get("type", "other"),
                "ocr_text": ocr,
                "visible_text": vision.get("visible_text", ""),
                "description": vision.get("description", ""),
                "extracted_data": vision.get("extracted_data", []),
                "importance": vision.get("importance", "medium"),
                "requires_chart_or_table_extraction": bool(vision.get("requires_chart_or_table_extraction", False)),
            }
        )

    notes = ["# Visual Notes", ""]
    for obs in observations:
        notes += [
            f"## {obs['timestamp']} — {obs['selected_id']}",
            "",
            f"Image: `{obs['image']}`",
            f"Type: {obs['type']}",
            f"Importance: {obs['importance']}",
            "",
            obs.get("description") or "No description.",
            "",
        ]
        if obs.get("visible_text"):
            notes += ["Visible text:", "", "```text", obs["visible_text"][:3000], "```", ""]
        if obs.get("extracted_data"):
            notes += ["Extracted data:", "", "```json", json.dumps(obs["extracted_data"], ensure_ascii=False, indent=2), "```", ""]

    write_json(ctx.visuals_dir / "visual_observations.json", {"enabled": True, "method": used_method, "observations": observations})
    write_text(ctx.visuals_dir / "visual_notes.md", "\n".join(notes))
    state.visual_observations = observations
    print(f"Visual observations: {len(observations)} using {used_method}")


# -----------------------------------------------------------------------------
# Stage 6: Chart / table extraction
# -----------------------------------------------------------------------------


def try_docling_image_markdown(ctx: PipelineContext, image_path: Path) -> str | None:
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(str(image_path))
        return result.document.export_to_markdown()
    except Exception as e:
        append_log(ctx, f"Docling image conversion failed for {image_path.name}: {e}")
        return None


def extract_charts_and_tables(ctx: PipelineContext, state: PipelineState) -> None:
    step(ctx, "6. Chart / table extraction")

    if not ctx.visual_analysis or not state.visual_observations:
        payload = {"enabled": False, "reason": "no visual observations", "tables": [], "charts": []}
        write_json(ctx.visuals_dir / "extracted_tables.json", payload)
        write_text(ctx.visuals_dir / "extracted_charts.md", "# Extracted Charts and Tables\n\nNo visual observations available.")
        print("Skipped.")
        return

    relevant = [
        obs for obs in state.visual_observations
        if obs.get("type") in {"chart", "table"} or obs.get("requires_chart_or_table_extraction")
    ]

    extracted_items = []
    md_lines = ["# Extracted Charts and Tables", ""]

    for obs in relevant:
        image_path = ctx.root_dir / obs["image"]
        docling_md = try_docling_image_markdown(ctx, image_path)
        method = "docling" if docling_md else "vision_description_fallback"
        content_md = docling_md or obs.get("description") or "No structured extraction available."

        item = {
            "timestamp": obs["timestamp"],
            "image": obs["image"],
            "type": obs.get("type"),
            "method": method,
            "extracted_data": obs.get("extracted_data", []),
            "markdown": content_md,
        }
        extracted_items.append(item)

        md_lines += [
            f"## {obs['timestamp']} — {obs.get('type', 'visual')}",
            "",
            f"Image: `{obs['image']}`",
            f"Method: {method}",
            "",
            content_md,
            "",
        ]

    if not relevant:
        md_lines.append("No chart/table frames were detected.")

    payload = {"enabled": True, "items": extracted_items}
    write_json(ctx.visuals_dir / "extracted_tables.json", payload)
    write_text(ctx.visuals_dir / "extracted_charts.md", "\n".join(md_lines))
    print(f"Chart/table candidates processed: {len(relevant)}")


# -----------------------------------------------------------------------------
# Stage 7: Timeline merge
# -----------------------------------------------------------------------------


def chapter_for_time(metadata: dict[str, Any], seconds: float) -> str | None:
    for chapter in metadata.get("chapters") or []:
        start = chapter.get("start_time")
        end = chapter.get("end_time")
        if start is None:
            continue
        if end is None:
            end = float("inf")
        if float(start) <= seconds < float(end):
            return chapter.get("title")
    return None


def merge_timeline(ctx: PipelineContext, state: PipelineState) -> None:
    step(ctx, "7. Timeline merge")

    metadata = state.metadata or read_json(ctx.raw_dir / "metadata.json", {})
    segments = state.transcript_segments or read_json(ctx.transcript_dir / "transcript_raw.json", []) or []
    observations = state.visual_observations or read_json(ctx.visuals_dir / "visual_observations.json", {}).get("observations", []) or []

    items = []
    for s in segments:
        sec = float(s.get("start", 0.0))
        items.append(
            {
                "timestamp_seconds": sec,
                "timestamp": seconds_to_ts(sec),
                "kind": "transcript",
                "chapter": chapter_for_time(metadata, sec),
                "text": s.get("text", ""),
            }
        )

    for obs in observations:
        sec = float(obs.get("timestamp_seconds", 0.0))
        items.append(
            {
                "timestamp_seconds": sec,
                "timestamp": obs.get("timestamp") or seconds_to_ts(sec),
                "kind": "visual",
                "chapter": chapter_for_time(metadata, sec),
                "image": obs.get("image"),
                "visual_type": obs.get("type"),
                "importance": obs.get("importance"),
                "text": obs.get("description", ""),
                "visible_text": obs.get("visible_text", ""),
            }
        )

    items.sort(key=lambda x: (x["timestamp_seconds"], 0 if x["kind"] == "visual" else 1))
    write_json(ctx.combined_dir / "timeline.json", {"items": items})

    md = ["# Combined Timeline", ""]
    last_chapter = None
    for item in items:
        chapter = item.get("chapter") or "General"
        if chapter != last_chapter:
            md += [f"## {chapter}", ""]
            last_chapter = chapter

        if item["kind"] == "visual":
            md += [
                f"### {item['timestamp']} — Visual: {item.get('visual_type', 'other')}",
                "",
                f"Image: `{item.get('image')}`",
                "",
                item.get("text") or "No visual description.",
                "",
            ]
            if item.get("visible_text"):
                md += ["Visible text:", "", "```text", item["visible_text"][:1500], "```", ""]
        else:
            md += [f"- [{item['timestamp']}] {item.get('text', '')}"]

    write_text(ctx.combined_dir / "combined_timeline.md", "\n".join(md))
    print(f"Timeline items: {len(items)}")


# -----------------------------------------------------------------------------
# Stage 8: Markdown composition
# -----------------------------------------------------------------------------


def compose_with_llm(ctx: PipelineContext) -> str | None:
    if not OPENAI_API_KEY:
        return None

    metadata = read_json(ctx.raw_dir / "metadata.json", {}) or {}
    transcript_md = (ctx.transcript_dir / "transcript_clean.md").read_text(encoding="utf-8", errors="ignore") if (ctx.transcript_dir / "transcript_clean.md").exists() else ""
    timeline_md = (ctx.combined_dir / "combined_timeline.md").read_text(encoding="utf-8", errors="ignore") if (ctx.combined_dir / "combined_timeline.md").exists() else ""
    visual_notes_md = (ctx.visuals_dir / "visual_notes.md").read_text(encoding="utf-8", errors="ignore") if (ctx.visuals_dir / "visual_notes.md").exists() else ""
    charts_md = (ctx.visuals_dir / "extracted_charts.md").read_text(encoding="utf-8", errors="ignore") if (ctx.visuals_dir / "extracted_charts.md").exists() else ""

    prompt = f"""Create a useful Markdown note from this YouTube video.

Do not create a skill package. The output is a normal Markdown document.

Requirements:
- Start with title, source URL, channel, upload date if available.
- Include a short summary.
- Include chapter-based or topic-based notes.
- Include important visual moments if available.
- Include extracted charts/tables if available.
- Include concise key takeaways.
- Do not include the full transcript unless it is very short; instead reference the transcript artifact.
- Keep timestamps where they help.

Metadata:
{json.dumps(metadata, ensure_ascii=False, indent=2)[:5000]}

Combined timeline:
{limit_text(timeline_md, 45000)}

Visual notes:
{limit_text(visual_notes_md, 20000)}

Extracted charts/tables:
{limit_text(charts_md, 15000)}

Transcript excerpt:
{limit_text(transcript_md, 20000)}
"""

    payload = {
        "model": OPENAI_TEXT_MODEL,
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "temperature": 0.2,
    }
    resp = call_openai_responses(payload, timeout=180)
    text = extract_response_text(resp)
    return text or None


def compose_template(ctx: PipelineContext) -> str:
    metadata = read_json(ctx.raw_dir / "metadata.json", {}) or {}
    title = metadata.get("title") or "YouTube Video Notes"
    channel = metadata.get("channel") or "Unknown"
    url = metadata.get("webpage_url") or ctx.url
    upload_date = metadata.get("upload_date") or "Unknown"

    lines = [
        f"# {title}",
        "",
        f"Source: {url}",
        f"Channel: {channel}",
        f"Upload date: {upload_date}",
        f"Created: {ctx.created_at}",
        "",
        "## Summary",
        "",
        "TODO: Generate summary from transcript and visual timeline.",
        "",
        "## Main Notes",
        "",
        "TODO: Compose chapter-based or topic-based notes.",
        "",
        "## Timeline",
        "",
        "See: `../combined/combined_timeline.md`",
        "",
        "## Transcript",
        "",
        "See: `../transcript/transcript_clean.md`",
    ]

    if ctx.visual_analysis:
        lines += [
            "",
            "## Visual Notes",
            "",
            "See: `../visuals/visual_notes.md`",
            "",
            "## Extracted Charts and Tables",
            "",
            "See: `../visuals/extracted_charts.md`",
        ]

    lines += [
        "",
        "## Key Takeaways",
        "",
        "TODO: Generate key takeaways.",
    ]
    return "\n".join(lines)


def compose_markdown(ctx: PipelineContext, state: PipelineState) -> None:
    step(ctx, "8. Markdown composition")

    md = compose_with_llm(ctx)
    method = "llm"
    if not md:
        md = compose_template(ctx)
        method = "deterministic_template"

    write_text(ctx.output_dir / "video_notes.md", md)
    write_json(ctx.output_dir / "composition_report.json", {"method": method, "output": "video_notes.md"})
    print(f"Created: {ctx.output_dir / 'video_notes.md'} using {method}")


# -----------------------------------------------------------------------------
# Final report
# -----------------------------------------------------------------------------


def write_run_report(ctx: PipelineContext, state: PipelineState, status: str = "complete") -> None:
    report = {
        "created_at": ctx.created_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "url": ctx.url,
        "video_id": ctx.video_id,
        "visual_analysis": ctx.visual_analysis,
        "status": status,
        "main_output": "output/video_notes.md",
        "artifacts": {
            "metadata": "raw/metadata.json",
            "captions": "raw/captions.vtt" if (ctx.raw_dir / "captions.vtt").exists() else None,
            "video": str(state.video_path.relative_to(ctx.root_dir)) if state.video_path and state.video_path.exists() else None,
            "audio": str(state.audio_path.relative_to(ctx.root_dir)) if state.audio_path and state.audio_path.exists() else None,
            "transcript": "transcript/transcript_clean.md",
            "frame_manifest": "visuals/frame_manifest.json",
            "selected_frames": "visuals/selected_frames.json",
            "visual_notes": "visuals/visual_notes.md",
            "charts_tables": "visuals/extracted_charts.md",
            "timeline": "combined/combined_timeline.md",
        },
        "counts": {
            "transcript_segments": len(state.transcript_segments or []),
            "frames": len(state.frames or []),
            "selected_frames": len(state.selected_frames or []),
            "visual_observations": len(state.visual_observations or []),
        },
    }
    write_json(ctx.logs_dir / "run_report.json", report)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a YouTube video into a Markdown notes bundle.")
    parser.add_argument("url", nargs="?", help="YouTube URL or video ID.")
    parser.add_argument(
        "--visual-analysis",
        choices=["yes", "no"],
        help="Whether to include visual frame analysis. If omitted, you will be asked.",
    )
    parser.add_argument("--out-dir", type=Path, default=None, help="Base output directory. Defaults to current directory.")
    parser.add_argument("--langs", default=DEFAULT_LANGS, help="Comma-separated caption language preferences. Default: en,de,tr")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Maximum number of selected frames.")
    parser.add_argument("--scene-threshold", type=float, default=DEFAULT_SCENE_THRESHOLD, help="PySceneDetect content threshold.")
    parser.add_argument("--video-height", type=int, default=DEFAULT_VIDEO_HEIGHT, help="Maximum downloaded video height.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        ctx = create_context(args)
    except Exception as e:
        print(f"Input error: {e}")
        return 2

    state = PipelineState()

    try:
        initialize_output_tree(ctx)
        acquire_source(ctx, state)
        extract_transcript(ctx, state)
        extract_keyframes(ctx, state)
        filter_frames(ctx, state)
        analyze_visuals(ctx, state)
        extract_charts_and_tables(ctx, state)
        merge_timeline(ctx, state)
        compose_markdown(ctx, state)
        write_run_report(ctx, state, status="complete")
    except KeyboardInterrupt:
        write_run_report(ctx, state, status="interrupted")
        print("Interrupted.")
        return 130
    except Exception as e:
        append_log(ctx, f"ERROR: {e}")
        write_run_report(ctx, state, status="failed")
        print(f"Error: {e}")
        print(f"See log: {ctx.logs_dir / 'pipeline.log'}")
        return 1

    print("\nDone.")
    print(f"Main output: {ctx.output_dir / 'video_notes.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
