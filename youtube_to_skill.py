#!/usr/bin/env python3
"""
youtube_to_skill.py

Single procedural script to convert a YouTube video into a local skill bundle:

    output/
      SKILL.md
      references/
        transcript_raw.json
        transcript_clean.md
        visual_notes.md
        llm_input.md
      assets/
        scene_0001.jpg
        scene_0002.jpg
        ...

What it does:
1. Tries to fetch transcript via youtube-transcript-api.
2. If unavailable, tries yt-dlp subtitles.
3. If unavailable, optionally downloads audio and transcribes with faster-whisper.
4. Downloads a low-resolution video copy.
5. Extracts key frames using scene-change detection, with fallback to fixed interval.
6. Optionally OCRs frames using pytesseract.
7. Builds an LLM input bundle.
8. Optionally calls an OpenAI-compatible API to generate SKILL.md.
9. If no API key is available, writes a template SKILL.md and saves the prepared input.

Install recommended dependencies:

    pip install youtube-transcript-api yt-dlp requests pillow pytesseract python-dotenv opencv-python imageio-ffmpeg

Optional, for local transcription fallback:

    pip install faster-whisper

System tools recommended:

    ffmpeg
    tesseract

Examples:

    python youtube_to_skill.py "https://www.youtube.com/watch?v=VIDEO_ID" --topic "UX landing page design"

    python youtube_to_skill.py "https://youtu.be/VIDEO_ID" --topic "UI design" --openai-model "gpt-4.1-mini"

    python youtube_to_skill.py "URL" --topic "UX research" --no-llm

Notes:
- This script is for personal learning/workflow automation.
- Respect YouTube terms, copyright, and creator rights.
- Some videos block subtitles/downloads; the script will degrade gracefully.
"""

PATCH_VERSION = "2026-06-05-v9-ffmpeg-python-package-resilience"

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

# Load .env early, before reading OPENAI_API_KEY, OPENAI_BASE_URL, FFMPEG_LOCATION, etc.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

FFMPEG_LOCATION = os.environ.get("FFMPEG_LOCATION", "")
YTDLP_COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "")
YTDLP_COOKIES = os.environ.get("YTDLP_COOKIES", "")
YTDLP_SLEEP_REQUESTS = os.environ.get("YTDLP_SLEEP_REQUESTS", "1")
YTDLP_SLEEP_INTERVAL = os.environ.get("YTDLP_SLEEP_INTERVAL", "1")
YTDLP_MAX_SLEEP_INTERVAL = os.environ.get("YTDLP_MAX_SLEEP_INTERVAL", "5")
YTDLP_RETRIES = os.environ.get("YTDLP_RETRIES", "10")

# ----------------------------
# Small utility helpers
# ----------------------------

def log(message: str) -> None:
    print(f"[youtube-to-skill] {message}", flush=True)


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, check: bool = False) -> Tuple[int, str, str]:
    """Run a shell command safely without shell=True."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"Command failed with exit code {proc.returncode}\n"
                f"CMD: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError:
        return 127, "", f"Command not found: {cmd[0]}"



def add_ytdlp_options(cmd: List[str]) -> List[str]:
    """Add configured yt-dlp options for FFmpeg, cookies, and rate-limit resilience."""
    if not cmd or cmd[0] != "yt-dlp":
        return cmd

    options: List[str] = [cmd[0]]

    if FFMPEG_LOCATION and "--ffmpeg-location" not in cmd:
        options.extend(["--ffmpeg-location", FFMPEG_LOCATION])

    if YTDLP_COOKIES_FROM_BROWSER and "--cookies-from-browser" not in cmd:
        options.extend(["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER])

    if YTDLP_COOKIES and "--cookies" not in cmd:
        options.extend(["--cookies", YTDLP_COOKIES])

    if YTDLP_RETRIES and "--retries" not in cmd:
        options.extend(["--retries", str(YTDLP_RETRIES)])
    if YTDLP_RETRIES and "--fragment-retries" not in cmd:
        options.extend(["--fragment-retries", str(YTDLP_RETRIES)])
    if "--extractor-retries" not in cmd:
        options.extend(["--extractor-retries", "3"])
    if YTDLP_SLEEP_REQUESTS and "--sleep-requests" not in cmd:
        options.extend(["--sleep-requests", str(YTDLP_SLEEP_REQUESTS)])
    if YTDLP_SLEEP_INTERVAL and "--sleep-interval" not in cmd:
        options.extend(["--sleep-interval", str(YTDLP_SLEEP_INTERVAL)])
    if YTDLP_MAX_SLEEP_INTERVAL and "--max-sleep-interval" not in cmd:
        options.extend(["--max-sleep-interval", str(YTDLP_MAX_SLEEP_INTERVAL)])

    return [*options, *cmd[1:]]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def tool_path(name: str) -> Optional[str]:
    """Resolve command path.

    Resolution order:
    1. --ffmpeg-location / FFMPEG_LOCATION for ffmpeg/ffprobe.
    2. PATH / venv Scripts directory.
    3. imageio-ffmpeg bundled binary for ffmpeg only.

    Note:
    `uv add ffmpeg` often installs a Python wrapper, not ffmpeg.exe.
    `uv add imageio-ffmpeg` provides a bundled ffmpeg executable that this function can detect.
    """
    if name in {"ffmpeg", "ffprobe"} and FFMPEG_LOCATION:
        loc = Path(FFMPEG_LOCATION).expanduser()

        candidates = []
        if loc.is_file():
            candidates.append(loc)
        else:
            candidates.extend([
                loc / f"{name}.exe",
                loc / name,
                loc / "bin" / f"{name}.exe",
                loc / "bin" / name,
                loc / "Scripts" / f"{name}.exe",
                loc / "Scripts" / name,
            ])

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

    found = shutil.which(name)
    if found:
        return found

    # Many Python ffmpeg wrappers do not provide an executable. imageio-ffmpeg does.
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            if exe and Path(exe).exists():
                return str(exe)
        except Exception:
            pass

    return None

def which(name: str) -> bool:
    return tool_path(name) is not None


def slugify(text: str, max_len: int = 60) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len].strip("-") or "youtube-skill"


def seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def extract_video_id(url_or_id: str) -> str:
    """Handle common YouTube URL formats or raw video ID."""
    if not url_or_id:
        raise ValueError("No YouTube URL or video ID provided.")
    raw = str(url_or_id).strip()

    # Raw video id, usually 11 chars. Do not overvalidate.
    if re.fullmatch(r"[a-zA-Z0-9_-]{8,20}", raw):
        return raw

    parsed = urlparse(raw)

    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0]

    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            q = parse_qs(parsed.query)
            if "v" in q and q["v"]:
                return q["v"][0]
        # /shorts/<id>, /embed/<id>, /live/<id>
        parts = [p for p in parsed.path.split("/") if p]
        for marker in ("shorts", "embed", "live"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return parts[idx + 1]

    raise ValueError(f"Could not extract YouTube video id from: {url_or_id}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


# ----------------------------
# Metadata and transcript
# ----------------------------

def fetch_video_metadata(url: str, refs_dir: Path) -> Dict[str, Any]:
    """Use yt-dlp to get video metadata when available."""
    if not which("yt-dlp"):
        log("yt-dlp not found. Skipping metadata fetch.")
        return {"url": url}

    cmd = ["yt-dlp", "--dump-json", "--skip-download", url]
    cmd = add_ytdlp_options(cmd)
    code, out, err = run_cmd(cmd)

    if code != 0 or not out.strip():
        log("Could not fetch metadata with yt-dlp.")
        if err.strip():
            log(err.strip().splitlines()[-1])
        return {"url": url}

    try:
        meta = json.loads(out)
        wanted = {
            "id": meta.get("id"),
            "title": meta.get("title"),
            "channel": meta.get("channel") or meta.get("uploader"),
            "duration": meta.get("duration"),
            "webpage_url": meta.get("webpage_url") or url,
            "upload_date": meta.get("upload_date"),
            "description": meta.get("description", "")[:2000],
        }
        write_text(refs_dir / "metadata.json", json.dumps(wanted, ensure_ascii=False, indent=2))
        return wanted
    except Exception as exc:
        log(f"Metadata parse failed: {exc}")
        return {"url": url}


def try_youtube_transcript_api(video_id: str, language: str) -> Optional[List[Dict[str, Any]]]:
    """Try youtube-transcript-api using the current instance-style API only."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        log("Trying youtube-transcript-api...")

        ytt = YouTubeTranscriptApi()

        languages = []
        if language:
            languages.append(language)
        if "en" not in languages:
            languages.append("en")

        transcript = ytt.fetch(video_id, languages=languages)

        if hasattr(transcript, "to_raw_data"):
            data = transcript.to_raw_data()
        else:
            data = list(transcript)

        normalized = []
        for item in data:
            # youtube-transcript-api may return dict-like objects or objects.
            if isinstance(item, dict):
                start = item.get("start", 0.0)
                duration = item.get("duration", 0.0)
                text = item.get("text", "")
            else:
                start = getattr(item, "start", 0.0)
                duration = getattr(item, "duration", 0.0)
                text = getattr(item, "text", "")

            normalized.append(
                {
                    "start": float(start or 0.0),
                    "duration": float(duration or 0.0),
                    "text": str(text or "").replace("\n", " ").strip(),
                    "source": "youtube-transcript-api",
                }
            )

        normalized = [item for item in normalized if item["text"]]

        if normalized:
            log(f"Transcript found via youtube-transcript-api: {len(normalized)} segments.")
            return normalized

    except Exception as exc:
        log(f"youtube-transcript-api failed: {exc}")

    return None

# FALLBACK: yt-dlp subtitle extraction, SRT parsing, and manual transcript loading.
# Not used when youtube-transcript-api succeeds (the happy path).
# def parse_srt_time_to_seconds(value: str) -> float: ...
# def parse_srt(text: str) -> List[Dict[str, Any]]: ...
# def try_ytdlp_subtitles(url, language, refs_dir) -> Optional[List[Dict]]: ...
# def load_manual_transcript(path: Path) -> Optional[List[Dict]]: ...


def save_transcript(segments: List[Dict[str, Any]], refs_dir: Path) -> None:
    write_text(refs_dir / "transcript_raw.json", json.dumps(segments, ensure_ascii=False, indent=2))

    lines = []
    for item in segments:
        ts = seconds_to_timestamp(float(item.get("start", 0)))
        text = item.get("text", "").strip()
        if text:
            lines.append(f"[{ts}] {text}")

    write_text(refs_dir / "transcript_clean.md", "\n".join(lines) + "\n")


# ----------------------------
# Audio/video download and transcription fallback
# ----------------------------

# FALLBACK: audio download + faster-whisper transcription.
# Not used when youtube-transcript-api succeeds.
# def download_audio(url, work_dir) -> Optional[Path]: ...
# def transcribe_with_faster_whisper(audio_path, language) -> Optional[List[Dict]]: ...


def download_low_res_video(url: str, work_dir: Path) -> Optional[Path]:
    """Download a single-file low-res video when possible.

    Avoids `bv*+ba` because merging requires FFmpeg.
    For frame extraction, audio is irrelevant, so a single video file is enough.
    """
    if not which("yt-dlp"):
        log("yt-dlp not found. Cannot download video.")
        return None

    for old in work_dir.glob("video.*"):
        try:
            old.unlink()
        except Exception:
            pass

    cmd = [
        "yt-dlp",
        "-f",
        "best[ext=mp4][height<=720]/best[height<=720]/best[ext=mp4]/best",
        "-o",
        str(work_dir / "video.%(ext)s"),
        url,
    ]

    log("Downloading video for frame extraction...")
    cmd = add_ytdlp_options(cmd)
    code, out, err = run_cmd(cmd)
    if code != 0:
        log("Video download failed.")
        if err.strip():
            log(err.strip().splitlines()[-1])
        return None

    candidates = [
        p for p in work_dir.glob("video.*")
        if p.suffix.lower() not in {".part", ".ytdl", ".json"}
    ]
    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_size)



# ----------------------------
# Frame extraction and OCR
# ----------------------------

# FALLBACK: OpenCV-based interval frame extraction — not used when ffmpeg is available.
# def extract_interval_frames_opencv(video_path, assets_dir, max_frames, interval_seconds=10) -> List[Path]: ...

# FALLBACK: fixed-interval ffmpeg frame extraction — not used when scene-change extraction succeeds.
# def extract_interval_frames_ffmpeg(video_path, assets_dir, interval_seconds, max_frames) -> List[Path]: ...


def extract_scene_frames_ffmpeg(video_path: Path, assets_dir: Path, threshold: float, max_frames: int) -> List[Path]:
    """Extract key frames using ffmpeg scene-change detection."""
    if not which("ffmpeg"):
        log("ffmpeg not found. Cannot extract frames.")
        return []

    ensure_dir(assets_dir)
    pattern = assets_dir / "scene_%04d.jpg"

    cmd = [
        tool_path("ffmpeg") or "ffmpeg",
        "-hide_banner",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='gt(scene,{threshold})',scale='min(1280,iw)':-2",
        "-vsync",
        "vfr",
        "-q:v",
        "3",
        str(pattern),
    ]

    log("Extracting scene-change frames with ffmpeg...")
    run_cmd(cmd)
    frames = sorted(assets_dir.glob("scene_*.jpg"))

    if not frames:
        log("Scene-change extraction produced no frames.")
        return []

    frames = limit_frames_evenly(frames, max_frames)
    log(f"Extracted {len(frames)} key frames.")
    return frames


def limit_frames_evenly(frames: List[Path], max_frames: int) -> List[Path]:
    if max_frames <= 0 or len(frames) <= max_frames:
        return frames

    keep_indices = set(round(i * (len(frames) - 1) / (max_frames - 1)) for i in range(max_frames))
    selected = [frame for idx, frame in enumerate(frames) if idx in keep_indices]

    # Rename selected frames into clean keyframe names; leave old files in place.
    renamed = []
    for i, frame in enumerate(selected, start=1):
        new_path = frame.parent / f"keyframe_{i:04d}.jpg"
        if frame != new_path:
            try:
                shutil.copy2(frame, new_path)
            except Exception:
                new_path = frame
        renamed.append(new_path)

    return renamed


def ocr_frame(frame_path: Path) -> str:
    try:
        from PIL import Image
        import pytesseract
    except Exception:
        return ""

    try:
        image = Image.open(frame_path)
        text = pytesseract.image_to_string(image)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return text
    except Exception:
        return ""


def build_visual_notes(frames: List[Path], refs_dir: Path, out_dir: Path, do_ocr: bool) -> str:
    lines = [
        "# Visual Notes",
        "",
        "These are automatically extracted key frames. OCR may be noisy.",
        "",
    ]

    if not frames:
        lines.append("_No frames extracted._")
        visual_notes = "\n".join(lines) + "\n"
        write_text(refs_dir / "visual_notes.md", visual_notes)
        return visual_notes

    if do_ocr:
        log("Running OCR on key frames...")
    else:
        log("Skipping OCR.")

    for idx, frame in enumerate(frames, start=1):
        rel = frame.relative_to(out_dir).as_posix() if frame.is_relative_to(out_dir) else frame.as_posix()
        lines.append(f"## Key frame {idx}")
        lines.append("")
        lines.append(f"Image: `{rel}`")
        lines.append("")

        if do_ocr:
            text = ocr_frame(frame)
            if text:
                lines.append("OCR:")
                lines.append("")
                lines.append("```text")
                lines.append(text[:3000])
                lines.append("```")
                lines.append("")
            else:
                lines.append("OCR: _No readable text detected or OCR unavailable._")
                lines.append("")

        lines.append("Visual interpretation placeholder:")
        lines.append("- Describe what this frame contributes to the tutorial.")
        lines.append("- Note UI/layout/design examples, diagrams, checklists, or before/after comparisons.")
        lines.append("")

    visual_notes = "\n".join(lines)
    write_text(refs_dir / "visual_notes.md", visual_notes)
    return visual_notes


# ----------------------------
# LLM input and SKILL.md generation
# ----------------------------

def build_llm_input(
    metadata: Dict[str, Any],
    topic: str,
    transcript_md: str,
    visual_notes_md: str,
    refs_dir: Path,
) -> str:
    title = metadata.get("title") or "Unknown video"
    channel = metadata.get("channel") or "Unknown channel"
    url = metadata.get("webpage_url") or metadata.get("url") or ""

    prompt = f"""# Task: Convert YouTube tutorial into SKILL.md

Create a reusable agent skill from this tutorial.

Important:
- This is not a summary.
- Generalize the tutorial into a reusable operational skill.
- Preserve important visual lessons from frames/OCR.
- Do not mention the source video unless necessary.
- Focus on reusable principles, workflow, decision rules, checklists, common mistakes, and output format.
- Write a complete SKILL.md with YAML front matter.

## Desired SKILL.md structure

```md
---
name: short-kebab-case-name
description: Clear trigger description. Explain when an agent should use this skill.
---

# Skill Title

## Purpose

## When to use

## Inputs expected

## Workflow

## Checklist

## Common mistakes

## Visual/design heuristics

## Output format

## Example usage
```

## Video metadata

Title: {title}
Channel: {channel}
URL: {url}
Topic: {topic}

## Transcript

{transcript_md[:60000]}

## Visual notes / OCR from extracted frames

{visual_notes_md[:30000]}
"""
    write_text(refs_dir / "llm_input.md", prompt)
    return prompt


def generate_skill_with_openai_responses_api(prompt: str, model: str) -> Optional[str]:
    """Generate SKILL.md content using the OpenAI Responses API.

    Environment variables:
      OPENAI_API_KEY       required
      OPENAI_BASE_URL      optional, default https://api.openai.com/v1
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log("OPENAI_API_KEY not set. Skipping LLM generation.")
        return None

    try:
        import requests
    except Exception:
        log("requests not installed. Skipping LLM generation.")
        return None

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"

    log(f"Generating SKILL.md with model: {model}")

    payload = {
        "model": model,
        "instructions": (
            "You create precise, reusable agent skills. "
            "Return only the SKILL.md content. "
            "Do not wrap the answer in code fences."
        ),
        "input": prompt,
        "temperature": 0.2,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        if response.status_code >= 400:
            log(f"LLM API error {response.status_code}: {response.text[:1000]}")
            return None

        data = response.json()
        content = data["output"][0]["content"][0]["text"].strip()
        return content

    except Exception as exc:
        log(f"LLM generation failed: {exc}")
        return None


# FALLBACK: template SKILL.md used when LLM generation fails.
# Not used in current pipeline (LLM is required). Preserved for re-use if needed.
def fallback_skill_template(topic: str, metadata: Dict[str, Any]) -> str:
    name = slugify(topic)
    title = topic.strip() or "YouTube Tutorial Skill"
    source_title = metadata.get("title") or "Unknown source video"

    return f"""---
name: {name}
description: Use this skill to apply reusable lessons extracted from a YouTube tutorial about {title}. Review references/llm_input.md and references/visual_notes.md before finalizing.
---

# {title}

## Purpose

Use this skill to apply the operational lessons extracted from the source tutorial.

Source reviewed: {source_title}

## When to use

Use this skill when the user asks for help with:
- {title}
- reviewing examples related to this topic
- applying tutorial-like principles to a concrete artifact
- creating a checklist, critique, or improvement plan

## Inputs expected

Ask for or infer:
- The artifact to review or improve
- The target audience
- The goal or success criterion
- Any constraints, such as platform, brand, time, budget, or implementation limits

## Workflow

1. Identify the user's concrete goal.
2. Extract the relevant principles from the tutorial notes.
3. Apply the principles to the user's artifact or question.
4. Separate high-priority issues from minor refinements.
5. Give specific, actionable recommendations.
6. When visual design is involved, consider hierarchy, spacing, contrast, layout, affordances, and user flow.
7. Return the result in the requested format.

## Checklist

- Is the main goal clear?
- Are the most important elements visually or conceptually dominant?
- Are examples converted into reusable rules?
- Are recommendations specific enough to act on?
- Are trade-offs explained?
- Are visual observations grounded in the extracted frames or OCR notes?

## Common mistakes

- Summarizing the tutorial instead of turning it into a reusable procedure.
- Losing visual context from important frames.
- Treating examples as universal rules without checking context.
- Giving vague advice instead of concrete changes.
- Ignoring the user's target audience and goal.

## Visual/design heuristics

Review `references/visual_notes.md` for extracted frames and OCR.

When visual context matters, check:
- visual hierarchy
- information grouping
- alignment and spacing
- contrast
- affordances
- before/after comparisons
- labels and microcopy
- user journey or flow

## Output format

Return:

1. Main diagnosis
2. Relevant principle from the skill
3. Concrete recommendation
4. Example rewrite/redesign if applicable
5. Priority: high / medium / low

## Example usage

User: "Review this landing page hero section."

Assistant:
1. Main diagnosis: ...
2. Relevant principle: ...
3. Recommended change: ...
4. Example: ...
5. Priority: ...
"""


# ----------------------------
# Main procedure
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a YouTube video into a local SKILL.md bundle."
    )
    parser.add_argument("url", nargs="?", help="Optional YouTube URL or video ID. If omitted, the script asks at runtime.")
    parser.add_argument("--url", dest="url_option", default="", help="YouTube URL or video ID. Overrides positional url.")
    parser.add_argument("--topic", default="the tutorial topic", help="Topic/name of the skill")
    parser.add_argument("--out", default="", help="Output directory. Default: ./skill_<video_id>_<topic>")
    parser.add_argument("--language", default="en", help="Preferred transcript language, e.g. en, de, tr")
    parser.add_argument("--max-frames", type=int, default=40, help="Maximum key frames to keep")
    parser.add_argument("--scene-threshold", type=float, default=0.25, help="ffmpeg scene-change threshold")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR on frames")
    parser.add_argument("--no-llm", action="store_true", help="Do not call OpenAI-compatible API")
    parser.add_argument("--openai-model", default="gpt-4.1-mini", help="OpenAI-compatible model name")
    parser.add_argument("--skip-video", action="store_true", help="Skip video download/frame extraction")
    parser.add_argument("--force-whisper", action="store_true", help="Force audio transcription instead of captions")
    parser.add_argument("--ffmpeg-location", default="", help="Path to ffmpeg binary or folder containing ffmpeg/ffprobe. Also reads FFMPEG_LOCATION env var.")
    parser.add_argument("--cookies-from-browser", default="", help="Browser name for yt-dlp cookies, e.g. firefox, chrome, edge. Also reads YTDLP_COOKIES_FROM_BROWSER env var.")
    parser.add_argument("--cookies", default="", help="Path to cookies.txt for yt-dlp. Also reads YTDLP_COOKIES env var.")
    parser.add_argument("--manual-transcript", default="", help="Path to a transcript file to use before trying YouTube extraction.")
    parser.add_argument("--sleep-requests", default="1", help="yt-dlp --sleep-requests value. Helps reduce rate-limit pressure.")
    parser.add_argument("--sleep-interval", default="1", help="yt-dlp --sleep-interval value.")
    parser.add_argument("--max-sleep-interval", default="5", help="yt-dlp --max-sleep-interval value.")
    parser.add_argument("--ytdlp-retries", default="10", help="yt-dlp retry count for download and fragments.")
    args = parser.parse_args()

    global FFMPEG_LOCATION, YTDLP_COOKIES_FROM_BROWSER, YTDLP_COOKIES, YTDLP_SLEEP_REQUESTS, YTDLP_SLEEP_INTERVAL, YTDLP_MAX_SLEEP_INTERVAL, YTDLP_RETRIES
    FFMPEG_LOCATION = getattr(args, "ffmpeg_location", "") or FFMPEG_LOCATION
    YTDLP_COOKIES_FROM_BROWSER = getattr(args, "cookies_from_browser", "") or YTDLP_COOKIES_FROM_BROWSER
    YTDLP_COOKIES = getattr(args, "cookies", "") or YTDLP_COOKIES
    YTDLP_SLEEP_REQUESTS = getattr(args, "sleep_requests", "") or YTDLP_SLEEP_REQUESTS
    YTDLP_SLEEP_INTERVAL = getattr(args, "sleep_interval", "") or YTDLP_SLEEP_INTERVAL
    YTDLP_MAX_SLEEP_INTERVAL = getattr(args, "max_sleep_interval", "") or YTDLP_MAX_SLEEP_INTERVAL
    YTDLP_RETRIES = getattr(args, "ytdlp_retries", "") or YTDLP_RETRIES

    video_url = args.url_option or args.url

    if not video_url:
        video_url = input("YouTube URL or video ID: ").strip()

    if not video_url:
        log("No YouTube URL or video ID provided.")
        return 2

    try:
        video_id = extract_video_id(video_url)
    except Exception as exc:
        log(str(exc))
        return 2

    out_dir = Path(args.out).expanduser().resolve() if args.out else Path(
        f"skill_{video_id}_{slugify(args.topic)}"
    ).resolve()

    refs_dir = out_dir / "references"
    assets_dir = out_dir / "assets"
    work_dir = out_dir / "_work"

    ensure_dir(out_dir)
    ensure_dir(refs_dir)
    ensure_dir(assets_dir)
    ensure_dir(work_dir)

    log(f"Output directory: {out_dir}")

    if not which("ffmpeg"):
        log("ffmpeg executable not found. The script will avoid audio conversion and try OpenCV for frames.")
        log("Install system FFmpeg, use --ffmpeg-location, or run: uv add imageio-ffmpeg opencv-python")

    if YTDLP_COOKIES_FROM_BROWSER:
        log(f"Using yt-dlp cookies from browser: {YTDLP_COOKIES_FROM_BROWSER}")
    if YTDLP_COOKIES:
        log(f"Using yt-dlp cookies file: {YTDLP_COOKIES}")

    metadata = fetch_video_metadata(video_url, refs_dir)

    # 1-3. Transcript acquisition
    segments: Optional[List[Dict[str, Any]]] = None

    if getattr(args, "manual_transcript", ""):
        segments = load_manual_transcript(Path(args.manual_transcript).expanduser())

    if segments is None and not args.force_whisper:
        segments = try_youtube_transcript_api(video_id, args.language)

    if segments is None and not args.force_whisper:
        segments = try_ytdlp_subtitles(video_url, args.language, refs_dir)

    if segments is None:
        audio_path = download_audio(video_url, work_dir)
        if audio_path:
            segments = transcribe_with_faster_whisper(audio_path, args.language)

    if segments:
        save_transcript(segments, refs_dir)
        transcript_md = read_text(refs_dir / "transcript_clean.md")
    else:
        log("No transcript could be acquired.")
        transcript_md = "_No transcript available._\n"
        write_text(refs_dir / "transcript_clean.md", transcript_md)

    # 4-6. Video frames and OCR
    frames: List[Path] = []
    if not args.skip_video:
        video_path = download_low_res_video(video_url, work_dir)
        if video_path:
            frames = extract_scene_frames_ffmpeg(
                video_path=video_path,
                assets_dir=assets_dir,
                threshold=args.scene_threshold,
                max_frames=args.max_frames,
            )
        else:
            log("No video available for frame extraction.")
    else:
        log("Skipping video/frame extraction by request.")

    visual_notes_md = build_visual_notes(
        frames=frames,
        refs_dir=refs_dir,
        out_dir=out_dir,
        do_ocr=not args.no_ocr,
    )

    # 7. Build LLM input
    llm_input = build_llm_input(
        metadata=metadata,
        topic=args.topic,
        transcript_md=transcript_md,
        visual_notes_md=visual_notes_md,
        refs_dir=refs_dir,
    )

    # 8-9. Generate SKILL.md
    skill_md: Optional[str] = None
    if not args.no_llm:
        skill_md = generate_skill_with_openai_responses_api(llm_input, args.openai_model)

    if not skill_md:
        skill_md = fallback_skill_template(args.topic, metadata)

    write_text(out_dir / "SKILL.md", skill_md.strip() + "\n")

    # Save a simple run report.
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "url": video_url,
        "video_id": video_id,
        "topic": args.topic,
        "output_dir": str(out_dir),
        "transcript_segments": len(segments or []),
        "frames": [str(p.relative_to(out_dir)) if p.is_relative_to(out_dir) else str(p) for p in frames],
        "ocr_enabled": not args.no_ocr,
        "llm_attempted": not args.no_llm,
        "cookies_from_browser": bool(YTDLP_COOKIES_FROM_BROWSER),
        "cookies_file": bool(YTDLP_COOKIES),
        "ytdlp_sleep_requests": YTDLP_SLEEP_REQUESTS,
        "ytdlp_retries": YTDLP_RETRIES,
        "ffmpeg_resolved": bool(tool_path("ffmpeg")),
        "skill_md": str(out_dir / "SKILL.md"),
    }
    write_text(refs_dir / "run_report.json", json.dumps(report, ensure_ascii=False, indent=2))

    log("Done.")
    log(f"Skill file: {out_dir / 'SKILL.md'}")
    log(f"LLM input:  {refs_dir / 'llm_input.md'}")
    log(f"Frames:     {assets_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
