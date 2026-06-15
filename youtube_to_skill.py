#!/usr/bin/env python3
"""youtube_to_skill.py — Convert a YouTube video into a local SKILL.md bundle."""

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

import config

load_dotenv()

YTDLP_COOKIES_FROM_BROWSER = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "")
YTDLP_COOKIES = os.environ.get("YTDLP_COOKIES", "")
YTDLP_RETRIES = os.environ.get("YTDLP_RETRIES", "10")
YTDLP_SLEEP_INTERVAL = os.environ.get("YTDLP_SLEEP_INTERVAL", "1")

_ffmpeg_loc = os.environ.get("FFMPEG_LOCATION", "")
if _ffmpeg_loc:
    FFMPEG_EXE = str(Path(_ffmpeg_loc)) if Path(_ffmpeg_loc).is_file() else str(Path(_ffmpeg_loc) / "ffmpeg.exe")
elif exe := shutil.which("ffmpeg"):
    FFMPEG_EXE = exe
else:
    try:
        import imageio_ffmpeg
        FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        FFMPEG_EXE = "ffmpeg"  # will fail at runtime if not found


def ytdlp_flags() -> list[str]:
    """Common yt-dlp flags built from env config."""
    flags = ["--retries", YTDLP_RETRIES, "--sleep-interval", YTDLP_SLEEP_INTERVAL]
    if YTDLP_COOKIES_FROM_BROWSER:
        flags += ["--cookies-from-browser", YTDLP_COOKIES_FROM_BROWSER]
    if YTDLP_COOKIES:
        flags += ["--cookies", YTDLP_COOKIES]
    return flags


def slugify(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^a-z0-9]+", "-", text.lower().strip())
    return re.sub(r"-+", "-", text).strip("-")[:max_len] or "youtube-skill"


def extract_video_id(url_or_id: str) -> str:
    raw = url_or_id.strip()
    if re.fullmatch(r"[a-zA-Z0-9_-]{8,20}", raw):
        return raw
    parsed = urlparse(raw)
    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/").split("/")[0]
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
    raise ValueError(f"Could not extract video ID from: {url_or_id}")


def fetch_metadata(url: str, refs_dir: Path) -> dict:
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--skip-download", *ytdlp_flags(), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        print("Could not fetch metadata.")
        return {"url": url}
    meta = json.loads(result.stdout)
    chapters = [
        {"start_time": c.get("start_time"), "end_time": c.get("end_time"), "title": c.get("title")}
        for c in (meta.get("chapters") or [])
    ]
    wanted = {
        "id": meta.get("id"),
        "title": meta.get("title"),
        "channel": meta.get("channel") or meta.get("uploader"),
        "duration": meta.get("duration"),
        "webpage_url": meta.get("webpage_url") or url,
        "upload_date": meta.get("upload_date"),
        "description": meta.get("description", "")[:2000],
        "chapters": chapters,
    }
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "metadata.json").write_text(json.dumps(wanted, ensure_ascii=False, indent=2), encoding="utf-8")
    return wanted


def fetch_transcript(video_id: str) -> list[dict] | None:
    from youtube_transcript_api import YouTubeTranscriptApi
    print("Fetching transcript...")
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=[config.LANGUAGE, "en"])
        data = transcript.to_raw_data() if hasattr(transcript, "to_raw_data") else list(transcript)
        segments = []
        for item in data:
            if isinstance(item, dict):
                start, duration, text = item.get("start", 0.0), item.get("duration", 0.0), item.get("text", "")
            else:
                start, duration, text = getattr(item, "start", 0.0), getattr(item, "duration", 0.0), getattr(item, "text", "")
            if text := str(text).replace("\n", " ").strip():
                segments.append({"start": float(start), "duration": float(duration), "text": text})
        print(f"Transcript: {len(segments)} segments.")
        return segments or None
    except Exception as e:
        print(f"Transcript fetch failed: {e}")
        return None


def save_transcript(segments: list[dict], refs_dir: Path) -> str:
    def ts(s):
        s = max(0, int(s))
        h, m, sec = s // 3600, (s % 3600) // 60, s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "transcript_raw.json").write_text(json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8")
    text = "\n".join(f"[{ts(s['start'])}] {s['text']}" for s in segments if s["text"]) + "\n"
    (refs_dir / "transcript_clean.md").write_text(text, encoding="utf-8")
    return text


def download_video(url: str, work_dir: Path) -> Path | None:
    work_dir.mkdir(parents=True, exist_ok=True)
    for old in work_dir.glob("video.*"):
        old.unlink(missing_ok=True)
    print("Downloading video...")
    result = subprocess.run(
        ["yt-dlp", "-f", "best[ext=mp4][height<=720]/best[height<=720]/best",
         "-o", str(work_dir / "video.%(ext)s"), *ytdlp_flags(), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("Video download failed.")
        return None
    candidates = [p for p in work_dir.glob("video.*") if p.suffix.lower() not in {".part", ".ytdl", ".json"}]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def extract_frames(video_path: Path, assets_dir: Path) -> list[Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    print("Extracting scene-change frames...")
    subprocess.run(
        [FFMPEG_EXE, "-hide_banner", "-y", "-i", str(video_path),
         "-vf", f"select='gt(scene,{config.SCENE_THRESHOLD})',scale='min(1280,iw)':-2",
         "-vsync", "vfr", "-q:v", "3", str(assets_dir / "scene_%04d.jpg")],
        capture_output=True,
    )
    frames = sorted(assets_dir.glob("scene_*.jpg"))
    if not frames:
        print("No frames extracted.")
        return []

    n = config.MAX_FRAMES
    if n > 0 and len(frames) > n:
        indices = {round(i * (len(frames) - 1) / (n - 1)) for i in range(n)}
        selected = [f for i, f in enumerate(frames) if i in indices]
        frames = []
        for i, f in enumerate(selected, 1):
            dst = f.parent / f"keyframe_{i:04d}.jpg"
            shutil.copy2(f, dst)
            frames.append(dst)

    print(f"Extracted {len(frames)} frames.")
    return frames


def build_visual_notes(frames: list[Path], refs_dir: Path, out_dir: Path) -> str:
    lines = ["# Visual Notes", "", "Automatically extracted key frames.", ""]

    if not frames:
        lines.append("_No frames extracted._")
    else:
        if config.OCR_ENABLED:
            print("Running OCR on frames...")
            from PIL import Image
            import pytesseract

        for i, frame in enumerate(frames, 1):
            rel = frame.relative_to(out_dir).as_posix() if frame.is_relative_to(out_dir) else frame.as_posix()
            lines += [f"## Frame {i}", "", f"Image: `{rel}`", ""]
            if config.OCR_ENABLED:
                try:
                    text = re.sub(r"\n{3,}", "\n\n", pytesseract.image_to_string(Image.open(frame))).strip()
                    lines += (["OCR:", "", "```text", text[:3000], "```", ""] if text else ["OCR: _No text detected._", ""])
                except Exception:
                    lines += ["OCR: _Failed._", ""]

    content = "\n".join(lines)
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "visual_notes.md").write_text(content, encoding="utf-8")
    return content


# JSON schema for structured output from the Responses API.
# Each chapter maps to a section in the Knowledge block.
# Empty category lists are omitted when rendering.
SKILL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "description", "summary", "knowledge"],
    "properties": {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "summary": {"type": "string"},
        "knowledge": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["chapter_title", "facts", "rules", "exceptions",
                             "decision_logic", "heuristics", "lessons_learned",
                             "suggestions", "warnings", "anti_patterns", "examples"],
                "properties": {
                    "chapter_title": {"type": "string"},
                    "facts":          {"type": "array", "items": {"type": "string"}},
                    "rules":          {"type": "array", "items": {"type": "string"}},
                    "exceptions":     {"type": "array", "items": {"type": "string"}},
                    "decision_logic": {"type": "array", "items": {"type": "string"}},
                    "heuristics":     {"type": "array", "items": {"type": "string"}},
                    "lessons_learned":{"type": "array", "items": {"type": "string"}},
                    "suggestions":    {"type": "array", "items": {"type": "string"}},
                    "warnings":       {"type": "array", "items": {"type": "string"}},
                    "anti_patterns":  {"type": "array", "items": {"type": "string"}},
                    "examples":       {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

CATEGORY_LABELS = {
    "facts": "Facts",
    "rules": "Rules",
    "exceptions": "Exceptions",
    "decision_logic": "Decision Logic",
    "heuristics": "Heuristics",
    "lessons_learned": "Lessons Learned",
    "suggestions": "Suggestions",
    "warnings": "Warnings",
    "anti_patterns": "Anti-Patterns",
    "examples": "Examples",
}


def build_llm_input(metadata: dict, transcript_md: str, visual_notes_md: str, refs_dir: Path) -> str:
    chapters = metadata.get("chapters") or []
    chapter_hint = (
        "The video has the following chapters:\n"
        + "\n".join(f"- {c['title']} ({c['start_time']:.0f}s – {c['end_time']:.0f}s)" for c in chapters)
        + "\n\nOrganize the knowledge section by these exact chapter titles."
        if chapters
        else "The video has no chapters. Use a single knowledge section with chapter_title = 'General'."
    )

    prompt = f"""You are converting a YouTube tutorial into a structured skill document.

Return a JSON object conforming exactly to the schema you were given. Do not add any text outside the JSON.

Rules:
- `name`: short kebab-case identifier for the skill topic.
- `description`: one sentence explaining when an agent should invoke this skill.
- `summary`: 3–5 sentences giving a concise overview of the video — its main ideas and what it covers.
- `knowledge`: list of chapter objects. Each chapter has a `chapter_title` and category arrays.
  Populate only the categories that have meaningful content; leave others as empty arrays.
  Distill the transcript into precise, atomic bullet points — not summaries.
  {chapter_hint}

Knowledge category definitions:
- facts: objective statements, definitions, measurements
- rules: prescriptive dos and don'ts
- exceptions: cases where rules do not apply
- decision_logic: if/when/then reasoning or comparison frameworks
- heuristics: practical rules of thumb, patterns, shortcuts
- lessons_learned: takeaways derived from examples or mistakes shown
- suggestions: optional improvements or ideas worth trying
- warnings: common pitfalls, things that break, failure modes
- anti_patterns: things explicitly called out as wrong or harmful
- examples: concrete cases, before/after comparisons, sample values

## Video metadata

Title: {metadata.get("title") or "Unknown"}
Channel: {metadata.get("channel") or "Unknown"}
URL: {metadata.get("webpage_url") or metadata.get("url") or ""}
Topic: {config.TOPIC}

## Transcript

{transcript_md[:60000]}

## Visual notes / OCR from extracted frames

{visual_notes_md[:30000]}
"""
    refs_dir.mkdir(parents=True, exist_ok=True)
    (refs_dir / "llm_input.md").write_text(prompt, encoding="utf-8")
    return prompt


def render_skill_md(data: dict, metadata: dict, created_at: str) -> str:
    """Render the structured LLM output dict into a SKILL.md string."""
    lines = [
        "---",
        f"name: {data['name']}",
        f"description: {data['description']}",
        f"created_at: {created_at}",
        f"source_url: {metadata.get('webpage_url') or metadata.get('url') or ''}",
        f"source_title: {metadata.get('title') or 'Unknown'}",
        "---",
        "",
        f"# {data['name']}",
        "",
        "## Summary",
        "",
        data["summary"],
        "",
        "## Knowledge",
        "",
    ]
    for chapter in data["knowledge"]:
        lines += [f"### {chapter['chapter_title']}", ""]
        for key, label in CATEGORY_LABELS.items():
            items = chapter.get(key) or []
            if items:
                lines += [f"#### {label}", ""]
                lines += [f"- {item}" for item in items]
                lines.append("")
    return "\n".join(lines)


def generate_skill(prompt: str, metadata: dict, created_at: str) -> str | None:
    import requests
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set.")
        return None
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    print(f"Generating SKILL.md with {config.OPENAI_MODEL}...")
    resp = requests.post(
        f"{base_url}/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": config.OPENAI_MODEL,
            "instructions": "You convert YouTube tutorials into structured skill documents. Return only valid JSON matching the provided schema. No markdown fences, no extra text.",
            "input": prompt,
            "temperature": 0.2,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "skill_output",
                    "schema": SKILL_SCHEMA,
                    "strict": True,
                }
            },
        },
        timeout=120,
    )
    if resp.status_code >= 400:
        print(f"API error {resp.status_code}: {resp.text[:500]}")
        return None
    raw_json = resp.json()["output"][0]["content"][0]["text"]
    data = json.loads(raw_json)
    return render_skill_md(data, metadata, created_at)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a YouTube video into a local SKILL.md bundle.")
    parser.add_argument("url", nargs="?", help="YouTube URL or video ID.")
    args = parser.parse_args()

    url = args.url or input("YouTube URL or video ID: ").strip()
    if not url:
        print("No URL provided.")
        return 2

    try:
        video_id = extract_video_id(url)
    except ValueError as e:
        print(e)
        return 2

    out_dir = Path(f"skill_{video_id}_{slugify(config.TOPIC)}").resolve()
    refs_dir = out_dir / "references"
    assets_dir = out_dir / "assets"
    work_dir = out_dir / "_work"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    metadata = fetch_metadata(url, refs_dir)

    segments = fetch_transcript(video_id)
    if segments:
        transcript_md = save_transcript(segments, refs_dir)
    else:
        transcript_md = "_No transcript available._\n"
        refs_dir.mkdir(parents=True, exist_ok=True)
        (refs_dir / "transcript_clean.md").write_text(transcript_md, encoding="utf-8")

    frames = []
    if not config.SKIP_VIDEO:
        video_path = download_video(url, work_dir)
        if video_path:
            frames = extract_frames(video_path, assets_dir)

    visual_notes_md = build_visual_notes(frames, refs_dir, out_dir)
    llm_input = build_llm_input(metadata, transcript_md, visual_notes_md, refs_dir)

    created_at = datetime.now().isoformat(timespec="seconds")
    skill_md = generate_skill(llm_input, metadata, created_at)
    if not skill_md:
        print("SKILL.md generation failed.")
        return 1

    (out_dir / "SKILL.md").write_text(skill_md.strip() + "\n", encoding="utf-8")

    report = {
        "created_at": created_at,
        "url": url, "video_id": video_id, "topic": config.TOPIC,
        "transcript_segments": len(segments or []), "frames": len(frames),
    }
    (refs_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Done. Skill: {out_dir / 'SKILL.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
