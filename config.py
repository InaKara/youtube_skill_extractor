"""
config.py — Runtime configuration for youtube_to_skill.py.

Edit these values to customize the skill extraction pipeline.
Secrets (API keys, cookies) belong in .env, not here.
"""

# Transcript language preference (e.g. "en", "de", "tr").
LANGUAGE = "en"

# Topic label used in output directory naming and SKILL.md metadata.
TOPIC = "the tutorial topic"

# Maximum number of key frames to keep from the video.
MAX_FRAMES = 40

# ffmpeg scene-change detection threshold (0.0 – 1.0).
# Lower values = more frames extracted. 0.25 is a good default.
SCENE_THRESHOLD = 0.25

# Set to False to disable OCR on extracted frames.
OCR_ENABLED = True

# Set to True to skip video download and frame extraction entirely.
SKIP_VIDEO = False

# OpenAI model used for SKILL.md generation.
OPENAI_MODEL = "gpt-4.1-mini"
