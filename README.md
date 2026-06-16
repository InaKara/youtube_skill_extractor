# youtube_skill_extractor
Download youtube videos, extract core content from them, store it as a structured skill.md file

Remove-Item -Recurse -Force .\youtube_wh489_XT5TI -ErrorAction SilentlyContinue
clear; uv run youtube_to_markdown.py "https://www.youtube.com/watch?v=wh489_XT5TI" --visual-analysis no --langs en-orig,en
Get-ChildItem .\youtube_wh489_XT5TI\raw\*.vtt | Select-Object Name, Length

clear; uv run youtube_to_markdown.py "https://www.youtube.com/watch?v=wh489_XT5TI" --visual-analysis no --langs en