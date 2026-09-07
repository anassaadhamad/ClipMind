<div align="center">

# 🎬 ClipMind

### AI-Powered Viral Clips Studio

**Drop a long video. Get viral short-form clips — automatically.**

ClipMind uses **GPT-4o** to find your most compelling moments, **Whisper** for word-level transcription, and **FFmpeg + MediaPipe** to render portrait-mode clips with dynamic captions, face tracking, and platform-ready social media copy.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/GPT--4o-Powered-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![Whisper](https://img.shields.io/badge/Whisper-Word--Level-10b981?style=flat-square&logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Media%20Processing-007808?style=flat-square&logo=ffmpeg&logoColor=white)](https://ffmpeg.org)
[![PySide6](https://img.shields.io/badge/PySide6-Desktop%20GUI-41CD52?style=flat-square&logo=qt&logoColor=white)](https://doc.qt.io/qtforpython)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

[🌐 Live Demo Page](https://anassaadhamad.github.io/ClipMind) · [📖 Docs](#setup) · [🐛 Issues](https://github.com/anassaadhamad/ClipMind/issues)

</div>

---

## ✨ What It Does

| Step | Tool | Output |
|------|------|--------|
| 🎙️ Transcribe | Whisper (word-level) | Cached JSON transcript |
| 🧠 Analyze | GPT-4o | Viral clip metadata with scores |
| ✂️ Cut | FFmpeg | Precise video segments |
| 📱 Reframe | MediaPipe + OpenCV | Smooth 9:16 face-tracked video |
| ✍️ Caption | Pillow (RAQM) | Dynamic word-highlight captions |
| 📣 Distribute | GPT-4o | YouTube / TikTok / Facebook copy |

---

## 🏗️ Architecture

```
Long Video
    │
    ▼
┌─────────────┐    FFmpeg     ┌─────────────────┐
│  Transcriber │ ──────────▶  │  Audio Extract   │
└─────────────┘              └────────┬─────────┘
                                      │ Whisper API
                                      ▼
                             ┌─────────────────┐
                             │  Word-Level JSON │  ◀── SRT Aligner (optional)
                             └────────┬─────────┘
                                      │ GPT-4o
                                      ▼
                             ┌─────────────────┐
                             │  ViralClip[]     │  (scored, timestamped)
                             └────────┬─────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                  ▼
             ┌──────────┐    ┌──────────────┐   ┌─────────────┐
             │  FFmpeg  │    │  MediaPipe   │   │   Pillow    │
             │  Cutter  │    │  Face Track  │   │  Captions   │
             └────┬─────┘    └──────┬───────┘   └──────┬──────┘
                  └─────────────────┴──────────────────┘
                                    │ FFmpeg Composite
                                    ▼
                           ┌─────────────────┐
                           │  1080×1920 MP4  │  + CMX EDL
                           └─────────────────┘
                                    │
                                    ▼
                           ┌─────────────────┐
                           │  Social Media   │  GPT-4o titles,
                           │      Hub        │  descriptions, hashtags
                           └─────────────────┘
```

---

## 🚀 Features

- **AI Viral Detection** — GPT-4o identifies 3–5 high-potential moments with viral scores, hook text, and emotional tags
- **Word-Level Transcription** — Whisper timestamps drive every caption, SFX trigger, and edit point; results are cached locally
- **9:16 Auto-Reframe** — MediaPipe face detection + smoothed OpenCV tracking keeps the speaker centered in portrait mode
- **Dynamic Captions** — Pillow-rendered 2–4 word groups with per-word gold highlight; full Arabic + English with RAQM shaping
- **SRT Alignment** — Fuse corrected `.srt` subtitles with Whisper timestamps for ground-truth Arabic captions
- **Word-Synced SFX** — Optional pop/whoosh sound effects fire in sync with caption words
- **Social Media Hub** — 3 alternative titles, SEO descriptions, CTAs, and hashtags for YouTube, TikTok, and Facebook
- **EDL Export** — CMX 3600 files for Premiere Pro and DaVinci Resolve workflows
- **Dark-Mode Desktop GUI** — PySide6 drag-and-drop interface with viral score cards, threaded processing, and live logs

---

## 📋 Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe` available on `PATH`
- OpenAI API key (GPT-4o + Whisper)
- A `.ttf` / `.otf` font with Arabic + English glyphs

---

## ⚙️ Setup

```powershell
# 1. Clone the repo
git clone https://github.com/anassaadhamad/ClipMind.git
cd ClipMind

# 2. Create a virtual environment and install
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

# 3. Configure environment
Copy-Item .env.example .env
```

Edit `.env` — at minimum set:

```env
OPENAI_API_KEY=sk-your-key-here
FONT_PATH=assets/fonts/YourFont.ttf
```

Place your font in `assets/fonts/` (a font supporting Arabic + Latin is included by default).

---

## 🖥️ Usage

### Desktop GUI

```powershell
clipmind gui
# or
clipmind-gui
```

Drag-and-drop your video, click **Analyze**, review viral score cards, then **Process All** or cherry-pick individual clips. The Social Media Hub generates per-platform copy with one click.

### CLI

```powershell
# Full pipeline — detect & render 5 viral clips
clipmind process input_videos\long_video.mp4 --clips 5

# With reference SRT for accurate Arabic captions
clipmind process input_videos\long_video.mp4 --srt input_videos\long_video.srt --clips 5

# Dry-run: analyze only, no rendering (saves API cost)
clipmind process input_videos\long_video.mp4 --dry-run

# Transcribe only (cache transcript, skip GPT)
clipmind transcribe input_videos\long_video.mp4

# GPT analysis only (uses cached transcript)
clipmind analyze input_videos\long_video.mp4 --clips 3

# Force re-analysis even if cache exists
clipmind process input_videos\long_video.mp4 --force-analysis
```

---

## 📁 Folder Layout

```
ClipMind/
├── src/clipmind/          # Core Python package
│   ├── analyzer.py        # GPT-4o viral detection
│   ├── transcriber.py     # Whisper transcription
│   ├── video_processor.py # FFmpeg + MediaPipe reframing
│   ├── captions.py        # Pillow caption rendering
│   ├── aligner.py         # SRT ↔ Whisper timestamp fusion
│   ├── social.py          # Social media copy generation
│   ├── social_widgets.py  # Social Media Hub UI
│   ├── gui.py             # PySide6 desktop application
│   ├── cli.py             # Click CLI entry points
│   ├── config.py          # Pydantic settings (via .env)
│   ├── edl.py             # CMX 3600 EDL export
│   └── models.py          # Pydantic data models
├── assets/
│   ├── fonts/             # Caption fonts (.ttf / .otf)
│   └── sfx/               # Optional pop/whoosh SFX (.mp3)
├── input_videos/          # Default video input location
├── cache/                 # Whisper transcripts & GPT clip JSON
├── outputs/               # Rendered MP4 clips & EDL files
├── temp/                  # Extracted audio & intermediate files
├── docs/                  # GitHub Pages landing page
└── tests/                 # Pytest test suite
```

---

## 🧪 Tests

```powershell
pytest
```

Covers JSON validation, caption grouping, timestamp filtering, timecode formatting, SRT alignment, and EDL generation. Full media rendering requires a configured `.env` and a short sample video.

---

## 💡 Notes

- **Caching** — Whisper transcripts and GPT clip JSON are cached locally. Reprocessing the same video skips paid API calls unless `--force-analysis` is used.
- **Arabic Support** — Pillow must be built with RAQM for correct Arabic shaping. The font must include Arabic glyphs.
- **SFX** — Place a short audio file at `assets/sfx/pop.mp3` (or set `SFX_POP_PATH` in `.env`) to enable word-synced sound effects. ClipMind renders normally if no file is found.
- **Dry-run** — Runs full transcription + GPT analysis without touching FFmpeg. Useful for previewing viral segments before committing to a render.

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ by [Anas Saad Hamad](https://github.com/anassaadhamad)

⭐ If this project helped you, give it a star!

</div>
