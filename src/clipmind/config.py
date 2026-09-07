from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


PROJECT_ROOT = Path.cwd()
DEFAULT_FONT_PATH = Path("assets/fonts/Cairo-Black.ttf")
DEFAULT_SFX_POP_PATH = Path("assets/sfx/pop.mp3")


class Settings(BaseModel):
    openai_api_key: str = ""
    openai_transcription_model: str = "whisper-1"
    openai_analysis_model: str = "gpt-4o"

    input_dir: Path = Path("input_videos")
    cache_dir: Path = Path("cache")
    output_dir: Path = Path("outputs")
    temp_dir: Path = Path("temp")
    font_path: Path = DEFAULT_FONT_PATH

    default_clip_count: int = Field(default=5, ge=1, le=10)
    min_clip_seconds: int = Field(default=30, ge=5)
    max_clip_seconds: int = Field(default=60, ge=10)

    export_width: int = Field(default=1080, ge=360)
    export_height: int = Field(default=1920, ge=640)
    export_fps: int = Field(default=30, ge=15, le=120)
    tracking_sample_fps: int = Field(default=4, ge=1, le=30)
    smoothing_window: int = Field(default=9, ge=1, le=101)

    caption_max_words: int = Field(default=4, ge=1, le=8)
    caption_min_words: int = Field(default=2, ge=1, le=4)
    caption_font_size: int = Field(default=78, ge=12)
    caption_active_color: str = "#FFD400"
    caption_inactive_color: str = "#FFFFFF"
    caption_stroke_color: str = "#000000"
    caption_stroke_width: int = Field(default=5, ge=0, le=20)
    caption_bottom_margin: int = Field(default=260, ge=0)

    youtube_channel_url: str = ""
    youtube_default_description: str = ""
    facebook_page_url: str = ""
    facebook_default_description: str = ""
    tiktok_profile_url: str = ""
    tiktok_default_description: str = ""

    sfx_pop_path: Path = DEFAULT_SFX_POP_PATH
    sfx_volume: float = Field(default=0.3, ge=0, le=1)
    enable_word_sfx: bool = True

    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"

    @field_validator(
        "input_dir",
        "cache_dir",
        "output_dir",
        "temp_dir",
        "font_path",
        "sfx_pop_path",
        mode="before",
    )
    @classmethod
    def expand_path(cls, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if path.is_absolute():
            return path
        return PROJECT_ROOT / path

    @field_validator("caption_active_color", "caption_inactive_color", "caption_stroke_color")
    @classmethod
    def validate_hex_color(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) != 7 or not stripped.startswith("#"):
            raise ValueError("colors must be hex values like #FFD400")
        int(stripped[1:], 16)
        return stripped.upper()

    @field_validator("smoothing_window")
    @classmethod
    def make_smoothing_window_odd(cls, value: int) -> int:
        return value if value % 2 == 1 else value + 1

    def ensure_directories(self) -> None:
        for directory in (self.input_dir, self.cache_dir, self.output_dir, self.temp_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def require_openai_key(self) -> None:
        if not self.openai_api_key or self.openai_api_key.startswith("sk-your-key"):
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to .env or your environment.")

    def require_font(self) -> None:
        if not self.font_path.exists():
            raise FileNotFoundError(
                f"Caption font not found at {self.font_path}. Place a .ttf/.otf file in "
                "assets/fonts and set FONT_PATH in .env."
            )


def load_settings(env_file: Path | str = ".env") -> Settings:
    env_path = Path(env_file)
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()

    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_transcription_model=os.getenv("OPENAI_TRANSCRIPTION_MODEL", "whisper-1"),
        openai_analysis_model=os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4o"),
        input_dir=os.getenv("INPUT_DIR", "input_videos"),
        cache_dir=os.getenv("CACHE_DIR", "cache"),
        output_dir=os.getenv("OUTPUT_DIR", "outputs"),
        temp_dir=os.getenv("TEMP_DIR", "temp"),
        font_path=os.getenv("FONT_PATH", str(DEFAULT_FONT_PATH)),
        default_clip_count=int(os.getenv("DEFAULT_CLIP_COUNT", "5")),
        min_clip_seconds=int(os.getenv("MIN_CLIP_SECONDS", "30")),
        max_clip_seconds=int(os.getenv("MAX_CLIP_SECONDS", "60")),
        export_width=int(os.getenv("EXPORT_WIDTH", "1080")),
        export_height=int(os.getenv("EXPORT_HEIGHT", "1920")),
        export_fps=int(os.getenv("EXPORT_FPS", "30")),
        tracking_sample_fps=int(os.getenv("TRACKING_SAMPLE_FPS", "4")),
        smoothing_window=int(os.getenv("SMOOTHING_WINDOW", "9")),
        caption_max_words=int(os.getenv("CAPTION_MAX_WORDS", "4")),
        caption_min_words=int(os.getenv("CAPTION_MIN_WORDS", "2")),
        caption_font_size=int(os.getenv("CAPTION_FONT_SIZE", "78")),
        caption_active_color=os.getenv("CAPTION_ACTIVE_COLOR", "#FFD400"),
        caption_inactive_color=os.getenv("CAPTION_INACTIVE_COLOR", "#FFFFFF"),
        caption_stroke_color=os.getenv("CAPTION_STROKE_COLOR", "#000000"),
        caption_stroke_width=int(os.getenv("CAPTION_STROKE_WIDTH", "5")),
        caption_bottom_margin=int(os.getenv("CAPTION_BOTTOM_MARGIN", "260")),
        youtube_channel_url=os.getenv("YOUTUBE_CHANNEL_URL", ""),
        youtube_default_description=_env_multiline("YOUTUBE_DEFAULT_DESCRIPTION"),
        facebook_page_url=os.getenv("FACEBOOK_PAGE_URL", ""),
        facebook_default_description=_env_multiline("FACEBOOK_DEFAULT_DESCRIPTION"),
        tiktok_profile_url=os.getenv("TIKTOK_PROFILE_URL", ""),
        tiktok_default_description=_env_multiline("TIKTOK_DEFAULT_DESCRIPTION"),
        sfx_pop_path=os.getenv("SFX_POP_PATH", str(DEFAULT_SFX_POP_PATH)),
        sfx_volume=float(os.getenv("SFX_VOLUME", "0.3")),
        enable_word_sfx=os.getenv("ENABLE_WORD_SFX", "true").lower() in {"1", "true", "yes", "on"},
        ffmpeg_bin=os.getenv("FFMPEG_BIN", "ffmpeg"),
        ffprobe_bin=os.getenv("FFPROBE_BIN", "ffprobe"),
    )


def _env_multiline(key: str) -> str:
    return os.getenv(key, "").replace("\\n", "\n")
