from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TranscriptWord(BaseModel):
    """A single word with absolute timestamps in the source video."""

    text: str = Field(min_length=1)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def validate_times(self) -> "TranscriptWord":
        if self.end < self.start:
            raise ValueError("word end time must be greater than or equal to start time")
        return self


class Transcript(BaseModel):
    """Normalized transcript cached by ClipMind."""

    source_video: str
    language: str | None = None
    duration: float | None = Field(default=None, ge=0)
    text: str = ""
    words: list[TranscriptWord] = Field(default_factory=list)
    raw_response: dict[str, Any] | None = None

    def words_for_range(self, start_time: float, end_time: float) -> list[TranscriptWord]:
        return [
            word
            for word in self.words
            if word.end >= start_time and word.start <= end_time
        ]


class ViralClip(BaseModel):
    """A clip selected by the AI analyzer."""

    clip_id: str = Field(min_length=1)
    start_time: float = Field(ge=0)
    end_time: float = Field(gt=0)
    viral_score: int = Field(ge=1, le=100)
    hook_text: str = Field(min_length=1)
    explanation: str = Field(min_length=1)

    @field_validator("clip_id", mode="before")
    @classmethod
    def normalize_clip_id(cls, value: object) -> str:
        return str(value).strip().replace(" ", "_")

    @field_validator("viral_score", mode="before")
    @classmethod
    def normalize_viral_score(cls, value: object) -> int:
        score = float(value)
        if 0 < score <= 10:
            score *= 10
        return round(score)

    @model_validator(mode="after")
    def validate_duration(self) -> "ViralClip":
        if self.end_time <= self.start_time:
            raise ValueError("clip end_time must be greater than start_time")
        return self

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class CaptionToken(BaseModel):
    text: str
    start: float
    end: float
    is_active: bool = False


class CaptionWindow(BaseModel):
    start: float
    end: float
    tokens: list[CaptionToken]


class FaceTrackPoint(BaseModel):
    time: float
    x_center: float = Field(ge=0, le=1)
    confidence: float = Field(default=0, ge=0, le=1)
    detected: bool = False


class ProcessedClip(BaseModel):
    clip: ViralClip
    raw_segment_path: Path
    vertical_path: Path
    captioned_path: Path

    model_config = ConfigDict(arbitrary_types_allowed=True)


SocialPlatform = Literal["youtube", "facebook", "tiktok"]


class SocialProfile(BaseModel):
    platform: SocialPlatform
    channel_url: str = ""
    default_description: str = ""


class SocialPost(BaseModel):
    platform: SocialPlatform
    clip_id: str
    title: str = Field(min_length=1)
    titles: list[str] = Field(default_factory=list)
    description: str = Field(min_length=1)
    hashtags: list[str] = Field(default_factory=list)
    call_to_action: str = ""
    compliance_notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_title_alternatives(self) -> "SocialPost":
        if not self.titles:
            self.titles = [self.title]
        if self.title not in self.titles:
            self.titles.insert(0, self.title)
        self.titles = self.titles[:3]
        return self


class SocialContentBundle(BaseModel):
    source_video: str
    posts: list[SocialPost] = Field(default_factory=list)
