from __future__ import annotations

import json
import re
from pathlib import Path

from openai import BadRequestError, OpenAI
from pydantic import TypeAdapter, ValidationError

from clipmind.config import Settings
from clipmind.models import Transcript, ViralClip
from clipmind.utils import get_logger, read_json, slugify, write_json


CLIP_ARRAY_ADAPTER = TypeAdapter(list[ViralClip])


SYSTEM_PROMPT = """You are an elite TikTok, YouTube Shorts, and Instagram Reels strategist.
Your job is to find complete, emotionally compelling moments in long-form transcripts.
Prioritize hooks, surprise, conflict, punchlines, clear educational payoff, and complete thoughts.
Avoid segments that begin or end mid-sentence unless the surrounding words still make the clip feel complete.
When helpful, add one relevant emoji to hook_text or punchline-oriented hook wording, such as 🔥, 🤯, 💡, 😂, or 👀. Use emojis sparingly and only when they match the emotion.
Return only strict JSON. Do not include markdown, prose, comments, or trailing commas."""


class AIAnalyzer:
    """Uses GPT-4o to identify viral short-form clip candidates."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self._client = client
        self.logger = get_logger()

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def clips_path(self, video_path: Path) -> Path:
        return self.settings.cache_dir / f"{slugify(video_path.stem)}.clips.json"

    def analyze(
        self,
        transcript: Transcript,
        video_path: Path,
        *,
        clip_count: int | None = None,
        force: bool = False,
    ) -> list[ViralClip]:
        output_path = self.clips_path(video_path)
        if output_path.exists() and not force:
            self.logger.info("Using cached AI clip analysis: %s", output_path)
            return self.parse_clips(read_json(output_path))

        self.settings.require_openai_key()
        requested_count = clip_count or self.settings.default_clip_count
        self.logger.info("Asking GPT to select viral clips")
        response_text = self._request_clip_json(transcript, requested_count)
        clips = self.parse_clips(response_text)
        clips = self._validate_clip_constraints(clips, requested_count)
        write_json(output_path, [clip.model_dump(mode="json") for clip in clips])
        self.logger.info("Clip analysis cached: %s", output_path)
        return clips

    def parse_clips(self, payload: str | list[dict] | dict) -> list[ViralClip]:
        if isinstance(payload, str):
            parsed = json.loads(self._extract_json(payload))
        else:
            parsed = payload

        if isinstance(parsed, dict) and "clips" in parsed:
            parsed = parsed["clips"]

        try:
            return CLIP_ARRAY_ADAPTER.validate_python(parsed)
        except ValidationError as exc:
            raise ValueError(f"AI clip JSON failed validation: {exc}") from exc

    def _request_clip_json(self, transcript: Transcript, clip_count: int) -> str:
        user_prompt = self._build_user_prompt(transcript, clip_count)
        schema = self._clip_object_schema()
        try:
            completion = self.client.chat.completions.create(
                model=self.settings.openai_analysis_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "viral_clip_selection",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )
        except BadRequestError:
            self.logger.warning("Structured output request failed; retrying with JSON object mode")
            completion = self.client.chat.completions.create(
                model=self.settings.openai_analysis_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": user_prompt
                        + "\nIf the API requires an object, return {\"clips\": [...]} with the array inside clips.",
                    },
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
            )

        message = completion.choices[0].message.content
        if not message:
            raise RuntimeError("OpenAI returned an empty clip analysis response")
        return message

    def _build_user_prompt(self, transcript: Transcript, clip_count: int) -> str:
        transcript_body = self._compact_transcript(transcript)
        return f"""Find the top {clip_count} viral short-form clips.

Hard constraints:
- Each clip must be between {self.settings.min_clip_seconds} and {self.settings.max_clip_seconds} seconds.
- Return a strict JSON array of objects, or an object with a single clips array if the API requires a root object.
- Each object must contain clip_id, start_time, end_time, viral_score, hook_text, and explanation.
- clip_id must be a string like "clip_1".
- viral_score must be an integer from 1 to 100, not a decimal and not a 1-10 score.
- start_time and end_time must be absolute seconds from the source video.
- hook_text should be short, bold, and suitable as a top-screen caption.
- hook_text may include one relevant emoji when it improves the hook or punchline. Do not overuse emojis.

Transcript with approximate timestamps:
{transcript_body}"""

    def _compact_transcript(self, transcript: Transcript, bucket_seconds: int = 15) -> str:
        if not transcript.words:
            return transcript.text

        lines: list[str] = []
        bucket_start = transcript.words[0].start
        current_words: list[str] = []
        for word in transcript.words:
            if word.start - bucket_start >= bucket_seconds and current_words:
                lines.append(f"[{bucket_start:.1f}s] {' '.join(current_words)}")
                bucket_start = word.start
                current_words = []
            current_words.append(word.text)
        if current_words:
            lines.append(f"[{bucket_start:.1f}s] {' '.join(current_words)}")
        return "\n".join(lines)

    def _validate_clip_constraints(self, clips: list[ViralClip], clip_count: int) -> list[ViralClip]:
        valid = []
        for clip in clips:
            if clip.duration < self.settings.min_clip_seconds or clip.duration > self.settings.max_clip_seconds:
                self.logger.warning(
                    "Skipping %s because duration %.1fs is outside %s-%ss",
                    clip.clip_id,
                    clip.duration,
                    self.settings.min_clip_seconds,
                    self.settings.max_clip_seconds,
                )
                continue
            valid.append(clip)

        if not valid:
            raise ValueError("AI did not return any clips within the configured duration constraints")

        return sorted(valid, key=lambda clip: clip.viral_score, reverse=True)[:clip_count]

    @staticmethod
    def _extract_json(payload: str) -> str:
        stripped = payload.strip()
        if stripped.startswith("[") or stripped.startswith("{"):
            return stripped

        fenced = re.search(r"```(?:json)?\s*(.*?)```", payload, re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        array_match = re.search(r"\[[\s\S]*\]", payload)
        if array_match:
            return array_match.group(0)

        object_match = re.search(r"\{[\s\S]*\}", payload)
        if object_match:
            return object_match.group(0)

        raise ValueError("Could not find a JSON array or object in the AI response")

    @staticmethod
    def _clip_object_schema() -> dict:
        item_schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "clip_id",
                "start_time",
                "end_time",
                "viral_score",
                "hook_text",
                "explanation",
            ],
            "properties": {
                "clip_id": {"type": "string"},
                "start_time": {"type": "number", "minimum": 0},
                "end_time": {"type": "number", "exclusiveMinimum": 0},
                "viral_score": {"type": "integer", "minimum": 1, "maximum": 100},
                "hook_text": {"type": "string"},
                "explanation": {"type": "string"},
            },
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["clips"],
            "properties": {
                "clips": {
                    "type": "array",
                    "items": item_schema,
                    "minItems": 1,
                    "maxItems": 10,
                }
            },
        }
