from __future__ import annotations

from pathlib import Path
from typing import Any

from openai import OpenAI

from clipmind.config import Settings
from clipmind.models import Transcript, TranscriptWord
from clipmind.utils import get_logger, read_json, run_command, slugify, write_json


class Transcriber:
    """Extracts audio and caches Whisper word-level transcription results."""

    def __init__(self, settings: Settings, client: OpenAI | None = None) -> None:
        self.settings = settings
        self._client = client
        self.logger = get_logger()

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    def transcript_path(self, video_path: Path) -> Path:
        return self.settings.cache_dir / f"{slugify(video_path.stem)}.transcript.json"

    def audio_path(self, video_path: Path) -> Path:
        return self.settings.temp_dir / f"{slugify(video_path.stem)}.whisper.mp3"

    def extract_audio(self, video_path: Path, force: bool = False) -> Path:
        output_path = self.audio_path(video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not force:
            self.logger.info("Using cached audio: %s", output_path)
            return output_path

        self.logger.info("Extracting audio for Whisper")
        run_command(
            [
                self.settings.ffmpeg_bin,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "64k",
                str(output_path),
            ]
        )
        return output_path

    def transcribe(self, video_path: Path, force: bool = False) -> Transcript:
        transcript_path = self.transcript_path(video_path)
        if transcript_path.exists() and not force:
            self.logger.info("Using cached transcript: %s", transcript_path)
            return Transcript.model_validate(read_json(transcript_path))

        self.settings.require_openai_key()
        audio_path = self.extract_audio(video_path, force=force)
        self.logger.info("Sending audio to Whisper")
        with audio_path.open("rb") as audio_file:
            response = self.client.audio.transcriptions.create(
                model=self.settings.openai_transcription_model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

        raw_response = self._to_dict(response)
        transcript = self._normalize_response(raw_response, video_path)
        write_json(transcript_path, transcript.model_dump(mode="json"))
        self.logger.info("Transcript cached: %s", transcript_path)
        return transcript

    def _normalize_response(self, response: dict[str, Any], video_path: Path) -> Transcript:
        raw_words = response.get("words") or []
        words: list[TranscriptWord] = []
        for raw_word in raw_words:
            text = str(raw_word.get("word") or raw_word.get("text") or "").strip()
            if not text:
                continue
            words.append(
                TranscriptWord(
                    text=text,
                    start=float(raw_word.get("start", 0)),
                    end=float(raw_word.get("end", raw_word.get("start", 0))),
                    confidence=raw_word.get("confidence"),
                )
            )

        return Transcript(
            source_video=str(video_path),
            language=response.get("language"),
            duration=response.get("duration"),
            text=str(response.get("text", "")),
            words=words,
            raw_response=response,
        )

    @staticmethod
    def _to_dict(response: Any) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return response
        return dict(response)
