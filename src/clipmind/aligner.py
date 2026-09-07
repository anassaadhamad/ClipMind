from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from clipmind.models import Transcript, TranscriptWord
from clipmind.utils import get_logger, slugify, write_json


SRT_TIMESTAMP_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)
TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[\w\u0600-\u06FF]+|[^\s\w\u0600-\u06FF]", re.UNICODE)
WORD_RE = re.compile(r"[\w\u0600-\u06FF]+", re.UNICODE)
ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")


@dataclass(frozen=True)
class SRTCue:
    start: float
    end: float
    text: str


class TextAligner:
    """Uses accurate SRT text to correct Whisper words while preserving timings."""

    def __init__(self) -> None:
        self.logger = get_logger()

    def aligned_path(self, cache_dir: Path, video_path: Path, srt_path: Path) -> Path:
        return cache_dir / f"{slugify(video_path.stem)}.{slugify(srt_path.stem)}.aligned.json"

    def align_from_srt(
        self,
        transcript: Transcript,
        srt_path: Path,
        *,
        cache_dir: Path | None = None,
        video_path: Path | None = None,
    ) -> Transcript:
        cues = self.parse_srt(srt_path)
        srt_text = self.cues_to_text(cues)
        srt_words = self.tokenize_words(srt_text)
        if not transcript.words or not srt_words:
            return transcript.model_copy(update={"text": srt_text})

        self.logger.info("Aligning Whisper word timestamps to SRT ground-truth text")
        aligned_words = self.align_words(transcript.words, srt_words)
        aligned = transcript.model_copy(
            update={
                "text": srt_text,
                "words": aligned_words,
                "raw_response": {
                    **(transcript.raw_response or {}),
                    "alignment_source": str(srt_path),
                    "alignment_word_count": len(aligned_words),
                },
            }
        )

        if cache_dir and video_path:
            output_path = self.aligned_path(cache_dir, video_path, srt_path)
            write_json(output_path, aligned.model_dump(mode="json"))
            self.logger.info("Aligned transcript cached: %s", output_path)
        return aligned

    def parse_srt(self, path: Path) -> list[SRTCue]:
        content = path.read_text(encoding="utf-8-sig", errors="replace")
        blocks = re.split(r"\n\s*\n", content.replace("\r\n", "\n").replace("\r", "\n").strip())
        cues: list[SRTCue] = []
        for block in blocks:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            if not lines:
                continue
            timestamp_index = next((index for index, line in enumerate(lines) if "-->" in line), None)
            if timestamp_index is None:
                continue
            match = SRT_TIMESTAMP_RE.search(lines[timestamp_index])
            if not match:
                continue
            text_lines = lines[timestamp_index + 1 :]
            text = self.clean_srt_text(" ".join(text_lines))
            if text:
                cues.append(
                    SRTCue(
                        start=self.parse_timestamp(match.group("start")),
                        end=self.parse_timestamp(match.group("end")),
                        text=text,
                    )
                )
        return cues

    @staticmethod
    def cues_to_text(cues: list[SRTCue]) -> str:
        return " ".join(cue.text for cue in cues).strip()

    @staticmethod
    def clean_srt_text(text: str) -> str:
        text = TAG_RE.sub("", text)
        text = text.replace("{\\an8}", "").replace("{\\an7}", "").replace("{\\an9}", "")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def parse_timestamp(value: str) -> float:
        hours, minutes, rest = value.split(":")
        seconds, millis = rest.split(",")
        return (
            int(hours) * 3600
            + int(minutes) * 60
            + int(seconds)
            + int(millis) / 1000
        )

    @staticmethod
    def tokenize_words(text: str) -> list[str]:
        return WORD_RE.findall(text)

    def align_words(
        self,
        whisper_words: list[TranscriptWord],
        srt_words: list[str],
    ) -> list[TranscriptWord]:
        whisper_norm = [self.normalize_word(word.text) for word in whisper_words]
        srt_norm = [self.normalize_word(word) for word in srt_words]
        matcher = SequenceMatcher(None, whisper_norm, srt_norm, autojunk=False)
        aligned: list[TranscriptWord] = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            source = whisper_words[i1:i2]
            target = srt_words[j1:j2]
            if tag == "equal":
                aligned.extend(
                    source_word.model_copy(update={"text": target_word})
                    for source_word, target_word in zip(source, target)
                )
            elif tag in {"replace", "insert"} and target:
                aligned.extend(self._timed_target_words(target, source, whisper_words, i1, aligned))
            elif tag == "delete":
                continue

        return self._dedupe_monotonic(aligned)

    def _timed_target_words(
        self,
        target_words: list[str],
        source_words: list[TranscriptWord],
        all_whisper_words: list[TranscriptWord],
        source_start_index: int,
        aligned_so_far: list[TranscriptWord],
    ) -> list[TranscriptWord]:
        if not target_words:
            return []

        if len(source_words) == len(target_words):
            return [
                source.model_copy(update={"text": target})
                for source, target in zip(source_words, target_words)
            ]

        if source_words:
            start = source_words[0].start
            end = source_words[-1].end
            confidence = self._average_confidence(source_words)
        else:
            previous_end = aligned_so_far[-1].end if aligned_so_far else None
            next_start = (
                all_whisper_words[source_start_index].start
                if source_start_index < len(all_whisper_words)
                else None
            )
            start = previous_end if previous_end is not None else (next_start or 0.0)
            end = next_start if next_start is not None and next_start > start else start + 0.2 * len(target_words)
            confidence = None

        if end <= start:
            end = start + 0.2 * len(target_words)

        step = (end - start) / len(target_words)
        return [
            TranscriptWord(
                text=target,
                start=round(start + index * step, 3),
                end=round(start + (index + 1) * step, 3),
                confidence=confidence,
            )
            for index, target in enumerate(target_words)
        ]

    @staticmethod
    def _average_confidence(words: list[TranscriptWord]) -> float | None:
        confidences = [word.confidence for word in words if word.confidence is not None]
        if not confidences:
            return None
        return round(sum(confidences) / len(confidences), 4)

    @staticmethod
    def _dedupe_monotonic(words: list[TranscriptWord]) -> list[TranscriptWord]:
        monotonic: list[TranscriptWord] = []
        previous_end = 0.0
        for word in words:
            start = max(word.start, previous_end)
            end = max(word.end, start)
            fixed = word.model_copy(update={"start": round(start, 3), "end": round(end, 3)})
            monotonic.append(fixed)
            previous_end = fixed.end
        return monotonic

    @staticmethod
    def normalize_word(value: str) -> str:
        text = unicodedata.normalize("NFKC", value).casefold()
        text = ARABIC_DIACRITICS_RE.sub("", text)
        text = text.replace("ـ", "")
        translations = str.maketrans(
            {
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ٱ": "ا",
                "ى": "ي",
                "ؤ": "و",
                "ئ": "ي",
                "ة": "ه",
            }
        )
        text = text.translate(translations)
        text = "".join(char for char in text if char.isalnum() or "\u0600" <= char <= "\u06FF")
        return text
