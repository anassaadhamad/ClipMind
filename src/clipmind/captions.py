from __future__ import annotations

import re
from pathlib import Path

try:
    from moviepy import AudioFileClip, CompositeAudioClip, VideoFileClip
except ImportError:  # pragma: no cover - moviepy v1 compatibility
    try:
        from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip
    except ImportError:  # pragma: no cover - optional fallback
        AudioFileClip = CompositeAudioClip = VideoFileClip = None

from clipmind.config import DEFAULT_FONT_PATH, DEFAULT_SFX_POP_PATH, Settings
from clipmind.models import CaptionToken, CaptionWindow, TranscriptWord, ViralClip
from clipmind.utils import ffprobe_duration, get_logger, run_command


FONT_PATH = DEFAULT_FONT_PATH
SFX_POP_PATH = DEFAULT_SFX_POP_PATH
ASS_FONT_WEIGHT_SUFFIXES = {
    "black",
    "bold",
    "extrabold",
    "extra-bold",
    "semibold",
    "semi-bold",
    "medium",
    "regular",
    "light",
}


class CaptionEngine:
    """Builds word windows and burns dynamic captions into vertical clips."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger()

    def build_windows(self, words: list[TranscriptWord], clip: ViralClip) -> list[CaptionWindow]:
        clip_words = [
            CaptionToken(
                text=word.text,
                start=round(max(0.0, word.start - clip.start_time), 3),
                end=round(max(0.0, word.end - clip.start_time), 3),
            )
            for word in words
            if word.end >= clip.start_time and word.start <= clip.end_time
        ]
        if not clip_words:
            return []

        windows: list[CaptionWindow] = []
        for index in range(0, len(clip_words), self.settings.caption_max_words):
            tokens = clip_words[index : index + self.settings.caption_max_words]
            windows.append(CaptionWindow(start=tokens[0].start, end=tokens[-1].end, tokens=tokens))

        if (
            len(windows) > 1
            and len(windows[-1].tokens) < self.settings.caption_min_words
            and len(windows[-2].tokens) + len(windows[-1].tokens) <= self.settings.caption_max_words
        ):
            last = windows.pop()
            previous = windows.pop()
            tokens = previous.tokens + last.tokens
            windows.append(CaptionWindow(start=tokens[0].start, end=tokens[-1].end, tokens=tokens))

        return windows

    def burn_captions(
        self,
        input_video: Path,
        output_video: Path,
        windows: list[CaptionWindow],
        hook_text: str,
    ) -> Path:
        self.settings.require_font()
        output_video.parent.mkdir(parents=True, exist_ok=True)
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)

        subtitle_path = self.generate_ass_file(
            windows,
            hook_text,
            self.settings.temp_dir / f"{output_video.stem}.captions.ass",
            duration=self._subtitle_duration(input_video, windows),
        )

        if self._should_use_word_sfx(windows):
            subtitle_video = output_video.with_suffix(".captioned.mp4")
            self._burn_ass_subtitles(
                input_video,
                subtitle_path,
                subtitle_video,
                include_audio=False,
            )
            try:
                self._mux_audio_with_sfx(subtitle_video, input_video, output_video, windows)
                return output_video
            except Exception as exc:
                self.logger.warning(
                    "MoviePy SFX mix failed; burning ASS captions with original audio: %s",
                    exc,
                )
            finally:
                subtitle_video.unlink(missing_ok=True)

        self._burn_ass_subtitles(input_video, subtitle_path, output_video, include_audio=True)
        return output_video

    def generate_ass_file(
        self,
        windows: list[CaptionWindow],
        hook_text: str,
        output_path: Path,
        *,
        duration: float | None = None,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            self.generate_ass_text(windows, hook_text, duration=duration),
            encoding="utf-8-sig",
        )
        return output_path

    def generate_ass_text(
        self,
        windows: list[CaptionWindow],
        hook_text: str,
        *,
        duration: float | None = None,
    ) -> str:
        font_name = self._ass_font_name()
        inactive_color = self._hex_to_ass_color(self.settings.caption_inactive_color)
        active_color = self._hex_to_ass_color(self.settings.caption_active_color)
        stroke_color = self._hex_to_ass_color(self.settings.caption_stroke_color)
        hook_font_size = max(36, int(self.settings.caption_font_size * 0.82))

        lines = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            f"PlayResX: {self.settings.export_width}",
            f"PlayResY: {self.settings.export_height}",
            "",
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
                "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
                "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding"
            ),
            self._ass_style_line(
                name="Caption",
                font_name=font_name,
                font_size=self.settings.caption_font_size,
                primary_color=inactive_color,
                secondary_color=active_color,
                outline_color=stroke_color,
                outline=self.settings.caption_stroke_width,
                alignment=2,
                margin_v=self.settings.caption_bottom_margin,
            ),
            self._ass_style_line(
                name="Hook",
                font_name=font_name,
                font_size=hook_font_size,
                primary_color=inactive_color,
                secondary_color=active_color,
                outline_color=stroke_color,
                outline=self.settings.caption_stroke_width,
                alignment=8,
                margin_v=90,
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]

        subtitle_end = self._subtitle_end_time(windows, duration)
        if hook_text.strip() and subtitle_end > 0:
            hook_dialogue = self._escape_ass_text(hook_text.strip())
            if self._contains_rtl(hook_text):
                hook_dialogue = self._force_rtl(hook_dialogue)
            lines.append(
                self._ass_dialogue(
                    layer=0,
                    start=0.0,
                    end=subtitle_end,
                    style="Hook",
                    text=hook_dialogue,
                )
            )

        for window in windows:
            if not window.tokens:
                continue
            if not self._has_caption_text(window.tokens):
                continue

            line_is_rtl = self._line_is_rtl(window.tokens)

            for index, token in enumerate(window.tokens):
                start = max(window.start, token.start)
                end = min(window.end, token.end)
                if end <= start:
                    continue
                dialogue_text = self._ass_line_for_active_token(
                    window.tokens,
                    index,
                    active_color=active_color,
                    inactive_color=inactive_color,
                )
                if line_is_rtl:
                    dialogue_text = self._force_rtl(dialogue_text)
                lines.append(
                    self._ass_dialogue(
                        layer=0,
                        start=start,
                        end=end,
                        style="Caption",
                        text=dialogue_text,
                    )
                )

        return "\n".join(lines) + "\n"

    def _burn_ass_subtitles(
        self,
        input_video: Path,
        subtitle_path: Path,
        output_video: Path,
        *,
        include_audio: bool,
    ) -> None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        ass_filter = (
            f"ass={self._escape_ffmpeg_filter_path(subtitle_path)}"
            f":fontsdir={self._escape_ffmpeg_filter_path(self._font_path().parent)}"
        )
        command = [
            self.settings.ffmpeg_bin,
            "-y",
            "-i",
            str(input_video),
            "-vf",
            ass_filter,
            "-map",
            "0:v:0",
        ]
        if include_audio:
            command.extend(["-map", "0:a?", "-c:a", "copy", "-shortest"])
        else:
            command.append("-an")
        command.extend(
            [
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-movflags",
                "+faststart",
                str(output_video),
            ]
        )
        run_command(command)

    def _mux_audio_with_sfx(
        self,
        silent_video: Path,
        audio_source: Path,
        output_video: Path,
        windows: list[CaptionWindow],
    ) -> None:
        if AudioFileClip is None or CompositeAudioClip is None or VideoFileClip is None:
            raise RuntimeError("moviepy is not installed")

        sfx_path = self.settings.sfx_pop_path or SFX_POP_PATH
        video_clip = VideoFileClip(str(silent_video))
        base_audio = AudioFileClip(str(audio_source))
        audio_clips = [base_audio]
        try:
            for start_time in self._sfx_start_times(windows):
                sfx_clip = AudioFileClip(str(sfx_path))
                sfx_clip = self._clip_with_volume(sfx_clip, self.settings.sfx_volume)
                sfx_clip = self._clip_with_start(sfx_clip, start_time)
                audio_clips.append(sfx_clip)

            composite_audio = CompositeAudioClip(audio_clips)
            final_clip = self._video_with_audio(video_clip, composite_audio)
            final_clip.write_videofile(
                str(output_video),
                codec="libx264",
                audio_codec="aac",
                preset="veryfast",
                logger=None,
            )
        finally:
            for clip in audio_clips:
                clip.close()
            video_clip.close()
            if "composite_audio" in locals():
                composite_audio.close()
            if "final_clip" in locals():
                final_clip.close()

    def _should_use_word_sfx(self, windows: list[CaptionWindow]) -> bool:
        sfx_path = self.settings.sfx_pop_path or SFX_POP_PATH
        return (
            self.settings.enable_word_sfx
            and bool(windows)
            and sfx_path.exists()
        )

    @staticmethod
    def _clip_with_start(clip, start_time: float):
        if hasattr(clip, "with_start"):
            return clip.with_start(start_time)
        return clip.set_start(start_time)

    @staticmethod
    def _clip_with_volume(clip, volume: float):
        if hasattr(clip, "with_volume_scaled"):
            return clip.with_volume_scaled(volume)
        return clip.volumex(volume)

    @staticmethod
    def _video_with_audio(video_clip, audio_clip):
        if hasattr(video_clip, "with_audio"):
            return video_clip.with_audio(audio_clip)
        return video_clip.set_audio(audio_clip)

    def _font_path(self) -> Path:
        font_path = self.settings.font_path or FONT_PATH
        if not font_path.exists():
            raise FileNotFoundError(
                f"Caption font not found at {font_path}. Set FONT_PATH to a TrueType/OpenType "
                "Arabic font such as assets/fonts/Cairo-Black.ttf."
            )
        return font_path

    def _subtitle_duration(self, input_video: Path, windows: list[CaptionWindow]) -> float:
        window_end = self._subtitle_end_time(windows, duration=None)
        try:
            return max(window_end, ffprobe_duration(self.settings.ffprobe_bin, input_video))
        except Exception as exc:
            self.logger.debug("Could not probe video duration for ASS captions: %s", exc)
            return window_end

    @staticmethod
    def _sfx_start_times(windows: list[CaptionWindow]) -> list[float]:
        starts: set[float] = set()
        for window in windows:
            for token in window.tokens:
                starts.add(round(token.start, 3))
        return sorted(starts)

    @staticmethod
    def _subtitle_end_time(windows: list[CaptionWindow], duration: float | None) -> float:
        window_end = max((window.end for window in windows), default=0.0)
        if duration is None:
            return window_end
        return max(window_end, duration)

    @staticmethod
    def _ass_style_line(
        *,
        name: str,
        font_name: str,
        font_size: int,
        primary_color: str,
        secondary_color: str,
        outline_color: str,
        outline: int,
        alignment: int,
        margin_v: int,
    ) -> str:
        return (
            f"Style: {name},{font_name},{font_size},{primary_color},{secondary_color},"
            f"{outline_color},&H00000000,-1,0,0,0,100,100,0,0,1,"
            f"{outline},0,{alignment},60,60,{margin_v},1"
        )

    @staticmethod
    def _ass_dialogue(
        *,
        layer: int,
        start: float,
        end: float,
        style: str,
        text: str,
    ) -> str:
        return (
            f"Dialogue: {layer},{CaptionEngine._format_ass_time(start)},"
            f"{CaptionEngine._format_ass_time(end)},{style},,0,0,0,,{text}"
        )

    @staticmethod
    def _has_caption_text(tokens: list[CaptionToken]) -> bool:
        return any(token.text.strip() for token in tokens)

    @classmethod
    def _line_is_rtl(cls, tokens: list[CaptionToken]) -> bool:
        return any(cls._contains_rtl(token.text) for token in tokens)

    @staticmethod
    def _contains_rtl(text: str) -> bool:
        return re.search(r"[\u0590-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]", text) is not None

    @staticmethod
    def _force_rtl(text: str) -> str:
        # Wrap the dialogue text in Unicode bidi controls so libass/fribidi
        # always treat the line as a Right-to-Left embedded paragraph, even
        # when the first character is an ASS override token (which some
        # libass builds use to misclassify the paragraph direction as LTR).
        # U+202B = RLE (Right-to-Left Embedding), U+202C = PDF (Pop
        # Directional Formatting).
        return f"\u202B{text}\u202C"

    def _ass_line_for_active_token(
        self,
        tokens: list[CaptionToken],
        active_index: int,
        *,
        active_color: str,
        inactive_color: str,
    ) -> str:
        # Emit tokens in their logical (typed) order with a single inline
        # color override around the active word. libass strips the override
        # tags, runs the line through fribidi as one paragraph, and renders
        # RTL text right-to-left, so the active word ends up on the correct
        # visual side automatically. Wrapping each word in its own override
        # block (or reversing token order) breaks libass's paragraph bidi
        # and shuffles word order across overrides.
        active_tag = self._ass_color_tag(active_color)
        inactive_tag = self._ass_color_tag(inactive_color)
        parts: list[str] = []
        for index, token in enumerate(tokens):
            text = self._escape_ass_text(token.text.strip())
            if not text:
                continue
            if index == active_index:
                parts.append(f"{active_tag}{text}{inactive_tag}")
            else:
                parts.append(text)
        return " ".join(parts)

    @staticmethod
    def _ass_color_tag(ass_color: str) -> str:
        value = ass_color.removeprefix("&H").removesuffix("&")
        if len(value) == 8:
            value = value[2:]
        return f"{{\\c&H{value}&}}"

    @staticmethod
    def _escape_ass_text(text: str) -> str:
        return (
            text.replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\r\n", r"\N")
            .replace("\n", r"\N")
            .replace("\r", r"\N")
        )

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        centiseconds = max(0, int(seconds * 100 + 0.5))
        cs = centiseconds % 100
        total_seconds = centiseconds // 100
        secs = total_seconds % 60
        mins = (total_seconds // 60) % 60
        hours = total_seconds // 3600
        return f"{hours}:{mins:02d}:{secs:02d}.{cs:02d}"

    @staticmethod
    def _hex_to_ass_color(hex_color: str) -> str:
        value = hex_color.strip().lstrip("#")
        if len(value) != 6:
            raise ValueError(f"Invalid hex color: {hex_color}")
        red, green, blue = value[0:2], value[2:4], value[4:6]
        return f"&H00{blue}{green}{red}&".upper()

    def _ass_font_name(self) -> str:
        path = self._font_path()
        font_name = self._font_name_from_file(path)
        if font_name:
            return font_name
        return self._fallback_font_name(path)

    @staticmethod
    def _font_name_from_file(path: Path) -> str | None:
        try:
            from fontTools.ttLib import TTFont  # type: ignore[import-not-found]
        except ImportError:
            return None

        try:
            font = TTFont(str(path))
            for record in font["name"].names:
                if record.nameID != 1:
                    continue
                value = record.toUnicode().strip()
                if value:
                    return value
        except Exception:
            return None
        return None

    @staticmethod
    def _fallback_font_name(path: Path) -> str:
        parts = re.split(r"[-_]", path.stem)
        if len(parts) > 1 and parts[-1].lower() in ASS_FONT_WEIGHT_SUFFIXES:
            parts = parts[:-1]
        return " ".join(part for part in parts if part).strip() or path.stem

    @staticmethod
    def _escape_ffmpeg_filter_path(path: Path) -> str:
        escaped = (
            path.resolve()
            .as_posix()
            .replace("\\", r"\\")
            .replace(":", r"\:")
            .replace("'", r"\'")
        )
        return f"'{escaped}'"
