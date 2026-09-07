from __future__ import annotations

from pathlib import Path

from clipmind.models import ViralClip
from clipmind.utils import seconds_to_timecode


class EDLExporter:
    """Writes a simple CMX 3600 EDL for professional NLE import."""

    def __init__(self, fps: int = 30) -> None:
        self.fps = fps

    def write(self, clips: list[ViralClip], source_video: Path, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"TITLE: {source_video.stem}",
            "FCM: NON-DROP FRAME",
            "",
        ]

        record_cursor = 0.0
        for index, clip in enumerate(clips, start=1):
            source_in = seconds_to_timecode(clip.start_time, self.fps)
            source_out = seconds_to_timecode(clip.end_time, self.fps)
            record_in = seconds_to_timecode(record_cursor, self.fps)
            record_out = seconds_to_timecode(record_cursor + clip.duration, self.fps)
            lines.append(
                f"{index:03d}  AX       V     C        "
                f"{source_in} {source_out} {record_in} {record_out}"
            )
            lines.append(f"* FROM CLIP NAME: {source_video.name}")
            lines.append(f"* CLIP ID: {clip.clip_id}")
            lines.append(f"* HOOK: {clip.hook_text}")
            lines.append("")
            record_cursor += clip.duration

        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path
