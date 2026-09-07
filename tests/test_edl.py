from pathlib import Path

from clipmind.edl import EDLExporter
from clipmind.models import ViralClip
from clipmind.utils import seconds_to_timecode


def test_seconds_to_timecode() -> None:
    assert seconds_to_timecode(61.5, fps=30) == "00:01:01:15"


def test_edl_export_contains_source_and_record_timecodes(tmp_path: Path) -> None:
    clip = ViralClip(
        clip_id="clip_1",
        start_time=30.0,
        end_time=60.0,
        viral_score=90,
        hook_text="Great hook",
        explanation="Complete thought",
    )
    output_path = tmp_path / "timeline.edl"

    EDLExporter(fps=30).write([clip], Path("source.mp4"), output_path)

    content = output_path.read_text(encoding="utf-8")
    assert "TITLE: source" in content
    assert "00:00:30:00 00:01:00:00 00:00:00:00 00:00:30:00" in content
    assert "* HOOK: Great hook" in content
