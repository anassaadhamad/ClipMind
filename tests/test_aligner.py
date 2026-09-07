from pathlib import Path

from clipmind.aligner import TextAligner
from clipmind.models import Transcript, TranscriptWord


def test_parse_srt_extracts_clean_text(tmp_path: Path) -> None:
    srt = tmp_path / "ref.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,000\n<font color=\"red\">مرحبا</font> يا جماعة\n\n"
        "2\n00:00:01,000 --> 00:00:02,000\nالخوارزمي وصل\n",
        encoding="utf-8",
    )

    cues = TextAligner().parse_srt(srt)

    assert len(cues) == 2
    assert TextAligner.cues_to_text(cues) == "مرحبا يا جماعة الخوارزمي وصل"


def test_align_replaces_wrong_whisper_words_and_preserves_timestamps() -> None:
    transcript = Transcript(
        source_video="video.mp4",
        text="مرحبا يا جماعه الخوارزني وصل",
        words=[
            TranscriptWord(text="مرحبا", start=0.0, end=0.3),
            TranscriptWord(text="يا", start=0.3, end=0.5),
            TranscriptWord(text="جماعه", start=0.5, end=0.9),
            TranscriptWord(text="الخوارزني", start=0.9, end=1.4),
            TranscriptWord(text="وصل", start=1.4, end=1.8),
        ],
    )

    aligned = TextAligner().align_words(
        transcript.words,
        ["مرحبا", "يا", "جماعة", "الخوارزمي", "وصل"],
    )

    assert [word.text for word in aligned] == ["مرحبا", "يا", "جماعة", "الخوارزمي", "وصل"]
    assert [(word.start, word.end) for word in aligned][3] == (0.9, 1.4)


def test_align_interpolates_one_whisper_word_to_multiple_srt_words() -> None:
    whisper_words = [
        TranscriptWord(text="السلامعليكم", start=0.0, end=1.0),
        TranscriptWord(text="جماعة", start=1.0, end=1.5),
    ]

    aligned = TextAligner().align_words(whisper_words, ["السلام", "عليكم", "جماعة"])

    assert [word.text for word in aligned] == ["السلام", "عليكم", "جماعة"]
    assert aligned[0].start == 0.0
    assert aligned[0].end == 0.5
    assert aligned[1].start == 0.5
    assert aligned[1].end == 1.0
    assert aligned[2].start == 1.0
    assert aligned[2].end == 1.5


def test_align_from_srt_uses_srt_text_as_transcript_text(tmp_path: Path) -> None:
    srt = tmp_path / "ref.srt"
    srt.write_text(
        "1\n00:00:00,000 --> 00:00:01,500\nالنص الصحيح هنا\n",
        encoding="utf-8",
    )
    transcript = Transcript(
        source_video="video.mp4",
        text="النص الغلط هنا",
        words=[
            TranscriptWord(text="النص", start=0.0, end=0.4),
            TranscriptWord(text="الغلط", start=0.4, end=0.9),
            TranscriptWord(text="هنا", start=0.9, end=1.5),
        ],
    )

    aligned = TextAligner().align_from_srt(transcript, srt)

    assert aligned.text == "النص الصحيح هنا"
    assert [word.text for word in aligned.words] == ["النص", "الصحيح", "هنا"]
    assert aligned.words[1].start == 0.4
    assert aligned.words[1].end == 0.9
