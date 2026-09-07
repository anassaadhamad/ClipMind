from clipmind.models import Transcript, TranscriptWord


def test_transcript_words_for_range_includes_overlapping_words() -> None:
    transcript = Transcript(
        source_video="video.mp4",
        words=[
            TranscriptWord(text="before", start=0.0, end=0.9),
            TranscriptWord(text="overlap", start=0.9, end=1.2),
            TranscriptWord(text="inside", start=1.3, end=1.8),
            TranscriptWord(text="after", start=2.1, end=2.5),
        ],
    )

    words = transcript.words_for_range(1.0, 2.0)

    assert [word.text for word in words] == ["overlap", "inside"]
