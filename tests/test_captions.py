from clipmind.captions import CaptionEngine, FONT_PATH, SFX_POP_PATH
from clipmind.config import DEFAULT_FONT_PATH, Settings
from clipmind.models import CaptionToken, CaptionWindow, TranscriptWord, ViralClip


def test_caption_windows_use_relative_clip_times() -> None:
    engine = CaptionEngine(Settings(openai_api_key="", caption_max_words=3, caption_min_words=2))
    words = [
        TranscriptWord(text="one", start=10.0, end=10.3),
        TranscriptWord(text="two", start=10.4, end=10.8),
        TranscriptWord(text="three", start=11.0, end=11.4),
        TranscriptWord(text="four", start=11.5, end=11.9),
        TranscriptWord(text="five", start=12.0, end=12.4),
    ]
    clip = ViralClip(
        clip_id="clip_1",
        start_time=10.0,
        end_time=40.0,
        viral_score=95,
        hook_text="Hook",
        explanation="Reason",
    )

    windows = engine.build_windows(words, clip)

    assert len(windows) == 2
    assert [token.text for token in windows[0].tokens] == ["one", "two", "three"]
    assert windows[0].start == 0.0
    assert windows[1].end == 2.4


def test_caption_windows_keep_tail_when_merge_would_exceed_max_words() -> None:
    engine = CaptionEngine(Settings(openai_api_key="", caption_max_words=4, caption_min_words=2))
    words = [
        TranscriptWord(text="one", start=0.0, end=0.2),
        TranscriptWord(text="two", start=0.3, end=0.5),
        TranscriptWord(text="three", start=0.6, end=0.8),
        TranscriptWord(text="four", start=0.9, end=1.1),
        TranscriptWord(text="five", start=1.2, end=1.4),
    ]
    clip = ViralClip(
        clip_id="clip_1",
        start_time=0.0,
        end_time=30.0,
        viral_score=95,
        hook_text="Hook",
        explanation="Reason",
    )

    windows = engine.build_windows(words, clip)

    assert len(windows) == 2
    assert len(windows[-1].tokens) == 1


def test_default_caption_font_path_is_cairo_bold() -> None:
    assert FONT_PATH == DEFAULT_FONT_PATH
    assert str(FONT_PATH).replace("\\", "/") == "assets/fonts/Cairo-Black.ttf"


def test_sfx_start_times_are_unique_and_sorted() -> None:
    windows = [
        CaptionWindow(
            start=0.0,
            end=1.0,
            tokens=[
                CaptionToken(text="one", start=0.2, end=0.3),
                CaptionToken(text="two", start=0.1, end=0.2),
                CaptionToken(text="again", start=0.2, end=0.4),
            ],
        )
    ]

    assert CaptionEngine._sfx_start_times(windows) == [0.1, 0.2]


def test_default_sfx_path() -> None:
    assert str(SFX_POP_PATH).replace("\\", "/") == "assets/sfx/pop.mp3"


def test_ass_time_format_uses_centiseconds() -> None:
    assert CaptionEngine._format_ass_time(0) == "0:00:00.00"
    assert CaptionEngine._format_ass_time(62.345) == "0:01:02.35"


def test_hex_to_ass_color_uses_ass_bbggrr_order() -> None:
    assert CaptionEngine._hex_to_ass_color("#FFD400") == "&H0000D4FF&"
    assert CaptionEngine._hex_to_ass_color("#FFFFFF") == "&H00FFFFFF&"


def test_generate_ass_text_keeps_raw_arabic_and_highlights_words(tmp_path) -> None:
    font_path = tmp_path / "Cairo-Black.ttf"
    font_path.write_bytes(b"not-a-real-font-but-good-enough-for-ass-tests")
    engine = CaptionEngine(
        Settings(
            openai_api_key="",
            font_path=font_path,
            caption_font_size=78,
            caption_active_color="#FFD400",
            caption_inactive_color="#FFFFFF",
        )
    )
    windows = [
        CaptionWindow(
            start=0.0,
            end=0.8,
            tokens=[
                CaptionToken(text="يا", start=0.0, end=0.25),
                CaptionToken(text="جماعة", start=0.25, end=0.55),
                CaptionToken(text="إيه", start=0.55, end=0.8),
            ],
        )
    ]

    ass_text = engine.generate_ass_text(windows, "Hook", duration=1.5)

    assert "Style: Caption,Cairo,78" in ass_text
    assert "Dialogue: 1," not in ass_text
    assert r"\alpha" not in ass_text
    assert (
        "Dialogue: 0,0:00:00.00,0:00:00.25,Caption,,0,0,0,,"
        "\u202B{\\c&H00D4FF&}يا{\\c&HFFFFFF&} جماعة إيه\u202C"
    ) in ass_text
    assert (
        "Dialogue: 0,0:00:00.25,0:00:00.55,Caption,,0,0,0,,"
        "\u202Bيا {\\c&H00D4FF&}جماعة{\\c&HFFFFFF&} إيه\u202C"
    ) in ass_text
    assert (
        "Dialogue: 0,0:00:00.55,0:00:00.80,Caption,,0,0,0,,"
        "\u202Bيا جماعة {\\c&H00D4FF&}إيه{\\c&HFFFFFF&}\u202C"
    ) in ass_text
    assert "Dialogue: 0,0:00:00.00,0:00:01.50,Hook,,0,0,0,,Hook" in ass_text


def test_generate_ass_text_keeps_ltr_word_order_inline(tmp_path) -> None:
    font_path = tmp_path / "Cairo-Black.ttf"
    font_path.write_bytes(b"font")
    engine = CaptionEngine(
        Settings(
            openai_api_key="",
            font_path=font_path,
            caption_active_color="#FFD400",
            caption_inactive_color="#FFFFFF",
        )
    )
    windows = [
        CaptionWindow(
            start=0.0,
            end=0.6,
            tokens=[
                CaptionToken(text="hello", start=0.0, end=0.2),
                CaptionToken(text="brave", start=0.2, end=0.4),
                CaptionToken(text="world", start=0.4, end=0.6),
            ],
        )
    ]

    ass_text = engine.generate_ass_text(windows, "", duration=1.0)

    assert (
        "Dialogue: 0,0:00:00.20,0:00:00.40,Caption,,0,0,0,,"
        "hello {\\c&H00D4FF&}brave{\\c&HFFFFFF&} world"
    ) in ass_text
    assert "\u202B" not in ass_text
    assert "\u202C" not in ass_text


def test_generate_ass_text_wraps_arabic_hook_with_bidi_marks(tmp_path) -> None:
    font_path = tmp_path / "Cairo-Black.ttf"
    font_path.write_bytes(b"font")
    engine = CaptionEngine(Settings(openai_api_key="", font_path=font_path))

    ass_text = engine.generate_ass_text(
        [],
        "قط شيرازي يمسحلي الداتا",
        duration=5.0,
    )

    assert (
        "Dialogue: 0,0:00:00.00,0:00:05.00,Hook,,0,0,0,,"
        "\u202Bقط شيرازي يمسحلي الداتا\u202C"
    ) in ass_text


def test_generate_ass_file_writes_utf8_sig_file(tmp_path) -> None:
    font_path = tmp_path / "Cairo-Black.ttf"
    font_path.write_bytes(b"font")
    engine = CaptionEngine(Settings(openai_api_key="", font_path=font_path))
    output_path = tmp_path / "captions.ass"

    engine.generate_ass_file([], "", output_path, duration=1.0)

    assert output_path.read_bytes().startswith(b"\xef\xbb\xbf")
    assert "[Script Info]" in output_path.read_text(encoding="utf-8-sig")
