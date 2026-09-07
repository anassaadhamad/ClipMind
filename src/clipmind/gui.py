from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

from PySide6.QtCore import QObject, QSize, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from clipmind.aligner import TextAligner
from clipmind.analyzer import AIAnalyzer
from clipmind.config import Settings, load_settings
from clipmind.edl import EDLExporter
from clipmind.models import SocialPlatform, SocialPost, ViralClip
from clipmind.social import SocialContentGenerator
from clipmind.social_widgets import SocialMediaHubWidget
from clipmind.transcriber import Transcriber
from clipmind.utils import ensure_tool, setup_logging
from clipmind.video_processor import VideoProcessor


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
SRT_SUFFIXES = {".srt"}


class SignalLogHandler(logging.Handler):
    """Forwards core pipeline logs into the Qt UI."""

    def __init__(self, signal: Signal) -> None:
        super().__init__()
        self.signal = signal
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.signal.emit(self.format(record))


class ClipMindWorker(QObject):
    """Runs OpenAI and FFmpeg work away from the GUI thread."""

    progress = Signal(int, str)
    log = Signal(str)
    clips_ready = Signal(list)
    social_ready = Signal(list)
    completed = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        *,
        action: Literal["analyze", "render", "social"],
        settings: Settings,
        video_path: Path,
        srt_path: Path | None = None,
        clip_count: int = 5,
        clips: list[ViralClip] | None = None,
        platforms: list[SocialPlatform] | None = None,
        target_clip_id: str | None = None,
        force: bool = False,
    ) -> None:
        super().__init__()
        self.action = action
        self.settings = settings
        self.video_path = video_path
        self.srt_path = srt_path
        self.clip_count = clip_count
        self.clips = clips or []
        self.platforms = platforms or ["youtube", "facebook", "tiktok"]
        self.target_clip_id = target_clip_id
        self.force = force

    @Slot()
    def run(self) -> None:
        logger = logging.getLogger("clipmind")
        handler = SignalLogHandler(self.log)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            self.settings.ensure_directories()
            ensure_tool(self.settings.ffmpeg_bin)
            ensure_tool(self.settings.ffprobe_bin)

            if self.action == "analyze":
                self._run_analysis()
            elif self.action == "render":
                self._run_render()
            else:
                self._run_social()
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            logger.removeHandler(handler)
            self.finished.emit()

    def _run_analysis(self) -> None:
        self.progress.emit(8, "Preparing transcription")
        self.log.emit(f"Loaded video: {self.video_path}")
        transcript = self._transcript()

        self.progress.emit(48, "Analyzing viral moments")
        clips = AIAnalyzer(self.settings).analyze(
            transcript,
            self.video_path,
            clip_count=self.clip_count,
            force=bool(self.srt_path),
        )

        target_output_dir = self.settings.output_dir / self.video_path.stem
        edl_path = target_output_dir / f"{self.video_path.stem}.edl"
        EDLExporter(self.settings.export_fps).write(clips, self.video_path, edl_path)
        self.progress.emit(100, "Analysis complete")
        self.clips_ready.emit(clips)
        self.completed.emit(f"Detected {len(clips)} viral clips. EDL: {edl_path}")

    def _run_render(self) -> None:
        if not self.clips:
            raise RuntimeError("No clips selected for rendering.")

        self.settings.require_font()
        transcript = self._transcript()
        output_dir = self.settings.output_dir / self.video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        processor = VideoProcessor(self.settings)
        total = len(self.clips)
        for index, clip in enumerate(self.clips, start=1):
            start_progress = int(((index - 1) / total) * 95)
            self.progress.emit(start_progress, f"Rendering {clip.clip_id}")
            self.log.emit(f"Rendering {clip.clip_id}: {clip.start_time:.2f}s - {clip.end_time:.2f}s")
            processor.process_clip(self.video_path, transcript, clip, output_dir)
            self.progress.emit(int((index / total) * 95), f"Rendered {clip.clip_id}")

        edl_path = output_dir / f"{self.video_path.stem}.edl"
        EDLExporter(self.settings.export_fps).write(self.clips, self.video_path, edl_path)
        self.progress.emit(100, "Rendering complete")
        self.completed.emit(str(output_dir))

    def _run_social(self) -> None:
        if not self.clips:
            raise RuntimeError("Analyze viral clips before generating social metadata.")

        self.progress.emit(15, "Loading transcript for social metadata")
        transcript = self._transcript()
        self.progress.emit(45, "Generating social titles and descriptions")
        bundle = SocialContentGenerator(self.settings).generate(
            transcript=transcript,
            clips=self.clips,
            video_path=self.video_path,
            platforms=self.platforms,
            force=self.force,
            target_clip_id=self.target_clip_id,
        )
        self.social_ready.emit(bundle.posts)
        self.progress.emit(100, "Social metadata ready")
        self.completed.emit(f"Generated {len(bundle.posts)} social media posts.")

    def _transcript(self):
        transcript = Transcriber(self.settings).transcribe(self.video_path, force=False)
        if self.srt_path:
            self.log.emit(f"Aligning transcript with reference subtitles: {self.srt_path}")
            transcript = TextAligner().align_from_srt(
                transcript,
                self.srt_path,
                cache_dir=self.settings.cache_dir,
                video_path=self.video_path,
            )
        return transcript


class DropArea(QFrame):
    file_dropped = Signal(Path)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("dropArea")
        self.setAcceptDrops(True)
        self.setMinimumHeight(190)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon = QLabel("Drop Video Here")
        icon.setObjectName("dropTitle")
        subtitle = QLabel("MP4, MOV, MKV, AVI, or WEBM")
        subtitle.setObjectName("mutedLabel")
        layout.addWidget(icon)
        layout.addWidget(subtitle)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if self._has_video(event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() in VIDEO_SUFFIXES:
                self.file_dropped.emit(path)
                event.acceptProposedAction()
                return

    @staticmethod
    def _has_video(urls) -> bool:
        return any(Path(url.toLocalFile()).suffix.lower() in VIDEO_SUFFIXES for url in urls)


class ClipCard(QFrame):
    process_requested = Signal(object)

    def __init__(self, clip: ViralClip) -> None:
        super().__init__()
        self.clip = clip
        self.setObjectName("card")
        self.setMinimumWidth(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        score = QLabel(str(clip.viral_score))
        score.setObjectName("scorePill")
        title = QLabel(clip.hook_text)
        title.setObjectName("cardTitle")
        title.setWordWrap(True)
        meta = QLabel(f"{clip.start_time:.1f}s - {clip.end_time:.1f}s  |  {clip.duration:.1f}s")
        meta.setObjectName("mutedLabel")
        explanation = QLabel(clip.explanation)
        explanation.setWordWrap(True)
        explanation.setObjectName("bodyLabel")
        button = QPushButton("Process")
        button.setObjectName("primaryButton")
        button.clicked.connect(lambda: self.process_requested.emit(self.clip))

        layout.addWidget(score, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)
        layout.addWidget(meta)
        layout.addWidget(explanation)
        layout.addStretch(1)
        layout.addWidget(button)


class SocialPostCard(QFrame):
    regenerate_requested = Signal(str)

    def __init__(self, post: SocialPost) -> None:
        super().__init__()
        self.post = post
        self.setObjectName("card")
        self.setMinimumWidth(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(self)
        platform = QLabel(f"{post.platform.upper()}  |  {post.clip_id}")
        platform.setObjectName("scorePill")
        title = QLabel(post.title)
        title.setObjectName("cardTitle")
        title.setWordWrap(True)
        description = QTextEdit()
        description.setObjectName("socialText")
        description.setReadOnly(True)
        description.setPlainText(post.description)
        description.setMinimumHeight(150)
        hashtags = QLabel(" ".join(post.hashtags))
        hashtags.setObjectName("mutedLabel")
        hashtags.setWordWrap(True)

        buttons = QHBoxLayout()
        copy_button = QPushButton("Copy")
        copy_button.setObjectName("secondaryButton")
        copy_button.clicked.connect(self._copy_to_clipboard)
        regenerate_button = QPushButton("Regenerate Clip")
        regenerate_button.setObjectName("primaryButton")
        regenerate_button.clicked.connect(lambda: self.regenerate_requested.emit(self.post.clip_id))
        buttons.addWidget(copy_button)
        buttons.addWidget(regenerate_button)

        layout.addWidget(platform, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(hashtags)
        if post.call_to_action:
            cta = QLabel(f"CTA: {post.call_to_action}")
            cta.setObjectName("mutedLabel")
            cta.setWordWrap(True)
            layout.addWidget(cta)
        layout.addLayout(buttons)

    def _copy_to_clipboard(self) -> None:
        text = (
            f"{self.post.title}\n\n"
            f"{self.post.description}\n\n"
            f"{' '.join(self.post.hashtags)}"
        ).strip()
        QApplication.clipboard().setText(text)


class ClipMindWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        setup_logging(False)
        self.settings = load_settings()
        self.settings.ensure_directories()
        self.selected_video: Path | None = None
        self.selected_srt: Path | None = None
        self.clips: list[ViralClip] = []
        self.social_posts: list[SocialPost] = []
        self.worker_thread: QThread | None = None
        self.worker: ClipMindWorker | None = None
        self.card_columns = 0
        self.nav_buttons: list[QPushButton] = []

        self.setWindowTitle("ClipMind Studio")
        self.resize(1320, 860)
        self.setMinimumSize(QSize(980, 680))
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        central = QWidget()
        shell = QHBoxLayout(central)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        self.setCentralWidget(central)

        shell.addWidget(self._build_sidebar())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(26, 20, 26, 18)
        right_layout.setSpacing(16)
        shell.addWidget(right, 1)

        right_layout.addWidget(self._build_header())

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.setObjectName("contentStack")
        self.stacked_widget.addWidget(self._build_dashboard_page())
        self.stacked_widget.addWidget(self._build_captions_page())
        self.stacked_widget.addWidget(self._build_render_queue_page())
        self.stacked_widget.addWidget(self._build_social_page())
        self.stacked_widget.addWidget(self._build_logs_page())
        right_layout.addWidget(self.stacked_widget, 1)
        self._navigate_to_page(0)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(14)

        logo = QLabel("ClipMind")
        logo.setObjectName("logo")
        sub = QLabel("AI Viral Studio")
        sub.setObjectName("mutedLabel")
        layout.addWidget(logo)
        layout.addWidget(sub)
        layout.addSpacing(24)

        for text in ("Dashboard", "Captions", "Render Queue", "Social Media Hub", "Logs"):
            button = QPushButton(text)
            button.setObjectName("navButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            index = len(self.nav_buttons)
            button.clicked.connect(lambda _checked=False, page_index=index: self._navigate_to_page(page_index))
            self.nav_buttons.append(button)
            layout.addWidget(button)
        layout.addStretch(1)
        version = QLabel("v0.1.0")
        version.setObjectName("mutedLabel")
        layout.addWidget(version)
        return sidebar

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setObjectName("header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 16, 20, 16)
        title_box = QVBoxLayout()
        title = QLabel("ClipMind Studio")
        title.setObjectName("headerTitle")
        self.video_label = QLabel("Choose or drop a video to begin")
        self.video_label.setObjectName("mutedLabel")
        title_box.addWidget(title)
        title_box.addWidget(self.video_label)
        layout.addLayout(title_box, 1)

        browse = QPushButton("Browse Video")
        browse.setObjectName("secondaryButton")
        browse.clicked.connect(self._browse_video)
        layout.addWidget(browse)
        return header

    def _build_scroll_page(self) -> tuple[QScrollArea, QVBoxLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setObjectName("contentScroll")
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(18)
        layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(content)
        return scroll, layout

    def _build_dashboard_page(self) -> QScrollArea:
        scroll, layout = self._build_scroll_page()
        self.drop_area = DropArea()
        self.drop_area.file_dropped.connect(self._set_video)
        layout.addWidget(self.drop_area)
        layout.addWidget(self._build_subtitle_picker_card())
        layout.addWidget(self._build_dashboard_card())
        layout.addWidget(
            self._build_placeholder_card(
                "Dashboard",
                "Detected clips appear here as viral score cards. Use each card's Process button "
                "to render a single clip, or switch to Render Queue to process everything.",
            )
        )
        layout.addStretch(1)
        return scroll

    def _build_subtitle_picker_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        title = QLabel("Reference Subtitles (.srt) [Optional]")
        title.setObjectName("sectionTitle")
        self.srt_label = QLabel("No reference subtitle file selected.")
        self.srt_label.setObjectName("mutedLabel")
        self.srt_label.setWordWrap(True)
        controls = QHBoxLayout()
        browse_srt = QPushButton("Upload Reference Subtitles (.srt)")
        browse_srt.setObjectName("secondaryButton")
        browse_srt.clicked.connect(self._browse_srt)
        clear_srt = QPushButton("Clear")
        clear_srt.setObjectName("secondaryButton")
        clear_srt.clicked.connect(self._clear_srt)
        controls.addWidget(browse_srt)
        controls.addWidget(clear_srt)
        controls.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(self.srt_label)
        layout.addLayout(controls)
        return card

    def _build_captions_page(self) -> QScrollArea:
        scroll, layout = self._build_scroll_page()
        layout.addWidget(self._build_font_card())
        layout.addWidget(
            self._build_placeholder_card(
                "Caption Preview",
                "Caption controls are applied to newly rendered clips. Adjust font, size, colors, "
                "and stroke before pressing Process.",
            )
        )
        layout.addStretch(1)
        return scroll

    def _build_render_queue_page(self) -> QScrollArea:
        scroll, layout = self._build_scroll_page()
        layout.addWidget(self._build_actions_card())
        layout.addWidget(
            self._build_placeholder_card(
                "Render Queue",
                "Use Analyze Viral Clips to populate the Dashboard. Process All renders every "
                "detected clip in a background worker so the interface stays responsive.",
            )
        )
        layout.addStretch(1)
        return scroll

    def _build_social_page(self) -> QScrollArea:
        scroll, layout = self._build_scroll_page()
        self.social_hub = SocialMediaHubWidget(self.settings)
        self.social_hub.log.connect(self._append_log)
        self.social_hub.progress.connect(self._update_progress)
        layout.addWidget(self.social_hub)
        layout.addStretch(1)
        return scroll

    def _build_logs_page(self) -> QScrollArea:
        scroll, layout = self._build_scroll_page()
        layout.addWidget(
            self._build_placeholder_card(
                "Live Logs",
                "Real-time transcription, GPT analysis, FFmpeg, and rendering status appears below.",
            )
        )
        self.log_console = QTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setMinimumHeight(360)
        layout.addWidget(self.log_console, 1)
        return scroll

    def _build_placeholder_card(self, title: str, body: str) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        body_label = QLabel(body)
        body_label.setObjectName("mutedLabel")
        body_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(body_label)
        return card

    def _build_social_settings_card(self) -> QGroupBox:
        group = QGroupBox("Social Media Settings")
        group.setObjectName("cardGroup")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        self.youtube_enabled = QCheckBox("YouTube")
        self.facebook_enabled = QCheckBox("Facebook")
        self.tiktok_enabled = QCheckBox("TikTok")
        for checkbox in (self.youtube_enabled, self.facebook_enabled, self.tiktok_enabled):
            checkbox.setChecked(True)

        self.youtube_url = QLineEdit(self.settings.youtube_channel_url)
        self.facebook_url = QLineEdit(self.settings.facebook_page_url)
        self.tiktok_url = QLineEdit(self.settings.tiktok_profile_url)
        self.youtube_description = QTextEdit(self.settings.youtube_default_description)
        self.facebook_description = QTextEdit(self.settings.facebook_default_description)
        self.tiktok_description = QTextEdit(self.settings.tiktok_default_description)
        for text_edit in (self.youtube_description, self.facebook_description, self.tiktok_description):
            text_edit.setMaximumHeight(90)

        save_button = QPushButton("Save Social Settings")
        save_button.setObjectName("secondaryButton")
        save_button.clicked.connect(self._save_social_settings)

        layout.addWidget(self.youtube_enabled, 0, 0)
        layout.addWidget(QLabel("YouTube channel URL"), 0, 1)
        layout.addWidget(self.youtube_url, 0, 2)
        layout.addWidget(QLabel("YouTube default footer"), 1, 1)
        layout.addWidget(self.youtube_description, 1, 2)

        layout.addWidget(self.facebook_enabled, 2, 0)
        layout.addWidget(QLabel("Facebook page URL"), 2, 1)
        layout.addWidget(self.facebook_url, 2, 2)
        layout.addWidget(QLabel("Facebook default footer"), 3, 1)
        layout.addWidget(self.facebook_description, 3, 2)

        layout.addWidget(self.tiktok_enabled, 4, 0)
        layout.addWidget(QLabel("TikTok profile URL"), 4, 1)
        layout.addWidget(self.tiktok_url, 4, 2)
        layout.addWidget(QLabel("TikTok default footer"), 5, 1)
        layout.addWidget(self.tiktok_description, 5, 2)
        layout.addWidget(save_button, 6, 2)
        return group

    def _build_social_actions_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        title = QLabel("Social Media Studio")
        title.setObjectName("sectionTitle")
        hint = QLabel(
            "Generate platform-native titles, descriptions, CTAs, and hashtags for every detected clip."
        )
        hint.setObjectName("mutedLabel")
        hint.setWordWrap(True)

        controls = QHBoxLayout()
        self.generate_social_button = QPushButton("Generate Social Metadata")
        self.generate_social_button.setObjectName("primaryButton")
        self.generate_social_button.setEnabled(False)
        self.generate_social_button.clicked.connect(lambda: self._start_social_generation(force=False))
        self.regenerate_social_button = QPushButton("Generate New Versions")
        self.regenerate_social_button.setObjectName("secondaryButton")
        self.regenerate_social_button.setEnabled(False)
        self.regenerate_social_button.clicked.connect(lambda: self._start_social_generation(force=True))
        controls.addWidget(self.generate_social_button)
        controls.addWidget(self.regenerate_social_button)
        controls.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addLayout(controls)
        return card

    def _build_social_results_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        title = QLabel("Generated Titles & Descriptions")
        title.setObjectName("sectionTitle")
        self.social_hint = QLabel("No social metadata generated yet.")
        self.social_hint.setObjectName("mutedLabel")
        self.social_cards_host = QWidget()
        self.social_cards_grid = QGridLayout(self.social_cards_host)
        self.social_cards_grid.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(self.social_hint)
        layout.addWidget(self.social_cards_host)
        return card

    def _build_actions_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        title = QLabel("Pipeline")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        controls = QHBoxLayout()
        self.clip_count = QSpinBox()
        self.clip_count.setRange(1, 10)
        self.clip_count.setValue(self.settings.default_clip_count)
        self.analyze_button = QPushButton("Analyze Viral Clips")
        self.analyze_button.setObjectName("primaryButton")
        self.analyze_button.clicked.connect(self._start_analysis)
        self.process_all_button = QPushButton("Process All")
        self.process_all_button.setObjectName("secondaryButton")
        self.process_all_button.setEnabled(False)
        self.process_all_button.clicked.connect(self._process_all)
        self.open_output_button = QPushButton("Open Outputs")
        self.open_output_button.setObjectName("secondaryButton")
        self.open_output_button.clicked.connect(self._open_outputs)

        controls.addWidget(QLabel("Clip count"))
        controls.addWidget(self.clip_count)
        controls.addStretch(1)
        controls.addWidget(self.analyze_button)
        controls.addWidget(self.process_all_button)
        controls.addWidget(self.open_output_button)
        layout.addLayout(controls)

        self.progress_label = QLabel("Idle")
        self.progress_label.setObjectName("mutedLabel")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)
        return card

    def _build_dashboard_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        title = QLabel("Viral Score Dashboard")
        title.setObjectName("sectionTitle")
        self.dashboard_hint = QLabel("Analyze a video to generate ranked viral clip cards.")
        self.dashboard_hint.setObjectName("mutedLabel")
        self.cards_host = QWidget()
        self.cards_grid = QGridLayout(self.cards_host)
        self.cards_grid.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(self.dashboard_hint)
        layout.addWidget(self.cards_host)
        return card

    def _build_font_card(self) -> QGroupBox:
        group = QGroupBox("Font Customizer")
        group.setObjectName("cardGroup")
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        self.font_path = QLineEdit(str(self.settings.font_path))
        font_browse = QPushButton("Choose Font")
        font_browse.setObjectName("secondaryButton")
        font_browse.clicked.connect(self._browse_font)

        self.font_size = QSpinBox()
        self.font_size.setRange(24, 160)
        self.font_size.setValue(self.settings.caption_font_size)
        self.active_color = QLineEdit(self.settings.caption_active_color)
        self.inactive_color = QLineEdit(self.settings.caption_inactive_color)
        self.stroke_color = QLineEdit(self.settings.caption_stroke_color)
        self.stroke_width = QSpinBox()
        self.stroke_width.setRange(0, 20)
        self.stroke_width.setValue(self.settings.caption_stroke_width)

        layout.addWidget(QLabel("Font path"), 0, 0)
        layout.addWidget(self.font_path, 0, 1)
        layout.addWidget(font_browse, 0, 2)
        layout.addWidget(QLabel("Font size"), 1, 0)
        layout.addWidget(self.font_size, 1, 1)
        self._add_color_row(layout, "Active word", self.active_color, 2)
        self._add_color_row(layout, "Inactive text", self.inactive_color, 3)
        self._add_color_row(layout, "Stroke", self.stroke_color, 4)
        layout.addWidget(QLabel("Stroke width"), 5, 0)
        layout.addWidget(self.stroke_width, 5, 1)
        return group

    def _build_log_console(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        self.log_toggle = QToolButton()
        self.log_toggle.setText("Show Logs")
        self.log_toggle.setCheckable(True)
        self.log_toggle.setObjectName("logToggle")
        self.log_toggle.toggled.connect(self._toggle_logs)
        self.log_console = QTextEdit()
        self.log_console.setObjectName("logConsole")
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumHeight(150)
        self.log_console.setVisible(False)
        layout.addWidget(self.log_toggle)
        layout.addWidget(self.log_console)
        return wrapper

    def _add_color_row(self, layout: QGridLayout, label: str, edit: QLineEdit, row: int) -> None:
        button = QPushButton("Pick")
        button.setObjectName("secondaryButton")
        button.clicked.connect(lambda: self._pick_color(edit))
        layout.addWidget(QLabel(label), row, 0)
        layout.addWidget(edit, row, 1)
        layout.addWidget(button, row, 2)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #1e1e1e; color: #f8fafc; font-family: Segoe UI, Arial; }
            #sidebar { background: #151515; border-right: 1px solid #2b2b2b; }
            #logo { color: #38bdf8; font-size: 28px; font-weight: 800; }
            #header, #card, QGroupBox#cardGroup { background: #252525; border: 1px solid #333333; border-radius: 18px; }
            #headerTitle { font-size: 30px; font-weight: 800; }
            #sectionTitle { font-size: 19px; font-weight: 700; color: #e2e8f0; }
            #dropArea { background: #202a33; border: 2px dashed #38bdf8; border-radius: 22px; }
            #dropTitle { font-size: 28px; font-weight: 800; color: #38bdf8; }
            #cardTitle { font-size: 18px; font-weight: 800; color: #f8fafc; }
            #bodyLabel { color: #cbd5e1; line-height: 1.4; }
            #mutedLabel { color: #94a3b8; }
            #scorePill { background: #0f2d3a; color: #38bdf8; border: 1px solid #38bdf8; border-radius: 14px; padding: 7px 12px; font-weight: 900; }
            QPushButton { border: 0; border-radius: 12px; padding: 10px 16px; font-weight: 700; }
            QPushButton#primaryButton { background: #38bdf8; color: #08111a; }
            QPushButton#primaryButton:hover { background: #7dd3fc; }
            QPushButton#secondaryButton, QPushButton#navButton, QToolButton#logToggle { background: #303030; color: #e2e8f0; }
            QPushButton#secondaryButton:hover, QPushButton#navButton:hover, QToolButton#logToggle:hover { background: #374151; color: #38bdf8; }
            QPushButton#navButton[active="true"] { background: #38bdf8; color: #08111a; }
            QPushButton:disabled { background: #2a2a2a; color: #64748b; }
            QLineEdit, QSpinBox, QTextEdit { background: #171717; color: #f8fafc; border: 1px solid #3a3a3a; border-radius: 10px; padding: 8px; }
            QCheckBox { color: #e2e8f0; font-weight: 700; }
            QProgressBar { background: #171717; border: 1px solid #333333; border-radius: 9px; height: 18px; text-align: center; }
            QProgressBar::chunk { background: #38bdf8; border-radius: 9px; }
            QScrollArea#contentScroll { border: 0; }
            QTextEdit#logConsole { color: #cbd5e1; font-family: Consolas, monospace; font-size: 12px; }
            QTextEdit#socialText { color: #e2e8f0; font-size: 13px; }
            QGroupBox { margin-top: 18px; padding: 16px; font-size: 17px; font-weight: 700; color: #38bdf8; }
            QGroupBox::title { subcontrol-origin: margin; left: 16px; padding: 0 8px; }
            """
        )

    def _navigate_to_page(self, index: int) -> None:
        self.stacked_widget.setCurrentIndex(index)
        for button_index, button in enumerate(self.nav_buttons):
            button.setProperty("active", button_index == index)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def _set_video(self, path: Path) -> None:
        self.selected_video = path
        self.video_label.setText(str(path))
        self.social_hub.set_video(path)
        self._append_log(f"Selected video: {path}")

    def _set_srt(self, path: Path) -> None:
        self.selected_srt = path
        self.srt_label.setText(str(path))
        self.social_hub.set_srt(path)
        self._append_log(f"Selected reference subtitles: {path}")

    def _browse_video(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose input video",
            str(self.settings.input_dir),
            "Videos (*.mp4 *.mov *.mkv *.avi *.webm)",
        )
        if file_name:
            self._set_video(Path(file_name))

    def _browse_srt(self) -> None:
        start_dir = str(self.selected_video.parent) if self.selected_video else str(self.settings.input_dir)
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose reference subtitles",
            start_dir,
            "SubRip subtitles (*.srt)",
        )
        if file_name:
            self._set_srt(Path(file_name))

    def _clear_srt(self) -> None:
        self.selected_srt = None
        self.srt_label.setText("No reference subtitle file selected.")
        self.social_hub.set_srt(None)
        self._append_log("Reference subtitles cleared.")

    def _browse_font(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose caption font",
            str(self.settings.font_path.parent),
            "Fonts (*.ttf *.otf)",
        )
        if file_name:
            self.font_path.setText(file_name)

    def _pick_color(self, target: QLineEdit) -> None:
        color = QColorDialog.getColor(QColor(target.text()), self, "Choose color")
        if color.isValid():
            target.setText(color.name().upper())

    def _start_analysis(self) -> None:
        if not self._require_video():
            return
        self.clips = []
        self._render_clip_cards()
        self._start_worker("analyze", clip_count=self.clip_count.value())

    def _process_all(self) -> None:
        if self.clips:
            self._start_worker("render", clips=self.clips)

    def _process_clip(self, clip: ViralClip) -> None:
        self._start_worker("render", clips=[clip])

    def _start_social_generation(self, *, force: bool, target_clip_id: str | None = None) -> None:
        if not self._require_video():
            return
        if not self.clips:
            QMessageBox.information(
                self,
                "Analyze clips first",
                "Analyze viral clips before generating platform titles and descriptions.",
            )
            return
        platforms = self._selected_social_platforms()
        if not platforms:
            QMessageBox.warning(self, "No platforms selected", "Select at least one social platform.")
            return
        self._start_worker(
            "social",
            clips=self.clips,
            platforms=platforms,
            force=force,
            target_clip_id=target_clip_id,
        )

    def _regenerate_social_clip(self, clip_id: str) -> None:
        self._start_social_generation(force=True, target_clip_id=clip_id)

    def _start_worker(
        self,
        action: Literal["analyze", "render", "social"],
        *,
        clip_count: int = 5,
        clips: list[ViralClip] | None = None,
        platforms: list[SocialPlatform] | None = None,
        target_clip_id: str | None = None,
        force: bool = False,
    ) -> None:
        if self._worker_is_running():
            QMessageBox.information(self, "ClipMind is busy", "A background job is already running.")
            return
        if not self.selected_video:
            return

        try:
            settings = self._settings_from_ui()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        self.progress.setValue(0)
        self.progress_label.setText("Starting...")
        self._set_busy(True)

        thread = QThread(self)
        worker = ClipMindWorker(
            action=action,
            settings=settings,
            video_path=self.selected_video,
            srt_path=self.selected_srt,
            clip_count=clip_count,
            clips=clips,
            platforms=platforms,
            target_clip_id=target_clip_id,
            force=force,
        )
        self.worker_thread = thread
        self.worker = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_progress)
        worker.log.connect(self._append_log)
        worker.clips_ready.connect(self._set_clips)
        worker.social_ready.connect(self._set_social_posts)
        worker.completed.connect(self._job_completed)
        worker.failed.connect(self._job_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._worker_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _worker_is_running(self) -> bool:
        if self.worker_thread is None:
            return False
        try:
            return self.worker_thread.isRunning()
        except RuntimeError:
            self.worker_thread = None
            self.worker = None
            return False

    def _worker_finished(self) -> None:
        self.worker_thread = None
        self.worker = None
        self._set_busy(False)

    def _settings_from_ui(self) -> Settings:
        data = self.settings.model_dump()
        data.update(
            {
                "font_path": Path(self.font_path.text()).expanduser(),
                "caption_font_size": self.font_size.value(),
                "caption_active_color": self.active_color.text(),
                "caption_inactive_color": self.inactive_color.text(),
                "caption_stroke_color": self.stroke_color.text(),
                "caption_stroke_width": self.stroke_width.value(),
            }
        )
        settings = Settings.model_validate(data)
        settings.ensure_directories()
        return settings

    def _set_clips(self, clips: list[ViralClip]) -> None:
        self.clips = clips
        self.process_all_button.setEnabled(bool(clips))
        self._render_clip_cards()
        self.social_hub.set_clips(clips)
        self._navigate_to_page(0)

    def _set_social_posts(self, posts: list[SocialPost]) -> None:
        self.social_posts = posts
        self._render_social_cards()
        self._navigate_to_page(3)

    def _render_clip_cards(self) -> None:
        while self.cards_grid.count():
            item = self.cards_grid.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()

        if not self.clips:
            self.dashboard_hint.setVisible(True)
            return

        self.dashboard_hint.setVisible(False)
        width = max(320, self.cards_host.width())
        columns = max(1, width // 340)
        self.card_columns = columns
        for index, clip in enumerate(self.clips):
            card = ClipCard(clip)
            card.process_requested.connect(self._process_clip)
            self.cards_grid.addWidget(card, index // columns, index % columns)

    def _render_social_cards(self) -> None:
        while self.social_cards_grid.count():
            item = self.social_cards_grid.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()

        if not self.social_posts:
            self.social_hint.setVisible(True)
            return

        self.social_hint.setVisible(False)
        columns = 2
        for index, post in enumerate(self.social_posts):
            card = SocialPostCard(post)
            card.regenerate_requested.connect(self._regenerate_social_clip)
            self.social_cards_grid.addWidget(card, index // columns, index % columns)

    def _selected_social_platforms(self) -> list[SocialPlatform]:
        platforms: list[SocialPlatform] = []
        if self.youtube_enabled.isChecked():
            platforms.append("youtube")
        if self.facebook_enabled.isChecked():
            platforms.append("facebook")
        if self.tiktok_enabled.isChecked():
            platforms.append("tiktok")
        return platforms

    def _update_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self.progress_label.setText(message)

    def _job_completed(self, message: str) -> None:
        self._append_log(message)
        self.progress_label.setText("Complete")
        if Path(message).exists():
            self.output_path = Path(message)

    def _job_failed(self, message: str) -> None:
        self._append_log(f"ERROR: {message}")
        self.progress_label.setText("Failed")
        QMessageBox.critical(self, "ClipMind error", message)

    def _set_busy(self, busy: bool) -> None:
        self.analyze_button.setEnabled(not busy)
        self.process_all_button.setEnabled((not busy) and bool(self.clips))
        self.social_hub.set_busy(busy)

    def _toggle_logs(self, checked: bool) -> None:
        self.log_console.setVisible(checked)
        self.log_toggle.setText("Hide Logs" if checked else "Show Logs")

    def _append_log(self, message: str) -> None:
        self.log_console.append(message)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def _save_social_settings(self) -> None:
        try:
            settings = self._settings_from_ui()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid social settings", str(exc))
            return
        self.settings = settings
        self._write_social_settings_to_env(settings)
        self._append_log("Social media settings saved to .env")

    def _write_social_settings_to_env(self, settings: Settings) -> None:
        env_path = Path(".env")
        updates = {
            "YOUTUBE_CHANNEL_URL": settings.youtube_channel_url,
            "YOUTUBE_DEFAULT_DESCRIPTION": settings.youtube_default_description,
            "FACEBOOK_PAGE_URL": settings.facebook_page_url,
            "FACEBOOK_DEFAULT_DESCRIPTION": settings.facebook_default_description,
            "TIKTOK_PROFILE_URL": settings.tiktok_profile_url,
            "TIKTOK_DEFAULT_DESCRIPTION": settings.tiktok_default_description,
        }
        lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
        seen: set[str] = set()
        updated_lines: list[str] = []
        for line in lines:
            key = line.split("=", 1)[0] if "=" in line and not line.lstrip().startswith("#") else None
            if key in updates:
                updated_lines.append(f"{key}={self._env_value(updates[key])}")
                seen.add(key)
            else:
                updated_lines.append(line)
        for key, value in updates.items():
            if key not in seen:
                updated_lines.append(f"{key}={self._env_value(value)}")
        env_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _env_value(value: str) -> str:
        return value.replace("\r", " ").replace("\n", "\\n")

    def _open_outputs(self) -> None:
        output = self.settings.output_dir
        if self.selected_video:
            output = self.settings.output_dir / self.selected_video.stem
        output.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(output)))

    def _require_video(self) -> bool:
        if self.selected_video and self.selected_video.exists():
            return True
        QMessageBox.warning(self, "No video selected", "Drop or browse for a video first.")
        return False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.clips:
            width = max(320, self.cards_host.width())
            columns = max(1, width // 340)
            if columns != self.card_columns:
                self._render_clip_cards()


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ClipMind Studio")
    window = ClipMindWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
