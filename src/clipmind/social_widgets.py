from __future__ import annotations

from pathlib import Path
from typing import cast

from PySide6.QtCore import QObject, QSettings, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from clipmind.aligner import TextAligner
from clipmind.config import Settings
from clipmind.models import SocialPlatform, SocialPost, ViralClip
from clipmind.social import SocialContentGenerator
from clipmind.transcriber import Transcriber


SETTINGS_ORG = "ClipMind"
SETTINGS_APP = "ClipMindStudio"


class SocialMediaSettingsWidget(QGroupBox):
    """Stores creator social profile settings with QSettings."""

    saved = Signal(dict)

    def __init__(self) -> None:
        super().__init__("Platform Settings")
        self.setObjectName("cardGroup")
        self.qsettings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self._build_ui()
        self.load()

    def _build_ui(self) -> None:
        layout = QGridLayout(self)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)

        self.youtube_channel_link = QLineEdit()
        self.facebook_page_link = QLineEdit()
        self.tiktok_profile_link = QLineEdit()
        self.instagram_profile_link = QLineEdit()
        self.default_cta_text = QLineEdit()
        self.global_hashtags = QLineEdit()

        rows = [
            ("YouTube Channel Link", self.youtube_channel_link),
            ("Facebook Page Link", self.facebook_page_link),
            ("TikTok Profile Link", self.tiktok_profile_link),
            ("Instagram Profile Link", self.instagram_profile_link),
            ("Default CTA Text", self.default_cta_text),
            ("Global Hashtags", self.global_hashtags),
        ]
        for row, (label, widget) in enumerate(rows):
            layout.addWidget(QLabel(label), row, 0)
            layout.addWidget(widget, row, 1)

        self.default_cta_text.setPlaceholderText("Subscribe for more content!")
        self.global_hashtags.setPlaceholderText("#ClipMind, #Shorts, #Viral")

        save_button = QPushButton("Save Platform Settings")
        save_button.setObjectName("primaryButton")
        save_button.clicked.connect(self.save)
        layout.addWidget(save_button, len(rows), 1)

    def load(self) -> None:
        for key, widget in self._fields().items():
            value = self.qsettings.value(key, "")
            widget.setText("" if value is None else str(value))

    def save(self) -> None:
        values = self.values()
        for key, value in values.items():
            self.qsettings.setValue(key, value)
        self.qsettings.sync()
        self.saved.emit(values)

    def values(self) -> dict[str, str]:
        return {
            key: widget.text().strip()
            for key, widget in self._fields().items()
        }

    def missing_required_links(self, platform: SocialPlatform) -> list[str]:
        values = self.values()
        required_by_platform = {
            "youtube": ["youtube_channel_link"],
            "facebook": ["facebook_page_link"],
            "tiktok": ["tiktok_profile_link"],
        }
        return [key for key in required_by_platform[platform] if not values.get(key)]

    def _fields(self) -> dict[str, QLineEdit]:
        return {
            "youtube_channel_link": self.youtube_channel_link,
            "facebook_page_link": self.facebook_page_link,
            "tiktok_profile_link": self.tiktok_profile_link,
            "instagram_profile_link": self.instagram_profile_link,
            "default_cta_text": self.default_cta_text,
            "global_hashtags": self.global_hashtags,
        }


class PlatformStrategyTab(QWidget):
    generate_requested = Signal(str, bool)
    regenerate_requested = Signal(str, bool)

    def __init__(self, platform: SocialPlatform) -> None:
        super().__init__()
        self.platform = platform
        self.current_post: SocialPost | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)

        self.clip_card = QFrame()
        self.clip_card.setObjectName("card")
        clip_layout = QVBoxLayout(self.clip_card)
        self.clip_title = QLabel("No clip selected")
        self.clip_title.setObjectName("sectionTitle")
        self.clip_meta = QLabel("Analyze clips first, then choose a clip.")
        self.clip_meta.setObjectName("mutedLabel")
        self.clip_meta.setWordWrap(True)
        clip_layout.addWidget(self.clip_title)
        clip_layout.addWidget(self.clip_meta)
        layout.addWidget(self.clip_card)

        self.generate_button = QPushButton("Generate Strategy")
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(lambda: self.generate_requested.emit(self.platform, False))
        layout.addWidget(self.generate_button)

        titles_card = QFrame()
        titles_card.setObjectName("card")
        titles_layout = QVBoxLayout(titles_card)
        titles_label = QLabel("Catchy Titles")
        titles_label.setObjectName("sectionTitle")
        title_controls = QHBoxLayout()
        self.titles_combo = QComboBox()
        self.regenerate_button = QPushButton("↻ Regenerate")
        self.regenerate_button.setObjectName("secondaryButton")
        self.regenerate_button.clicked.connect(lambda: self.regenerate_requested.emit(self.platform, True))
        self.copy_title_button = QPushButton("Copy Title")
        self.copy_title_button.setObjectName("secondaryButton")
        self.copy_title_button.clicked.connect(self.copy_title)
        title_controls.addWidget(self.titles_combo, 1)
        title_controls.addWidget(self.regenerate_button)
        title_controls.addWidget(self.copy_title_button)
        titles_layout.addWidget(titles_label)
        titles_layout.addLayout(title_controls)
        layout.addWidget(titles_card)

        description_card = QFrame()
        description_card.setObjectName("card")
        description_layout = QVBoxLayout(description_card)
        description_label = QLabel("Generated Description")
        description_label.setObjectName("sectionTitle")
        self.description = QTextEdit()
        self.description.setObjectName("socialText")
        self.description.setMinimumHeight(260)
        self.copy_description_button = QPushButton("Copy Description")
        self.copy_description_button.setObjectName("secondaryButton")
        self.copy_description_button.clicked.connect(self.copy_description)
        description_layout.addWidget(description_label)
        description_layout.addWidget(self.description)
        description_layout.addWidget(self.copy_description_button)
        layout.addWidget(description_card)
        layout.addStretch(1)

    def set_clip(self, clip: ViralClip | None) -> None:
        if clip is None:
            self.clip_title.setText("No clip selected")
            self.clip_meta.setText("Analyze clips first, then choose a clip.")
            return
        self.clip_title.setText(f"{clip.clip_id} | Score {clip.viral_score}")
        self.clip_meta.setText(
            f"{clip.start_time:.1f}s - {clip.end_time:.1f}s | {clip.hook_text}\n{clip.explanation}"
        )

    def set_post(self, post: SocialPost) -> None:
        self.current_post = post
        self.titles_combo.clear()
        self.titles_combo.addItems(post.titles[:3])
        self.description.setPlainText(post.description)

    def copy_title(self) -> None:
        QApplication.clipboard().setText(self.titles_combo.currentText())

    def copy_description(self) -> None:
        QApplication.clipboard().setText(self.description.toPlainText())


class SocialStrategyWorker(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    result = Signal(object)
    failed = Signal(str)
    finished = Signal()

    def __init__(
        self,
        *,
        settings: Settings,
        video_path: Path,
        srt_path: Path | None,
        clip: ViralClip,
        platform: SocialPlatform,
        social_settings: dict[str, str],
        force: bool,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.video_path = video_path
        self.srt_path = srt_path
        self.clip = clip
        self.platform = platform
        self.social_settings = social_settings
        self.force = force

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(20, f"Loading transcript for {self.platform}")
            transcript = Transcriber(self.settings).transcribe(self.video_path, force=False)
            if self.srt_path:
                transcript = TextAligner().align_from_srt(
                    transcript,
                    self.srt_path,
                    cache_dir=self.settings.cache_dir,
                    video_path=self.video_path,
                )
            self.progress.emit(55, f"Generating {self.platform} strategy")
            variation = (
                "Regenerate with a different psychological angle, fresh wording, and new hooks."
                if self.force
                else "Create the strongest first-pass strategy."
            )
            post = SocialContentGenerator(self.settings).generate_strategy(
                transcript=transcript,
                clip=self.clip,
                video_path=self.video_path,
                platform=self.platform,
                social_settings=self.social_settings,
                force=self.force,
                temperature=0.85 if self.force else 0.65,
                prompt_variation=variation,
            )
            self.progress.emit(100, f"{self.platform} strategy ready")
            self.result.emit(post)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            self.finished.emit()


class SocialMediaHubWidget(QWidget):
    """Premium tabbed social media strategy hub."""

    progress = Signal(int, str)
    log = Signal(str)

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self.video_path: Path | None = None
        self.srt_path: Path | None = None
        self.clips: list[ViralClip] = []
        self.posts: dict[tuple[SocialPlatform, str], SocialPost] = {}
        self.worker_thread: QThread | None = None
        self.worker: SocialStrategyWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(18)
        self.settings_widget = SocialMediaSettingsWidget()
        self.settings_widget.saved.connect(lambda _values: self.log.emit("Social media QSettings saved."))
        layout.addWidget(self.settings_widget)

        selector_card = QFrame()
        selector_card.setObjectName("card")
        selector_layout = QHBoxLayout(selector_card)
        title = QLabel("Selected Clip")
        title.setObjectName("sectionTitle")
        self.clip_selector = QComboBox()
        self.clip_selector.currentIndexChanged.connect(self._selected_clip_changed)
        selector_layout.addWidget(title)
        selector_layout.addWidget(self.clip_selector, 1)
        layout.addWidget(selector_card)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("socialTabs")
        self.platform_tabs: dict[SocialPlatform, PlatformStrategyTab] = {}
        for platform, label in (("youtube", "YouTube"), ("tiktok", "TikTok"), ("facebook", "Facebook")):
            tab = PlatformStrategyTab(platform)
            tab.generate_requested.connect(self.generate_strategy)
            tab.regenerate_requested.connect(self.generate_strategy)
            self.platform_tabs[platform] = tab
            self.tabs.addTab(tab, label)
        layout.addWidget(self.tabs, 1)
        self.set_busy(False)

    def set_video(self, path: Path | None) -> None:
        self.video_path = path

    def set_srt(self, path: Path | None) -> None:
        self.srt_path = path

    def set_clips(self, clips: list[ViralClip]) -> None:
        self.clips = clips
        self.clip_selector.blockSignals(True)
        self.clip_selector.clear()
        for clip in clips:
            self.clip_selector.addItem(f"{clip.clip_id} | {clip.hook_text}", clip.clip_id)
        self.clip_selector.blockSignals(False)
        self._selected_clip_changed()
        self.set_busy(False)

    def set_busy(self, busy: bool) -> None:
        for tab in self.platform_tabs.values():
            tab.generate_button.setEnabled((not busy) and bool(self.clips))
            tab.regenerate_button.setEnabled((not busy) and bool(self.clips))

    def generate_strategy(self, platform_value: str, force: bool) -> None:
        platform = cast(SocialPlatform, platform_value)
        clip = self.current_clip()
        if clip is None:
            QMessageBox.information(self, "No clip selected", "Analyze clips and select a clip first.")
            return
        if self.video_path is None:
            QMessageBox.information(self, "No video selected", "Choose a video before generating strategy.")
            return
        missing = self.settings_widget.missing_required_links(platform)
        if missing:
            QMessageBox.warning(
                self,
                "Platform settings needed",
                "Please fill and save this field first:\n" + "\n".join(missing),
            )
            return
        if self._worker_is_running():
            QMessageBox.information(self, "Social Media Hub is busy", "A strategy is already being generated.")
            return

        try:
            self.settings.require_openai_key()
        except Exception as exc:
            QMessageBox.warning(self, "OpenAI key needed", str(exc))
            return
        thread = QThread(self)
        worker = SocialStrategyWorker(
            settings=self.settings,
            video_path=self.video_path,
            srt_path=self.srt_path,
            clip=clip,
            platform=platform,
            social_settings=self.settings_widget.values(),
            force=force,
        )
        self.worker_thread = thread
        self.worker = worker
        self.set_busy(True)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress)
        worker.log.connect(self.log)
        worker.result.connect(self._set_post)
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._worker_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def current_clip(self) -> ViralClip | None:
        clip_id = self.clip_selector.currentData()
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        return self.clips[0] if self.clips else None

    def _selected_clip_changed(self) -> None:
        clip = self.current_clip()
        for platform, tab in self.platform_tabs.items():
            tab.set_clip(clip)
            if clip:
                post = self.posts.get((platform, clip.clip_id))
                if post:
                    tab.set_post(post)

    def _set_post(self, post: SocialPost) -> None:
        self.posts[(post.platform, post.clip_id)] = post
        self.platform_tabs[post.platform].set_post(post)
        self.log.emit(f"{post.platform} strategy generated for {post.clip_id}.")

    def _failed(self, message: str) -> None:
        self.log.emit(f"ERROR: {message}")
        QMessageBox.critical(self, "Social generation failed", message)

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
        self.set_busy(False)
