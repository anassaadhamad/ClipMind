from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from clipmind.captions import CaptionEngine
from clipmind.config import Settings
from clipmind.models import FaceTrackPoint, ProcessedClip, Transcript, ViralClip
from clipmind.utils import clamp, get_logger, run_command, slugify, write_json


class VideoProcessor:
    """Cuts clips, performs smoothed 9:16 auto-framing, and burns captions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.logger = get_logger()
        self.caption_engine = CaptionEngine(settings)
        self._mp_detector = None
        self._haar_detector = None

    def process_clips(
        self,
        source_video: Path,
        transcript: Transcript,
        clips: list[ViralClip],
        *,
        output_dir: Path | None = None,
    ) -> list[ProcessedClip]:
        target_dir = output_dir or self.settings.output_dir / slugify(source_video.stem)
        target_dir.mkdir(parents=True, exist_ok=True)
        processed: list[ProcessedClip] = []

        for clip in tqdm(clips, desc="Rendering clips", unit="clip"):
            processed.append(self.process_clip(source_video, transcript, clip, target_dir))
        return processed

    def process_clip(
        self,
        source_video: Path,
        transcript: Transcript,
        clip: ViralClip,
        output_dir: Path,
    ) -> ProcessedClip:
        clip_slug = slugify(clip.clip_id)
        raw_segment = self.settings.temp_dir / f"{slugify(source_video.stem)}.{clip_slug}.raw.mp4"
        vertical_video = self.settings.temp_dir / f"{slugify(source_video.stem)}.{clip_slug}.vertical.mp4"
        final_video = output_dir / f"{clip_slug}.mp4"

        self.cut_segment(source_video, clip, raw_segment)
        track = self.track_faces(raw_segment)
        track_path = self.settings.cache_dir / f"{slugify(source_video.stem)}.{clip_slug}.tracking.json"
        write_json(track_path, [point.model_dump(mode="json") for point in track])

        self.render_vertical(raw_segment, vertical_video, track)
        windows = self.caption_engine.build_windows(transcript.words, clip)
        self.caption_engine.burn_captions(vertical_video, final_video, windows, clip.hook_text)

        return ProcessedClip(
            clip=clip,
            raw_segment_path=raw_segment,
            vertical_path=vertical_video,
            captioned_path=final_video,
        )

    def cut_segment(self, source_video: Path, clip: ViralClip, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration = clip.end_time - clip.start_time
        self.logger.info("Cutting %s from %.2fs to %.2fs", clip.clip_id, clip.start_time, clip.end_time)
        run_command(
            [
                self.settings.ffmpeg_bin,
                "-y",
                "-i",
                str(source_video),
                "-ss",
                f"{clip.start_time:.3f}",
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        return output_path

    def track_faces(self, video_path: Path) -> list[FaceTrackPoint]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for face tracking: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS) or self.settings.export_fps
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        sample_interval = max(1, round(fps / self.settings.tracking_sample_fps))
        points: list[FaceTrackPoint] = []

        try:
            frame_index = 0
            with tqdm(
                total=(total_frames // sample_interval) if total_frames else None,
                desc="Tracking face",
                unit="sample",
                leave=False,
            ) as progress:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if frame_index % sample_interval == 0:
                        timestamp = frame_index / fps
                        detection = self._detect_face_center(frame)
                        if detection:
                            center, confidence = detection
                            points.append(
                                FaceTrackPoint(
                                    time=timestamp,
                                    x_center=center,
                                    confidence=confidence,
                                    detected=True,
                                )
                            )
                        else:
                            points.append(FaceTrackPoint(time=timestamp, x_center=0.5))
                        progress.update(1)
                    frame_index += 1
        finally:
            capture.release()

        return self._smooth_track(points)

    def render_vertical(
        self,
        input_video: Path,
        output_video: Path,
        track: list[FaceTrackPoint],
    ) -> Path:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        silent_path = output_video.with_suffix(".silent.mp4")
        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video for auto-framing: {input_video}")

        source_fps = capture.get(cv2.CAP_PROP_FPS) or self.settings.export_fps
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        writer = cv2.VideoWriter(
            str(silent_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            source_fps,
            (self.settings.export_width, self.settings.export_height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create vertical video: {silent_path}")

        times = np.array([point.time for point in track], dtype=float) if track else np.array([0.0])
        centers = np.array([point.x_center for point in track], dtype=float) if track else np.array([0.5])

        try:
            frame_index = 0
            with tqdm(total=total_frames or None, desc="Auto-framing", unit="frame", leave=False) as progress:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp = frame_index / source_fps
                    center_x = float(np.interp(timestamp, times, centers))
                    vertical = self._crop_to_vertical(frame, center_x)
                    writer.write(vertical)
                    frame_index += 1
                    progress.update(1)
        finally:
            capture.release()
            writer.release()

        self._mux_audio(silent_path, input_video, output_video)
        silent_path.unlink(missing_ok=True)
        return output_video

    def _crop_to_vertical(self, frame, center_x: float):
        height, width = frame.shape[:2]
        target_ratio = self.settings.export_width / self.settings.export_height
        source_ratio = width / height

        if source_ratio > target_ratio:
            crop_height = height
            crop_width = int(height * target_ratio)
        else:
            crop_width = width
            crop_height = int(width / target_ratio)

        crop_width = min(crop_width, width)
        crop_height = min(crop_height, height)
        center_px = center_x * width
        left = int(clamp(center_px - crop_width / 2, 0, width - crop_width))
        top = int(clamp((height - crop_height) / 2, 0, height - crop_height))
        cropped = frame[top : top + crop_height, left : left + crop_width]
        return cv2.resize(cropped, (self.settings.export_width, self.settings.export_height))

    def _smooth_track(self, points: list[FaceTrackPoint]) -> list[FaceTrackPoint]:
        if not points:
            return [FaceTrackPoint(time=0.0, x_center=0.5)]

        detected = [point for point in points if point.detected]
        if not detected:
            return [point.model_copy(update={"x_center": 0.5}) for point in points]

        times = np.array([point.time for point in points], dtype=float)
        detected_times = np.array([point.time for point in detected], dtype=float)
        detected_centers = np.array([point.x_center for point in detected], dtype=float)
        interpolated = np.interp(times, detected_times, detected_centers)

        window = min(self.settings.smoothing_window, len(interpolated))
        if window > 1:
            kernel = np.ones(window) / window
            padded = np.pad(interpolated, (window // 2, window // 2), mode="edge")
            smoothed = np.convolve(padded, kernel, mode="valid")
        else:
            smoothed = interpolated

        return [
            point.model_copy(update={"x_center": float(clamp(center, 0, 1))})
            for point, center in zip(points, smoothed)
        ]

    def _detect_face_center(self, frame) -> tuple[float, float] | None:
        mp_detection = self._detect_with_mediapipe(frame)
        if mp_detection:
            return mp_detection
        return self._detect_with_haar(frame)

    def _detect_with_mediapipe(self, frame) -> tuple[float, float] | None:
        try:
            import mediapipe as mp
        except ImportError:
            return None

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
            return None

        if self._mp_detector is None:
            self._mp_detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._mp_detector.process(rgb)
        if not result.detections:
            return None

        detection = max(result.detections, key=lambda item: item.score[0] if item.score else 0)
        box = detection.location_data.relative_bounding_box
        center_x = box.xmin + box.width / 2
        confidence = detection.score[0] if detection.score else 0.5
        return float(clamp(center_x, 0, 1)), float(clamp(confidence, 0, 1))

    def _detect_with_haar(self, frame) -> tuple[float, float] | None:
        if self._haar_detector is None:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._haar_detector = cv2.CascadeClassifier(str(cascade_path))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None

        x, _y, width, height = max(faces, key=lambda face: face[2] * face[3])
        center_x = (x + width / 2) / frame.shape[1]
        return float(clamp(center_x, 0, 1)), 0.5

    def _mux_audio(self, silent_video: Path, audio_source: Path, output_video: Path) -> None:
        run_command(
            [
                self.settings.ffmpeg_bin,
                "-y",
                "-i",
                str(silent_video),
                "-i",
                str(audio_source),
                "-map",
                "0:v:0",
                "-map",
                "1:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-shortest",
                "-movflags",
                "+faststart",
                str(output_video),
            ]
        )
