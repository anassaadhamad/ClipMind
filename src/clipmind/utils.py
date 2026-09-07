from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


LOGGER_NAME = "clipmind"


def setup_logging(verbose: bool = False) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(LOGGER_NAME)


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def run_command(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    logger = get_logger()
    logger.debug("Running command: %s", " ".join(command))
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            "Command failed with exit code "
            f"{result.returncode}: {' '.join(command)}\n{result.stderr.strip()}"
        )
    return result


def ensure_tool(binary: str) -> None:
    try:
        run_command([binary, "-version"], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required executable not found on PATH: {binary}") from exc


def ffprobe_duration(ffprobe_bin: str, video_path: Path) -> float:
    result = run_command(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return float(result.stdout.strip())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def slugify(value: str, fallback: str = "clip") -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip().lower()).strip("-._")
    return slug or fallback


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]


def seconds_to_timecode(seconds: float, fps: int = 30) -> str:
    total_frames = max(0, round(seconds * fps))
    frames = total_frames % fps
    total_seconds = total_frames // fps
    secs = total_seconds % 60
    mins = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}:{frames:02d}"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def resolve_input_video(input_dir: Path, video: Path) -> Path:
    candidate = video.expanduser()
    if candidate.exists():
        return candidate.resolve()

    project_candidate = input_dir / video
    if project_candidate.exists():
        return project_candidate.resolve()

    raise FileNotFoundError(f"Input video not found: {video}")
