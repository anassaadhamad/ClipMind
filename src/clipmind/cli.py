from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from clipmind.config import load_settings
from clipmind.utils import ensure_tool, resolve_input_video, setup_logging


app = typer.Typer(help="Extract viral short-form clips from long-form videos.")
console = Console()

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def process(
    video: Path = typer.Argument(..., help="Source video path or filename inside INPUT_DIR."),
    clips: int | None = typer.Option(None, "--clips", "-n", help="Number of viral clips to request."),
    output_dir: Path | None = typer.Option(None, "--output-dir", "-o", help="Override output directory."),
    force_transcript: bool = typer.Option(False, help="Ignore cached Whisper transcript."),
    force_analysis: bool = typer.Option(False, help="Ignore cached GPT clip analysis."),
    dry_run: bool = typer.Option(False, help="Analyze clips but skip rendering."),
    srt: Path | None = typer.Option(None, "--srt", help="Optional reference .srt subtitles for alignment."),
    env_file: Path = typer.Option(Path(".env"), help="Path to .env file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs."),
) -> None:
    """Run the full transcription, analysis, rendering, and EDL pipeline."""

    logger = setup_logging(verbose)
    settings = load_settings(env_file)
    settings.ensure_directories()
    _check_external_tools(settings)

    try:
        source_video = resolve_input_video(settings.input_dir, video)

        from clipmind.aligner import TextAligner
        from clipmind.analyzer import AIAnalyzer
        from clipmind.edl import EDLExporter
        from clipmind.transcriber import Transcriber
        from clipmind.video_processor import VideoProcessor

        transcript = Transcriber(settings).transcribe(source_video, force=force_transcript)
        if srt:
            transcript = TextAligner().align_from_srt(
                transcript,
                srt,
                cache_dir=settings.cache_dir,
                video_path=source_video,
            )
        selected_clips = AIAnalyzer(settings).analyze(
            transcript,
            source_video,
            clip_count=clips,
            force=force_analysis or bool(srt),
        )
        _print_clip_table(selected_clips)

        target_output_dir = output_dir or settings.output_dir / source_video.stem
        edl_path = target_output_dir / f"{source_video.stem}.edl"
        EDLExporter(settings.export_fps).write(selected_clips, source_video, edl_path)

        if dry_run:
            console.print(f"[yellow]Dry run complete.[/] EDL written to {edl_path}")
            return

        settings.require_font()
        processed = VideoProcessor(settings).process_clips(
            source_video,
            transcript,
            selected_clips,
            output_dir=target_output_dir,
        )
        console.print(f"[green]Rendered {len(processed)} clips to {target_output_dir}[/]")
        console.print(f"[green]EDL written to {edl_path}[/]")
    except Exception as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc


@app.command()
def transcribe(
    video: Path = typer.Argument(..., help="Source video path or filename inside INPUT_DIR."),
    force: bool = typer.Option(False, help="Ignore cached transcript."),
    srt: Path | None = typer.Option(None, "--srt", help="Optional reference .srt subtitles for alignment."),
    env_file: Path = typer.Option(Path(".env"), help="Path to .env file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs."),
) -> None:
    """Extract audio and cache a Whisper word-level transcript."""

    setup_logging(verbose)
    settings = load_settings(env_file)
    settings.ensure_directories()
    _check_external_tools(settings)

    from clipmind.aligner import TextAligner
    from clipmind.transcriber import Transcriber

    source_video = resolve_input_video(settings.input_dir, video)
    transcript = Transcriber(settings).transcribe(source_video, force=force)
    if srt:
        transcript = TextAligner().align_from_srt(
            transcript,
            srt,
            cache_dir=settings.cache_dir,
            video_path=source_video,
        )
    console.print(f"[green]Transcript ready:[/] {len(transcript.words)} words")


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Source video path or filename inside INPUT_DIR."),
    clips: int | None = typer.Option(None, "--clips", "-n", help="Number of viral clips to request."),
    force: bool = typer.Option(False, help="Ignore cached analysis."),
    srt: Path | None = typer.Option(None, "--srt", help="Optional reference .srt subtitles for alignment."),
    env_file: Path = typer.Option(Path(".env"), help="Path to .env file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logs."),
) -> None:
    """Use a cached transcript to select viral clips without rendering."""

    setup_logging(verbose)
    settings = load_settings(env_file)
    settings.ensure_directories()

    from clipmind.aligner import TextAligner
    from clipmind.analyzer import AIAnalyzer
    from clipmind.transcriber import Transcriber

    source_video = resolve_input_video(settings.input_dir, video)
    transcript = Transcriber(settings).transcribe(source_video, force=False)
    if srt:
        transcript = TextAligner().align_from_srt(
            transcript,
            srt,
            cache_dir=settings.cache_dir,
            video_path=source_video,
        )
    selected_clips = AIAnalyzer(settings).analyze(transcript, source_video, clip_count=clips, force=force or bool(srt))
    _print_clip_table(selected_clips)


@app.command()
def gui() -> None:
    """Launch the premium desktop GUI."""

    from clipmind.gui import main

    main()


def _check_external_tools(settings) -> None:
    ensure_tool(settings.ffmpeg_bin)
    ensure_tool(settings.ffprobe_bin)


def _print_clip_table(clips) -> None:
    table = Table(title="Selected Viral Clips")
    table.add_column("ID")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Hook")

    for clip in clips:
        table.add_row(
            clip.clip_id,
            f"{clip.start_time:.2f}s",
            f"{clip.end_time:.2f}s",
            str(clip.viral_score),
            clip.hook_text,
        )
    console.print(table)


if __name__ == "__main__":
    app()
