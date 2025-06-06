"""
FFmpegManager class for SRT Generator Phase 1 refactoring.

This class handles audio extraction and normalization using FFmpeg.
"""

import subprocess
import logging
from typing import List, Tuple

class FFmpegError(Exception):
    """Error during FFmpeg execution."""
    pass

class FFmpegManager:
    def __init__(self, config: dict, logger: logging.Logger):
        self.ffmpeg_path = config.get("FFMPEG_PATH", "ffmpeg")
        # ffprobe_path stored but validation happens in AvtPipeline._validate_config
        self.ffprobe_path = config.get("FFPROBE_PATH", "ffprobe")
        self.config = config
        self.logger = logger

    def _run_ffmpeg_command(self, command: List[str]) -> Tuple[str, str]:
        """Executes an FFmpeg command, raises FFmpegError on failure."""
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                stderr_snippet = (stderr[:500] + '...') if len(stderr) > 500 else stderr
                # Raise specific error
                raise FFmpegError(f"FFmpeg command failed. RC={process.returncode}. Path={command[0]}. Stderr: {stderr_snippet}")

            self.logger.info(f"FFmpeg command executed successfully: {' '.join(command)}")
            self.logger.debug(f"FFmpeg stdout: {stdout}")
            if stderr: self.logger.debug(f"FFmpeg stderr: {stderr}")
            return stdout, stderr

        except FileNotFoundError as e:
            # Raise specific error
            raise FFmpegError(f"FFmpeg executable not found at path: {command[0]}. Please check FFMPEG_PATH config.") from e
        except Exception as e:
            # Catch other potential Popen errors
            raise FFmpegError(f"Error running FFmpeg command {' '.join(command)}: {e}") from e

    def extract_and_normalize_audio(self, video_path: str, output_audio_path: str) -> str:
        """Extracts, normalizes, and converts audio. Raises FFmpegError on failure."""
        self.logger.info(f"Starting audio extraction and normalization for: {video_path}")

        target_loudness = self.config.get("AUDIO_TARGET_LOUDNESS_LUFS")
        target_lpr = self.config.get("AUDIO_TARGET_LPR_DB")
        target_srate = self.config.get("AUDIO_TARGET_SRATE_HZ")

        ffmpeg_command = [
            self.ffmpeg_path, "-y", "-i", video_path,
            "-vn", "-ar", str(target_srate), "-ac", "1",
            "-af", f"loudnorm=I={target_loudness}:LRA={target_lpr}:tp=-1.5",
            "-map_metadata", "-1", "-c:a", "pcm_s16le",
            output_audio_path
        ]

        try:
            # Call directly, error will be raised if it fails
            self._run_ffmpeg_command(ffmpeg_command)
            self.logger.info(f"Audio extracted and normalized to: {output_audio_path}")
            return output_audio_path
        except FFmpegError as e:
            self.logger.error(f"Audio extraction/normalization failed for {video_path}: {e}")
            # Re-raise to stop the pipeline
            raise

