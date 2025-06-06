"""
AvtPipeline class for SRT Generator Phase 1 refactoring.

This class orchestrates the entire pipeline for generating SRT subtitles.
"""

import os
import time
import logging
import shutil
from typing import List, Dict, Any

# Import custom exceptions
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass
class ConfigError(PipelineError, ValueError):
    """Error related to configuration values."""
    pass
class FFmpegError(PipelineError):
    """Error during FFmpeg execution."""
    pass
class ModelLoadError(PipelineError):
    """Error loading ML models (Whisper, MarianMT)."""
    pass
class TranscriptionError(PipelineError):
    """Error during Whisper transcription."""
    pass
class TranslationError(PipelineError):
    """Error during MarianMT translation."""
    pass
class MappingError(PipelineError, ValueError):
    """Error related to idiom mapping file or format."""
    pass
class SRTFormatError(PipelineError, ValueError):
    """Error during SRT formatting or timestamp issues."""
    pass

# Import utility functions
def get_logger(name: str, level: str, log_format: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level.upper())
        ch = logging.StreamHandler()
        ch.setLevel(level.upper())
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class AvtPipeline:
    def __init__(self, config: dict):
        self.config = config.copy()
        self.logger = get_logger(
            "AvtPipeline",
            self.config.get("LOG_LEVEL"),
            self.config.get("LOG_FORMAT")
        )
        # Validate config before initializing components that depend on it
        self._validate_config()

        # Import components here to avoid circular imports
        from srt_generator_refactor.ffmpeg_manager import FFmpegManager
        from srt_generator_refactor.transcription_engine import TranscriptionEngine
        from srt_generator_refactor.translation_engine import TranslationEngine
        from srt_generator_refactor.basic_segmenter import BasicSegmenter
        from srt_generator_refactor.idiom_mapper import IdiomCsiMapper
        from srt_generator_refactor.srt_formatter import SRTFormatter

        self.ffmpeg = FFmpegManager(self.config, self.logger)
        self.transcriber = TranscriptionEngine(self.config, self.logger)
        self.translator = TranslationEngine(self.config, self.logger)
        self.basic_segmenter = BasicSegmenter(self.config, self.logger)
        self.idiom_mapper = None  # Initialized in run()
        self.srt_formatter = SRTFormatter(self.config, self.logger)
        self.temp_audio_path = None

    def _validate_config(self):
        """Validates critical configuration values."""
        self.logger.info("Validating configuration...")
        # Check executable paths
        ffmpeg_path = self.config.get("FFMPEG_PATH")
        if not shutil.which(ffmpeg_path):
            raise ConfigError(f"FFmpeg executable not found or not in PATH: {ffmpeg_path}")
        ffprobe_path = self.config.get("FFPROBE_PATH")
        if not shutil.which(ffprobe_path):
            raise ConfigError(f"FFprobe executable not found or not in PATH: {ffprobe_path}")

        # Check numeric values
        try:
            srate = int(self.config.get("AUDIO_TARGET_SRATE_HZ"))
            if srate <= 0:
                raise ValueError("Sample rate must be positive")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid AUDIO_TARGET_SRATE_HZ: {self.config.get('AUDIO_TARGET_SRATE_HZ')}. Must be positive integer.") from e

        try:
            batch_size = int(self.config.get("MARIANMT_BATCH_SIZE"))
            if batch_size <= 0:
                raise ValueError("Batch size must be positive")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid MARIANMT_BATCH_SIZE: {self.config.get('MARIANMT_BATCH_SIZE')}. Must be positive integer.") from e

        # Check strategy values
        allowed_strategies = ['fail', 'fallback_english', 'skip']
        if self.config.get("TRANSLATION_ERROR_STRATEGY") not in allowed_strategies:
             raise ConfigError(f"Invalid TRANSLATION_ERROR_STRATEGY: {self.config.get('TRANSLATION_ERROR_STRATEGY')}. Allowed: {allowed_strategies}")

        # Check boolean
        if not isinstance(self.config.get("MAPPING_FILE_REQUIRED"), bool):
             raise ConfigError(f"Invalid MAPPING_FILE_REQUIRED: {self.config.get('MAPPING_FILE_REQUIRED')}. Must be True or False.")

        # Validate Phase 1 Segmentation Configs
        try:
            max_pause_sec = float(self.config.get("SEGMENT_MAX_PAUSE_SEC"))
            if max_pause_sec < 0:
                raise ValueError("SEGMENT_MAX_PAUSE_SEC must be non-negative")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid SEGMENT_MAX_PAUSE_SEC: {self.config.get('SEGMENT_MAX_PAUSE_SEC')}. Must be non-negative number.") from e

        try:
            max_words = int(self.config.get("SEGMENT_MAX_WORDS"))
            if max_words <= 0:
                raise ValueError("SEGMENT_MAX_WORDS must be positive")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid SEGMENT_MAX_WORDS: {self.config.get('SEGMENT_MAX_WORDS')}. Must be positive integer.") from e

        try:
            min_block_pause = float(self.config.get("SRT_MINIMAL_BLOCK_PAUSE_SEC"))
            if min_block_pause < 0:
                raise ValueError("SRT_MINIMAL_BLOCK_PAUSE_SEC must be non-negative")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid SRT_MINIMAL_BLOCK_PAUSE_SEC: {self.config.get('SRT_MINIMAL_BLOCK_PAUSE_SEC')}. Must be non-negative number.") from e

        try:
            min_duration = float(self.config.get("SRT_MIN_DURATION_PER_BLOCK_SEC"))
            if min_duration <= 0:
                raise ValueError("SRT_MIN_DURATION_PER_BLOCK_SEC must be positive")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid SRT_MIN_DURATION_PER_BLOCK_SEC: {self.config.get('SRT_MIN_DURATION_PER_BLOCK_SEC')}. Must be positive number.") from e

        if not isinstance(self.config.get("BYPASS_POST_TRANSLATION_MERGE"), bool):
            raise ConfigError(f"Invalid BYPASS_POST_TRANSLATION_MERGE: {self.config.get('BYPASS_POST_TRANSLATION_MERGE')}. Must be True or False.")

        self.logger.info("Configuration validation successful.")

    def run(self, input_video: str, output_srt_path: str, mapping_json_path: str):
        self.logger.info(f"--- Starting AVT Pipeline (Phase 1 - Segment-Based) ---")
        self.logger.info(f"Input Video: {input_video}")
        self.logger.info(f"Output SRT: {output_srt_path}")
        self.logger.info(f"Idiom Mapping: {mapping_json_path}")

        start_time = time.time()
        output_dir = os.path.dirname(output_srt_path) or "."
        # Output dir existence/writability checked in main()
        self.temp_audio_path = os.path.join(output_dir, self.config.get("TEMP_AUDIO_FILENAME"))

        try:  # Main pipeline execution block
            # 1. Extract & Normalize Audio (Raises FFmpegError)
            self.logger.info("Step 1: Extracting and Normalizing Audio...")
            normalized_audio = self.ffmpeg.extract_and_normalize_audio(input_video, self.temp_audio_path)
            self.logger.info(f"Audio processed: {normalized_audio}")

            # 2. Transcribe Audio (Raises TranscriptionError, ModelLoadError)
            self.logger.info("Step 2: Transcribing Audio (Word-Level)...")
            word_entries = self.transcriber.get_word_level_timestamps(normalized_audio)
            self.logger.info(f"Transcription complete: {len(word_entries)} words found.")

            # 3. Segment Words (Phase 1 Addition)
            self.logger.info("Step 3: Segmenting Words...")
            segments = self.basic_segmenter.segment_words(word_entries)
            self.logger.info(f"Segmentation complete: {len(segments)} segments created.")

            # 4. Initialize Idiom Mapper (Raises MappingError)
            self.logger.info("Step 4: Initializing Idiom Mapper...")
            # Pass config to mapper for MAPPING_FILE_REQUIRED check
            from srt_generator_refactor.idiom_mapper import IdiomCsiMapper
            self.idiom_mapper = IdiomCsiMapper(mapping_json_path, self.config, self.logger)
            
            # 5. Apply Idiom Mapping to Segments
            self.logger.info("Step 5: Applying Idiom/CSI Mapping to Segments...")
            mapped_segments = self.idiom_mapper.map_segments_for_idioms(segments)
            self.logger.info(f"Idiom mapping complete: {len(mapped_segments)} segments processed.")

            # 6. Translate Segments (Raises TranslationError, ModelLoadError)
            self.logger.info("Step 6: Translating Segments...")
            translated_segments = self.translator.translate_segments(mapped_segments)
            self.logger.info(f"Translation complete: {len(translated_segments)} segments processed.")

            # 7. Format to SRT (Raises SRTFormatError)
            self.logger.info("Step 7: Formatting to SRT...")
            srt_content = self.srt_formatter.create_srt_from_segments(translated_segments)
            if not srt_content:
                self.logger.warning("Final SRT content is empty.")

            # 8. Save SRT File Atomically
            self.logger.info(f"Step 8: Saving SRT file to: {output_srt_path}")
            temp_srt_path = output_srt_path + ".tmp"
            try:
                with open(temp_srt_path, 'w', encoding='utf-8') as f:
                    f.write(srt_content)
                # Atomic rename
                os.rename(temp_srt_path, output_srt_path)
                self.logger.info(f"Successfully wrote SRT file: {output_srt_path}")
            except OSError as e:
                self.logger.error(f"Failed to write or rename SRT file {output_srt_path}: {e}")
                # Attempt cleanup of temp file
                if os.path.exists(temp_srt_path):
                    try:
                        os.remove(temp_srt_path)
                    except OSError:
                        pass
                raise PipelineError(f"Failed to save SRT file: {e}") from e

        # Specific exception handling
        except (ConfigError, FFmpegError, ModelLoadError, TranscriptionError, TranslationError, MappingError, SRTFormatError, FileNotFoundError, PermissionError, ValueError) as e:
            self.logger.error(f"Pipeline failed for video '{input_video}'. Error Type: {type(e).__name__}. Message: {e}", exc_info=False)  # Log specific error type
            self.logger.debug("Traceback:", exc_info=True)  # Log full traceback at debug level
            raise  # Re-raise the specific error for main() to catch
        except Exception as e:  # Catch-all for truly unexpected errors
            self.logger.critical(f"Unexpected critical error in pipeline for video '{input_video}': {e}", exc_info=True)
            raise PipelineError(f"Unexpected critical error: {e}") from e  # Wrap in PipelineError
        finally:
            # 9. Cleanup Temp Files
            if self.temp_audio_path and self.config.get("CLEANUP_TEMP_AUDIO") and os.path.exists(self.temp_audio_path):
                self.logger.info(f"Step 9: Cleaning up temporary audio file: {self.temp_audio_path}")
                try:
                    os.remove(self.temp_audio_path)
                    self.logger.info("Temporary file removed.")
                except OSError as e:
                    self.logger.warning(f"Could not remove temporary file {self.temp_audio_path}: {e}")

            end_time = time.time()
            self.logger.info(f"--- AVT Pipeline Finished ---")
            self.logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")

