"""
Configuration for SRT Generator Phase 1 refactoring.

This module contains the default configuration for the SRT generator.
"""

import torch

# --- Default Configuration ---
DEFAULT_APP_CONFIG = {
    "FFMPEG_PATH": "ffmpeg",
    "FFPROBE_PATH": "ffprobe",
    "LOG_LEVEL": "INFO",
    "LOG_FORMAT": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    "TEMP_AUDIO_FILENAME": "temp_normalized_audio.wav",
    "CLEANUP_TEMP_AUDIO": True,
    "AUDIO_TARGET_LOUDNESS_LUFS": -16.0,
    "AUDIO_TARGET_LPR_DB": 15.0,
    "AUDIO_TARGET_SRATE_HZ": 16000,  # Must be positive integer
    "WHISPER_MODEL_NAME": "medium",  # atau "small"
    "WHISPER_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WHISPER_LANGUAGE": "en",
    "WHISPER_FP16": torch.cuda.is_available(),
    "WHISPER_BEAM_SIZE": 5,
    "WHISPER_TEMPERATURE": 0.0,
    "WHISPER_PROMPT": "This is a professional narration.",
    "WHISPER_WORD_TIMESTAMPS": True,

    "MARIANMT_MODEL_PATH_EN_ID": "Helsinki-NLP/opus-mt-en-id",
    "MARIANMT_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MARIANMT_BATCH_SIZE": 4,  # atau 2

    # Error Handling Configs
    "MAPPING_FILE_REQUIRED": False,  # If True, pipeline fails if mapping file is missing/invalid
    "TRANSLATION_ERROR_STRATEGY": "fail",  # Options: 'fail', 'fallback_english', 'skip'

    # Segmentation Configs
    "SEGMENTATION_STRATEGY": "pause",  # Options: 'pause', 'max_words', 'combined'
    "SEGMENTATION_PAUSE_THRESHOLD_SEC": 0.7,  # Threshold for pause-based segmentation
    "SEGMENTATION_MAX_WORDS": 10,  # Max words per segment for 'max_words' or 'combined' strategy
    "SEGMENTATION_MAX_DURATION_SEC": 5.0,  # Max duration per segment (optional limit)
    "BYPASS_POST_TRANSLATION_MERGE": True,  # Bypass the old merge_close_timed_words after segment translation

    "SRT_MAX_CHARS_PER_LINE": 42,
    "SRT_MAX_LINES_PER_BLOCK": 2,
    "SRT_MERGE_TIME_THRESHOLD_SEC": 0.7,
    "SRT_MERGE_MIN_DURATION_SEC": 0.5,
    "SRT_TIMESTAMP_PADDING_SEC": 0.05,
    "SRT_MERGE_MAX_WORDS": 7,
    "SRT_MERGE_MAX_GAP_SEC": 0.2,

    # Phase 1 - New Segmentation & SRT Configs
    "SEGMENT_MAX_PAUSE_SEC": 0.8,  # float: max pause between words to end a segment
    "SEGMENT_MAX_WORDS": 12,  # int: max words per segment before forced split
    "SRT_MINIMAL_BLOCK_PAUSE_SEC": 0.05,  # float: minimal pause between split SRT blocks
    "SRT_MIN_DURATION_PER_BLOCK_SEC": 1.0,  # float: minimum duration for any SRT block
}

