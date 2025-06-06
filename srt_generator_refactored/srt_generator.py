# ['''
🚀 ULTRA-OPTIMIZED Word-Level SRT Generator v3.3 (Manus Enhanced & Robustified - Revision 2: Error Handling)
🎯 Production-Ready Enterprise Solution with AI-Powered Quality Enhancement & AVT Specialization
⚡ Maximum Performance & Precision Implementation for English -> Indonesian Translation

📝 Pipeline Overview:
1.  Parse & Validate command-line arguments (video path, output path, mapping path).
2.  Load & Validate configuration settings.
3.  Initialize pipeline components (FFmpeg, Whisper, MarianMT, IdiomMapper, SRTFormatter).
4.  Extract and normalize audio (raises FFmpegError on failure).
5.  Transcribe audio (raises TranscriptionError on failure or no words).
6.  Apply idiom/CSI mapping (raises MappingError if required file fails).
7.  Translate remaining entries (raises TranslationError based on strategy).
8.  Merge translated entries.
9.  Format into SRT blocks (raises SRTFormatError on invalid data).
10. Save the final SRT file atomically.
11. Cleanup temporary files.

🌟 REVISION 2 CHANGES (Based on AnalisisMendalamdanPenguatanErrorHandling.pdf):
✨ Custom Exceptions: Defined specific error classes (PipelineError, ConfigError, FFmpegError, etc.).
✨ Robust Input Validation: Checks file existence/permissions for video, mapping, output dir in `main`.
✨ Configuration Validation: Checks FFmpeg/FFprobe paths and key config values in `AvtPipeline._validate_config`.
✨ Specific Error Raising: Replaced generic returns/exceptions with specific custom exceptions in each component (FFmpeg, Models, Mapping, SRT).
✨ Configurable Translation Errors: Added `TRANSLATION_ERROR_STRATEGY` (
'fail
', 
'fallback_english
', 
'skip
').
✨ Configurable Mapping Requirement: Added `MAPPING_FILE_REQUIRED`.
✨ Atomic SRT Write: Ensures SRT file integrity during saving.
✨ Enhanced SRT Formatting Validation: Checks timestamps and entry structure.
✨ Granular Pipeline Error Handling: Catches specific exceptions in `AvtPipeline.run`.
✨ Graceful Exit in `main`: Catches pipeline errors and exits with code 1.

🐞 KNOWN ISSUES / TODO for NEXT REVISION:
-   [ ] Implement Phrase-Level Translation for better quality.
-   [ ] Review and Simplify Subtitle Merging Logic.
-   [ ] Add Optional Memory Management (`torch.cuda.empty_cache`, `gc.collect`).
-   [ ] Add Optional FFprobe validation before extraction.
-   [ ] Enhance punctuation handling in idiom mapping.
'''

import os
import json
import subprocess
import logging
import shutil
import time
import re
import argparse
import sys
import gc
from typing import List, Dict, Any, Optional, Tuple, Union

# --- Dependency Check ---
try:
    import torch
    import whisper
    from transformers import MarianMTModel, MarianTokenizer
except ImportError as e:
    print(f"ERROR: Essential library not found: {e}")
    print("Please ensure torch, whisper, and transformers are installed.")
    print("In Google Colab, run: !pip install -q torch torchaudio git+https://github.com/openai/whisper.git transformers sentencepiece sacremoses")
    raise ImportError("Essential libraries missing. Cannot continue.") from e

# --- Custom Exceptions (Point 1) ---
class PipelineError(RuntimeError):
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
    "AUDIO_TARGET_SRATE_HZ": 16000, # Must be positive integer
    "WHISPER_MODEL_NAME": "medium", # atau "small"
    "WHISPER_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "WHISPER_LANGUAGE": "en",
    "WHISPER_FP16": torch.cuda.is_available(),
    "WHISPER_BEAM_SIZE": 5,
    "WHISPER_TEMPERATURE": 0.0,
    "WHISPER_PROMPT": "This is a professional narration.",
    "WHISPER_WORD_TIMESTAMPS": True,

    "MARIANMT_MODEL_PATH_EN_ID": "Helsinki-NLP/opus-mt-en-id",
    "MARIANMT_DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "MARIANMT_BATCH_SIZE": 4, # atau 2

    # Error Handling Configs
    "MAPPING_FILE_REQUIRED": False, # If True, pipeline fails if mapping file is missing/invalid (Point 7)
    "TRANSLATION_ERROR_STRATEGY": "fail", # Options: 
'fail
', 
'fallback_english
', 
'skip
' (Point 6)

    # Segmentation Configs
    "SEGMENTATION_STRATEGY": "pause", # Options: 
'pause
', 
'max_words
', 
'combined
'
    "SEGMENTATION_PAUSE_THRESHOLD_SEC": 0.7, # Threshold for pause-based segmentation
    "SEGMENTATION_MAX_WORDS": 10, # Max words per segment for 
'max_words
' or 
'combined
' strategy
    "SEGMENTATION_MAX_DURATION_SEC": 5.0, # Max duration per segment (optional limit)
    "BYPASS_POST_TRANSLATION_MERGE": True, # Bypass the old merge_close_timed_words after segment translation

    "SRT_MAX_CHARS_PER_LINE": 42,
    "SRT_MAX_LINES_PER_BLOCK": 2,
    "SRT_MERGE_TIME_THRESHOLD_SEC": 0.7,
    "SRT_MERGE_MIN_DURATION_SEC": 0.5,
    "SRT_TIMESTAMP_PADDING_SEC": 0.05,
    "SRT_MERGE_MAX_WORDS": 7,
    "SRT_MERGE_MAX_GAP_SEC": 0.2,

    # Phase 1 - New Segmentation & SRT Configs
    "SEGMENT_MAX_PAUSE_SEC": 0.8, # float: max pause between words to end a segment
    "SEGMENT_MAX_WORDS": 12, # int: max words per segment before forced split
    "SRT_MINIMAL_BLOCK_PAUSE_SEC": 0.05, # float: minimal pause between split SRT blocks
    "SRT_MIN_DURATION_PER_BLOCK_SEC": 1.0, # float: minimum duration for any SRT block
}

# --- Utility Functions ---
def get_logger(name: str, level: str, log_format: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level.upper())
        ch = logging.StreamHandler()
        ch.setLevel(level.upper())
        formatter = logging.Formatter(log_format)
        ch.setFormatter(formatter)
        # Optional: Add FileHandler here (Point 10)
        # log_file_path = 
'pipeline.log
'
        # fh = logging.FileHandler(log_file_path)
        # fh.setLevel(level.upper())
        # fh.setFormatter(formatter)
        # logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

# --- Core Components ---

class FFmpegManager:
    def __init__(self, config: dict, logger: logging.Logger):
        self.ffmpeg_path = config.get("FFMPEG_PATH", "ffmpeg")
        # ffprobe_path stored but validation happens in AvtPipeline._validate_config
        self.ffprobe_path = config.get("FFPROBE_PATH", "ffprobe")
        self.config = config
        self.logger = logger

    def _run_ffmpeg_command(self, command: List[str]) -> Tuple[str, str]: # Return stdout, stderr (Point 4)
        """Executes an FFmpeg command, raises FFmpegError on failure."""
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                stderr_snippet = (stderr[:500] + 
'...
') if len(stderr) > 500 else stderr
                # Raise specific error (Point 4)
                raise FFmpegError(f"FFmpeg command failed. RC={process.returncode}. Path={command[0]}. Stderr: {stderr_snippet}")

            self.logger.info(f"FFmpeg command executed successfully: {
' 
'.join(command)}")
            self.logger.debug(f"FFmpeg stdout: {stdout}")
            if stderr: self.logger.debug(f"FFmpeg stderr: {stderr}")
            return stdout, stderr

        except FileNotFoundError as e:
            # Raise specific error (Point 4)
            raise FFmpegError(f"FFmpeg executable not found at path: {command[0]}. Please check FFMPEG_PATH config.") from e
        except Exception as e:
            # Catch other potential Popen errors
            raise FFmpegError(f"Error running FFmpeg command {
' 
'.join(command)}: {e}") from e

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
            # Call directly, error will be raised if it fails (Point 4)
            self._run_ffmpeg_command(ffmpeg_command)
            self.logger.info(f"Audio extracted and normalized to: {output_audio_path}")
            return output_audio_path
        except FFmpegError as e:
            self.logger.error(f"Audio extraction/normalization failed for {video_path}: {e}")
            # Re-raise to stop the pipeline
            raise

class TranscriptionEngine:
    def __init__(self, config: dict, logger: logging.Logger):
        self.model_name = config.get("WHISPER_MODEL_NAME")
        self.device = config.get("WHISPER_DEVICE")
        self.config = config
        self.logger = logger
        self.model = None
        self._load_model()

    def _load_model(self):
        self.logger.info(f"Loading Whisper model: {self.model_name} on device: {self.device}")
        if "large" in self.model_name and self.device == "cuda":
             self.logger.warning("Loading large Whisper model on GPU. Requires significant VRAM.")
        elif "large" in self.model_name and self.device == "cpu":
             self.logger.warning("Loading large Whisper model on CPU. Very slow, requires significant RAM.")

        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            self.logger.info("Whisper model loaded successfully.")
        except Exception as e:
            # Raise specific error (Point 5)
            raise ModelLoadError(f"Failed to load Whisper model {self.model_name}: {e}") from e

    def _validate_word_entry(self, word_info: Dict, index: int) -> bool:
        """Validates structure and values of a single word entry."""
        if not isinstance(word_info, dict):
            self.logger.warning(f"Invalid word entry type at index {index}: {type(word_info)}. Skipping.")
            return False
        if not all(k in word_info for k in (
'word
', 
'start
', 
'end
')): # Ganti 
'text
' menjadi 
'word
'
            self.logger.warning(f"Word entry missing required keys at index {index}: {word_info}. Skipping.")
            return False
        try:
            start = float(word_info[
'start
'])
            end = float(word_info[
'end
'])
            if start < 0 or end < 0 or end < start:
                self.logger.warning(f"Invalid start/end times in word entry at index {index}: {word_info}. Skipping.")
                return False
        except (ValueError, TypeError):
            self.logger.warning(f"Non-numeric start/end times in word entry at index {index}: {word_info}. Skipping.")
            return False
        return True

    def get_word_level_timestamps(self, audio_path: str) -> List[Dict[str, Union[str, float]]]:
        """Transcribes audio, returns word timestamps. Raises TranscriptionError on failure or no words."""
        if not self.model:
            # This case should ideally not happen if _load_model raises correctly
            raise ModelLoadError("Whisper model is not loaded. Cannot transcribe.")

        self.logger.info(f"Starting word-level transcription for: {audio_path}")

        transcribe_options = {
            "language": self.config.get("WHISPER_LANGUAGE"),
            "beam_size": self.config.get("WHISPER_BEAM_SIZE"),
            "temperature": float(self.config.get("WHISPER_TEMPERATURE")), # Ensure float
            "word_timestamps": True,
            "fp16": self.config.get("WHISPER_FP16")
        }
        if self.config.get("WHISPER_PROMPT"):
             transcribe_options["initial_prompt"] = self.config.get("WHISPER_PROMPT")

        try:
            self.logger.debug(f"Whisper transcribe options: {transcribe_options}")
            result = self.model.transcribe(audio_path, **transcribe_options)
            self.logger.info("Whisper transcription complete.")
        except Exception as e:
            # Raise specific error (Point 5)
            raise TranscriptionError(f"Whisper transcription failed for {audio_path}: {e}") from e

        raw_word_entries = []
        if "words" in result and result["words"] and isinstance(result["words"], list):
            self.logger.info("Extracting words from top-level 
'words
' key.")
            raw_word_entries = result["words"]
        elif "segments" in result and isinstance(result["segments"], list):
            self.logger.info("Extracting words from 
'segments
'.")
            for segment in result["segments"]:
                if "words" in segment and segment["words"] and isinstance(segment["words"], list):
                    raw_word_entries.extend(segment["words"])
                elif segment.get("text","").strip(): # Fallback for segments without 
'words
'
                    raw_word_entries.append({
                        "word": segment["text"].strip(), # Treat segment text as one "word"
                        "start": segment["start"],
                        "end": segment["end"]
                    })

        # Validate and format extracted entries (Point 5 - Validation)
        word_entries = []
        for i, word_info in enumerate(raw_word_entries):
             if self._validate_word_entry(word_info, i):
                 word_entries.append({
                     "text": str(word_info["word"]).strip(),
                     "start": float(word_info["start"]),
                     "end": float(word_info["end"])
                 })

        # Raise error if no valid words found (Point 5)
        if not word_entries:
            self.logger.error(f"Transcription for {audio_path} yielded no valid word timestamps after processing.")
            raise TranscriptionError("Transcription completed but yielded no valid word timestamps.")

        self.logger.info(f"Extracted and validated {len(word_entries)} word-level entries from {audio_path}.")
        return word_entries

class TranslationEngine:
    def __init__(self, config: dict, logger: logging.Logger):
        self.model_path = config.get("MARIANMT_MODEL_PATH_EN_ID")
        self.device = config.get("MARIANMT_DEVICE")
        self.batch_size = config.get("MARIANMT_BATCH_SIZE")
        self.error_strategy = config.get("TRANSLATION_ERROR_STRATEGY", "fail") # Point 6
        self.config = config
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        self.logger.info(f"Loading MarianMT model: {self.model_path} on device: {self.device}")
        if self.device == "cuda":
             self.logger.warning("Loading MarianMT model on GPU. Requires VRAM.")
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_path)
            self.model = MarianMTModel.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            self.logger.info("MarianMT model loaded successfully.")
        except Exception as e:
            # Raise specific error (Point 6)
            raise ModelLoadError(f"Failed to load MarianMT model {self.model_path}: {e}") from e

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translates a batch of texts, handling errors based on configured strategy."""
        if not self.model or not self.tokenizer:
            raise ModelLoadError("MarianMT model/tokenizer not loaded. Cannot translate.")

        if not texts or all(not t.strip() for t in texts):
            return ["" for _ in texts]

        translated_texts = []
        original_indices = list(range(len(texts)))

        try:
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i : i + self.batch_size]
                batch_indices = original_indices[i : i + self.batch_size]
                self.logger.debug(f"Translating batch starting at index {batch_indices[0]} ({len(batch_texts)} items)")

                try:
                    encoded_batch = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                    with torch.no_grad():
                        translated_tokens = self.model.generate(**encoded_batch)
                    decoded_batch = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
                    translated_texts.extend(decoded_batch)
                except Exception as batch_error:
                    # Handle error based on strategy (Point 6)
                    self.logger.error(f"Error translating batch starting at index {batch_indices[0]}: {batch_error}")
                    if self.error_strategy == 
'fail
':
                        raise TranslationError(f"Translation failed for batch starting at index {batch_indices[0]}: {batch_error}") from batch_error
                    elif self.error_strategy == 
'fallback_english
':
                        self.logger.warning(f"Falling back to English for failed batch (indices {batch_indices}).")
                        translated_texts.extend(batch_texts) # Add original English texts
                    elif self.error_strategy == 
'skip
':
                        self.logger.warning(f"Skipping failed translation batch (indices {batch_indices}).")
                        translated_texts.extend(["" for _ in batch_texts]) # Add empty strings
                    else: # Default to fail for unknown strategy
                         raise TranslationError(f"Unknown TRANSLATION_ERROR_STRATEGY 
'{self.error_strategy}
'. Failing batch.")

            # Final validation (Point 6)
            if len(translated_texts) != len(texts):
                raise TranslationError(f"Mismatch in translation input/output count. Expected {len(texts)}, got {len(translated_texts)}.")

            self.logger.debug(f"Finished translating {len(texts)} texts.")
            return translated_texts

        except Exception as e:
            # Catch unexpected errors during the batching loop itself
            self.logger.error(f"Unexpected error during translation batch processing: {e}")
            raise TranslationError(f"Unexpected error during translation: {e}") from e

class IdiomCsclass IdiomCsiMapper:
    def __init__(self, mapping_path: str, config: dict, logger: logging.Logger):
        self.mapping_path = mapping_path
        self.config = config
        self.logger = logger
        self.mapping = {}
        self.is_required = self.config.get("MAPPING_FILE_REQUIRED", False) # Point 7
        self._load_mapping()

    def _load_mapping(self):
        self.logger.info(f"Loading idiom/CSI mapping from: {self.mapping_path}")
        try:
            if not os.path.exists(self.mapping_path):
                if self.is_required:
                    # Raise specific error (Point 7)
                    raise MappingError(f"Required mapping file not found: {self.mapping_path}")
                else:
                    self.logger.warning(f"Mapping file not found: {self.mapping_path}. Idiom mapping will be disabled.")
                    return # Proceed with empty mapping

            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)

            # Validate format (Point 7)
            if not isinstance(raw_mapping, dict):
                 raise MappingError(f"Mapping file content is not a valid JSON dictionary: {self.mapping_path}")

            # Clean keys (strip whitespace)
            self.mapping = {key.strip(): value for key, value in raw_mapping.items()}
            self.logger.info(f"Loaded {len(self.mapping)} idiom/CSI rules from {self.mapping_path}.")

        except json.JSONDecodeError as e:
            log_msg = f"Error decoding JSON mapping file {self.mapping_path}: {e}"
            if self.is_required:
                raise MappingError(log_msg) from e # Fail if required (Point 7)
            else:
                self.logger.error(log_msg + " Idiom mapping disabled.")
                self.mapping = {}
        except Exception as e:
            log_msg = f"Error loading mapping file {self.mapping_path}: {e}"
            if self.is_required:
                raise MappingError(log_msg) from e # Fail if required (Point 7)
            else:
                self.logger.error(log_msg + " Idiom mapping disabled.")
                self.mapping = {}

    def _normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[.,!?;:"\']$', '', text)
        text = text.strip()
        return text

    def map_segments_for_idioms(self, segments: List[Dict]) -> List[Dict]:
        """
        Maps segments to idioms based on the entire segment text.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of segment dictionaries with idioms mapped
        """
        if not self.mapping:
            self.logger.info("No idiom mapping loaded or available. All segments will be marked for NMT.")
            return segments
        
        self.logger.info(f"Applying idiom/CSI mapping to {len(segments)} segments...")
        mapped_segments = []
        
        for segment in segments:
            # Skip segments that don't have the required fields
            if not isinstance(segment, dict) or not all(k in segment for k in ("original_text", "start_time", "end_time")):
                self.logger.warning(f"Skipping invalid segment during idiom mapping: {segment}")
                mapped_segments.append(segment)
                continue
            
            # Normalize the segment text for matching
            normalized_text = self._normalize_text(segment["original_text"])
            
            # Check if the entire segment matches an idiom
            found_match = False
            for idiom_key, idiom_translation in self.mapping.items():
                normalized_idiom = self._normalize_text(idiom_key)
                if normalized_text == normalized_idiom:
                    # Match found, update the segment
                    self.logger.debug(f"Mapped idiom for segment: '{segment['original_text']}' -> '{idiom_translation}'")
                    segment["translated_text"] = idiom_translation
                    segment["translation_source"] = "IDIOM_MAPPED"
                    found_match = True
                    break
            
            # If no match found, keep the segment as is (NEEDS_NMT)
            if not found_match:
                self.logger.debug(f"No idiom match for segment: '{segment['original_text']}'")
            
            mapped_segments.append(segment)
        
        self.logger.info(f"Idiom mapping complete. {sum(1 for s in mapped_segments if s.get('translation_source') == 'IDIOM_MAPPED')} segments mapped.")
        return mapped_segments

    def map_words_to_idioms(self, word_entries: List[Dict]) -> List[Dict]:
        if not self.mapping:
            self.logger.info("No idiom mapping loaded or available. Marking all words for NMT.")
            return [{**word, "translation_source": "NMT"} for word in word_entries]

        self.logger.info("Applying idiom/CSI mapping...")
        output_entries = []
        i = 0
        n = len(word_entries)
        max_idiom_len = 0
        normalized_mapping = {}
        for key, value in self.mapping.items():
            normalized_key_words = [self._normalize_text(w) for w in key.split()]
            if normalized_key_words:
                normalized_mapping[tuple(normalized_key_words)] = value
                max_idiom_len = max(max_idiom_len, len(normalized_key_words))

        self.logger.debug(f"Max idiom length (words): {max_idiom_len}")

        while i < n:
            found_match = False
            for length in range(min(max_idiom_len, n - i), 0, -1):
                phrase_words = word_entries[i : i + length]
                phrase_texts_normalized = tuple(self._normalize_text(w['text']) for w in phrase_words)

                if phrase_texts_normalized in normalized_mapping:
                    mapped_translation = normalized_mapping[phrase_texts_normalized]
                    start_time = phrase_words[0]['start']
                    end_time = phrase_words[-1]['end']
                    original_phrase = " ".join([w['text'] for w in phrase_words])

                    output_entries.append({
                        "text": mapped_translation,
                        "start": start_time,
                        "end": end_time,
                        "original_english": original_phrase,
                        "translation_source": "Idiom/CSI Mapping"
                    })
                    self.logger.debug(f"Mapped idiom: '{original_phrase}' -> '{mapped_translation}'")
                    i += length
                    found_match = True
                    break

            if not found_match:
                output_entries.append({**word_entries[i], "translation_source": "NMT"})
                i += 1

        self.logger.info(f"Idiom mapping complete. Result has {len(output_entries)} entries.")
        return output_entriesies

class SRTFormatter:
    def __init__(self, config: dict, logger: logging.Logger):
        self.max_chars_per_line = config.get("SRT_MAX_CHARS_PER_LINE")
        self.max_lines_per_block = config.get("SRT_MAX_LINES_PER_BLOCK")
        self.config = config
        self.logger = logger

    def _format_timestamp(self, seconds: float) -> str:
        # Validate timestamp (Point 8)
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise SRTFormatError(f"Invalid non-negative timestamp value received: {seconds} ({type(seconds)})")

        milliseconds = round(seconds * 1000.0)
        hours = milliseconds // 3_600_000
        milliseconds %= 3_600_000
        minutes = milliseconds // 60_000
        milliseconds %= 60_000
        seconds_part = milliseconds // 1_000
        milliseconds %= 1_000
        return f"{hours:02d}:{minutes:02d}:{seconds_part:02d},{milliseconds:03d}"

    def _format_text_for_srt(self, text: str) -> str:
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            if not current_line:
                if len(word) > self.max_chars_per_line:
                    lines.append(word[:self.max_chars_per_line])
                    words.insert(words.index(word) + 1, word[self.max_chars_per_line:])
                    current_line = ""
                else:
                    current_line = word
            else:
                if len(current_line) + 1 + len(word) <= self.max_chars_per_line:
                    current_line += " " + word
                else:
                    lines.append(current_line)
                    if len(word) > self.max_chars_per_line:
                        lines.append(word[:self.max_chars_per_line])
                        words.insert(words.index(word) + 1, word[self.max_chars_per_line:])
                        current_line = ""
                    else:
                        current_line = word
        if current_line:
            lines.append(current_line)

        if len(lines) > self.max_lines_per_block:
            original_full_text = text
            formatted_lines = lines[:self.max_lines_per_block]
            if formatted_lines:
                 formatted_lines[-1] = formatted_lines[-1][:self.max_chars_per_line-3] + "..."
            displayed = 
'\\n
'.join(formatted_lines)
            self.logger.warning(
                f"SRT text truncated: Exceeded {self.max_lines_per_block} lines. "
                f"Original: 
'{original_full_text}
' -> Displayed: 
'{displayed}
'"
            )
        else:
            formatted_lines = lines
        return "\n".join(formatted_lines)

    def create_srt_content(self, timed_entries: List[Dict]) -> str:
        # Validate input type (Point 8)
        if not isinstance(timed_entries, list):
            raise SRTFormatError(f"Input timed_entries must be a list, got {type(timed_entries)}")

        srt_blocks = []
        for i, entry in enumerate(timed_entries):
            try:
                # Validate entry structure and content (Point 8)
                if not isinstance(entry, dict):
                    self.logger.warning(f"Skipping invalid entry at index {i}: Not a dictionary ({type(entry)}).")
                    continue
                if not all(k in entry for k in (
'text
', 
'start
', 
'end
')):
                    self.logger.warning(f"Skipping invalid entry at index {i}: Missing keys ({entry.keys()}).")
                    continue
                if not entry[
'text
'] or not str(entry[
'text
']).strip():
                    self.logger.debug(f"Skipping empty text entry at index {i}")
                    continue

                start_time_str = self._format_timestamp(entry[
'start
'])
                end_time_str = self._format_timestamp(entry[
'end
'])

                # Basic duration check
                if entry[
'end
'] < entry[
'start
']:
                     self.logger.warning(f"Correcting negative duration at index {i}: start={entry[
'start
']}, end={entry[
'end
']}. Setting end = start.")
                     end_time_str = start_time_str # Or use start + minimal duration

                formatted_text = self._format_text_for_srt(str(entry[
'text
']))

                if not formatted_text:
                     self.logger.debug(f"Skipping entry at index {i} due to empty formatted text: {entry}")
                     continue

                block = f"{i + 1}\n{start_time_str} --> {end_time_str}\n{formatted_text}\n"
                srt_blocks.append(block)

            except (KeyError, TypeError, ValueError, SRTFormatError) as e:
                # Catch errors during processing of a single entry
                self.logger.error(f"Skipping invalid entry at index {i} due to error: {e}. Entry data: {entry}", exc_info=False)
                continue # Skip this entry and proceed

        return "\n".join(srt_blocks)

# --- Segmentation Component ---

class BasicSegmenter:
    """
    Class for segmenting word entries into coherent segments based on pause duration, max words, and punctuation.
    """
    
    def __init__(self, config: Dict, logger: logging.Logger):
        """
        Initialize the BasicSegmenter.
        
        Args:
            config: Configuration dictionary containing segmentation parameters
            logger: Logger instance for logging
        """
        self.config = config
        self.logger = logger
        self.max_pause_sec = config.get("SEGMENT_MAX_PAUSE_SEC", 0.8)
        self.max_words = config.get("SEGMENT_MAX_WORDS", 12)
        
        self.logger.info(f"BasicSegmenter initialized with max_pause_sec={self.max_pause_sec}, max_words={self.max_words}")
    
    def segment_words(self, word_entries: List[Dict]) -> List[Dict]:
        """
        Segment word entries into coherent segments.
        
        Args:
            word_entries: List of word entry dictionaries, each with 'text', 'start', 'end' keys
            
        Returns:
            List of segment dictionaries
        """
        if not word_entries:
            self.logger.warning("No word entries provided for segmentation.")
            return []
        
        self.logger.info(f"Starting basic segmentation for {len(word_entries)} word entries.")
        
        segments = []
        current_segment_words = []
        
        # Helper function to create a segment from a list of words
        def create_segment(words: List[Dict]) -> Dict:
            if not words:
                return None
            
            start_time = words[0]["start"]
            end_time = words[-1]["end"]
            duration = end_time - start_time
            original_text = " ".join([word["text"] for word in words])
            
            return {
                "original_words": words.copy(),  # Store a copy of the original word entries
                "original_text": original_text,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration,
                "translation_source": "NEEDS_NMT",  # Initial status
                "translated_text": None  # To be filled by TranslationEngine
            }
        
        # Process each word entry
        for i, word in enumerate(word_entries):
            # Add the first word to the current segment
            if not current_segment_words:
                current_segment_words.append(word)
                continue
            
            # Check if we need to create a new segment
            create_new_segment = False
            
            # Check pause duration between current word and previous word
            prev_word = current_segment_words[-1]
            pause_duration = word["start"] - prev_word["end"]
            if pause_duration > self.max_pause_sec:
                self.logger.debug(f"Creating new segment due to pause: {pause_duration:.2f}s > {self.max_pause_sec:.2f}s")
                create_new_segment = True
            
            # Check if current segment has reached max words
            elif len(current_segment_words) >= self.max_words:
                self.logger.debug(f"Creating new segment due to max words: {len(current_segment_words)} >= {self.max_words}")
                create_new_segment = True
            
            # Optional: Check if previous word ends with strong punctuation
            elif prev_word["text"].strip().endswith((".", "?", "!")):
                self.logger.debug(f"Creating new segment due to punctuation: '{prev_word['text']}'")
                create_new_segment = True
            
            # If we need to create a new segment, do it and start a new one
            if create_new_segment:
                segment = create_segment(current_segment_words)
                if segment:
                    segments.append(segment)
                current_segment_words = [word]  # Start a new segment with the current word
            else:
                # Add the current word to the current segment
                current_segment_words.append(word)
        
        # Don't forget to add the last segment if there are words left
        if current_segment_words:
            segment = create_segment(current_segment_words)
            if segment:
                segments.append(segment)
        
        self.logger.info(f"Basic segmentation complete. Created {len(segments)} segments.")
        return segments

class Segmenter:
    """Segments word entries into coherent phrases or sentences for translation."""
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.strategy = config.get("SEGMENTATION_STRATEGY", "pause")
        self.pause_threshold = config.get("SEGMENTATION_PAUSE_THRESHOLD_SEC", 0.7)
        self.max_words = config.get("SEGMENTATION_MAX_WORDS", 10)
        self.max_duration = config.get("SEGMENTATION_MAX_DURATION_SEC", 5.0)
        self.logger.info(f"Segmenter initialized with strategy: {self.strategy}, pause: {self.pause_threshold}s, max_words: {self.max_words}, max_duration: {self.max_duration}s")

    def segment(self, processed_entries: List[Dict]) -> List[Dict]:
        """Segments processed entries (word-level or idiom-mapped) into translation units."""
        if not processed_entries:
            return []

        self.logger.info(f"Starting segmentation for {len(processed_entries)} entries using strategy 
'{self.strategy}
'.")
        segments = []
        current_segment_words = []
        last_word_end_time = None

        def finalize_segment(word_buffer: List[Dict]):
            """Helper to create a segment dict from a buffer of word entries."""
            if not word_buffer:
                return None
            
            # Check if it
's a pre-mapped idiom segment
            if len(word_buffer) == 1 and word_buffer[0].get("translation_source") == "Idiom/CSI Mapping":
                self.logger.debug(f"Finalizing pre-mapped segment: {word_buffer[0][
'text
']}")
                return {
                    "original_english_text": word_buffer[0].get("original_english", word_buffer[0]["text"]), # Use original if available
                    "translated_text": word_buffer[0]["text"], # Mapped text is the 
'translation
'
                    "start_time": word_buffer[0]["start"],
                    "end_time": word_buffer[0]["end"],
                    "translation_source": "Idiom/CSI Mapping"
                }
            
            # Otherwise, it
's an NMT segment
            segment_text = " ".join([word["text"] for word in word_buffer])
            start_time = word_buffer[0]["start"]
            end_time = word_buffer[-1]["end"]
            self.logger.debug(f"Finalizing NMT segment: 
'{segment_text}
' ({start_time:.2f}s - {end_time:.2f}s)")
            return {
                "original_english_text": segment_text,
                "translated_text": None, # To be filled by TranslationEngine
                "start_time": start_time,
                "end_time": end_time,
                "translation_source": "NMT_Segment"
            }

        for i, entry in enumerate(processed_entries):
            # Basic validation
            if not isinstance(entry, dict) or not all(k in entry for k in ("text", "start", "end", "translation_source")):
                self.logger.warning(f"Skipping invalid entry during segmentation at index {i}: {entry}")
                continue
            
            current_word_start_time = entry["start"]
            current_word_end_time = entry["end"]
            is_idiom_mapped = entry.get("translation_source") == "Idiom/CSI Mapping"

            # If the current entry is an idiom, it forms its own segment.
            # Finalize any pending NMT segment first.
            if is_idiom_mapped:
                if current_segment_words:
                    segment_dict = finalize_segment(current_segment_words)
                    if segment_dict: segments.append(segment_dict)
                    current_segment_words = []
                    last_word_end_time = None # Reset time for next NMT segment
                
                # Add the idiom segment directly
                idiom_segment = finalize_segment([entry]) # Pass as list
                if idiom_segment: segments.append(idiom_segment)
                last_word_end_time = current_word_end_time # Update time for potential next NMT segment
                continue

            # --- Handle NMT words --- 
            # Calculate pause if applicable
            pause_duration = (current_word_start_time - last_word_end_time) if last_word_end_time is not None else 0
            
            # Determine if a break should occur BEFORE adding the current word
            should_break = False
            if current_segment_words: # Only break if there
's an existing segment being built
                segment_duration = current_word_end_time - current_segment_words[0]["start"]
                word_count = len(current_segment_words)

                if self.strategy == "pause":
                    if pause_duration > self.pause_threshold:
                        self.logger.debug(f"Break due to pause > {self.pause_threshold}s (pause: {pause_duration:.2f}s)")
                        should_break = True
                elif self.strategy == "max_words":
                    if word_count >= self.max_words:
                        self.logger.debug(f"Break due to max words >= {self.max_words} (count: {word_count})")
                        should_break = True
                elif self.strategy == "combined":
                    if pause_duration > self.pause_threshold:
                        self.logger.debug(f"Break due to pause > {self.pause_threshold}s (pause: {pause_duration:.2f}s)")
                        should_break = True
                    elif word_count >= self.max_words:
                        self.logger.debug(f"Break due to max words >= {self.max_words} (count: {word_count})")
                        should_break = True
                
                # Optional duration check (applies to all strategies if > 0)
                if not should_break and self.max_duration > 0 and segment_duration > self.max_duration:
                     self.logger.debug(f"Break due to max duration > {self.max_duration}s (duration: {segment_duration:.2f}s)")
                     should_break = True

            # If break condition met, finalize the previous segment
            if should_break:
                segment_dict = finalize_segment(current_segment_words)
                if segment_dict: segments.append(segment_dict)
                current_segment_words = [] # Start a new segment buffer
                # last_word_end_time is updated below after adding the current word

            # Add the current NMT word to the buffer
            current_segment_words.append(entry)
            last_word_end_time = current_word_end_time

        # Finalize any remaining words in the buffer
        if current_segment_words:
            segment_dict = finalize_segment(current_segment_words)
            if segment_dict: segments.append(segment_dict)

        self.logger.info(f"Segmentation complete. Produced {len(segments)} segments.")
        return segments


# --- Pipeline ---

class AvtPipeline:
    def __init__(self, config: dict):
        self.config = config.copy()
        self.logger = get_logger(
            "AvtPipeline",
            self.config.get("LOG_LEVEL"),
            self.config.get("LOG_FORMAT")
        )
        # Validate config before initializing components that depend on it (Point 3)
        self._validate_config()

        self.ffmpeg = FFmpegManager(self.config, self.logger)
        self.transcriber = TranscriptionEngine(self.config, self.logger)
        self.translator = TranslationEngine(self.config, self.logger)
        self.idiom_mapper = None # Initialized in run()
        self.segmenter = Segmenter(self.config, self.logger) # Initialize Segmenter
        self.srt_formatter = SRTFormatter(self.config, self.logger)
        self.temp_audio_path = None

    def _validate_config(self): # (Point 3)
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
            raise ConfigError(f"Invalid AUDIO_TARGET_SRATE_HZ: {self.config.get(
'AUDIO_TARGET_SRATE_HZ
')}. Must be positive integer.") from e

        try:
            batch_size = int(self.config.get("MARIANMT_BATCH_SIZE"))
            if batch_size <= 0:
                raise ValueError("Batch size must be positive")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid MARIANMT_BATCH_SIZE: {self.config.get(
'MARIANMT_BATCH_SIZE
')}. Must be positive integer.") from e

        # Check strategy values
        allowed_strategies = [
'fail
', 
'fallback_english
', 
'skip
']
        if self.config.get("TRANSLATION_ERROR_STRATEGY") not in allowed_strategies:
             raise ConfigError(f"Invalid TRANSLATION_ERROR_STRATEGY: {self.config.get(
'TRANSLATION_ERROR_STRATEGY
')}. Allowed: {allowed_strategies}")

        # Check boolean
        if not isinstance(self.config.get("MAPPING_FILE_REQUIRED"), bool):
             raise ConfigError(f"Invalid MAPPING_FILE_REQUIRED: {self.config.get("MAPPING_FILE_REQUIRED")}. Must be True or False.")

        # Validate Segmentation Configs
        allowed_segmentation_strategies = ["pause", "max_words", "combined"]
        seg_strategy = self.config.get("SEGMENTATION_STRATEGY")
        if seg_strategy not in allowed_segmentation_strategies:
            raise ConfigError(f"Invalid SEGMENTATION_STRATEGY: {seg_strategy}. Allowed: {allowed_segmentation_strategies}")

        try:
            pause_threshold = float(self.config.get("SEGMENTATION_PAUSE_THRESHOLD_SEC"))
            if pause_threshold < 0:
                raise ValueError("Pause threshold must be non-negative")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid SEGMENTATION_PAUSE_THRESHOLD_SEC: {self.config.get("SEGMENTATION_PAUSE_THRESHOLD_SEC")}. Must be non-negative number.") from e

        try:
            max_words = int(self.config.get("SEGMENTATION_MAX_WORDS"))
            if max_words <= 0:
                raise ValueError("Max words must be positive")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid SEGMENTATION_MAX_WORDS: {self.config.get("SEGMENTATION_MAX_WORDS")}. Must be positive integer.") from e

        try:
            max_duration = float(self.config.get("SEGMENTATION_MAX_DURATION_SEC"))
            if max_duration < 0:
                raise ValueError("Max duration must be non-negative")
        except (TypeError, ValueError) as e:
            raise ConfigError(f"Invalid SEGMENTATION_MAX_DURATION_SEC: {self.config.get("SEGMENTATION_MAX_DURATION_SEC")}. Must be non-negative number.") from e

        if not isinstance(self.config.get("BYPASS_POST_TRANSLATION_MERGE"), bool):
            raise ConfigError(f"Invalid BYPASS_POST_TRANSLATION_MERGE: {self.config.get("BYPASS_POST_TRANSLATION_MERGE")}. Must be True or False.")

        self.logger.info("Configuration validation successful.")

    # TODO: Review and Simplify this method
    def merge_close_timed_words(self, word_list: List[Dict]) -> List[Dict]:
        if not word_list:
            return []
        self.logger.info(f"Starting subtitle merging process for {len(word_list)} items...")
        time_threshold = self.config.get("SRT_MERGE_TIME_THRESHOLD_SEC")
        min_duration = self.config.get("SRT_MERGE_MIN_DURATION_SEC")
        padding = self.config.get("SRT_TIMESTAMP_PADDING_SEC")
        max_words = self.config.get("SRT_MERGE_MAX_WORDS")
        max_gap = self.config.get("SRT_MERGE_MAX_GAP_SEC")
        merged_list = []
        buffer = []
        for i, current_item in enumerate(word_list):
            try:
                # Basic validation within merge logic (Point 7)
                if not isinstance(current_item, dict) or not all(k in current_item for k in (
'text
', 
'start
', 
'end
')):
                    self.logger.warning(f"Skipping invalid item during merge at index {i}: {current_item}")
                    continue
                current_text = str(current_item[
'text
']).strip()
                current_start = float(current_item[
'start
'])
                current_end = float(current_item[
'end
'])
                if not current_text:
                    self.logger.debug(f"Skipping empty text item in merge: {current_item}")
                    continue

                if not buffer:
                    buffer.append(current_item)
                else:
                    last_item_in_buffer = buffer[-1]
                    last_end = float(last_item_in_buffer[
'end
'])
                    last_start = float(last_item_in_buffer[
'start
'])

                    time_diff_start = current_start - last_start
                    gap_between = current_start - last_end

                    time_condition_met = (gap_between >= 0 and gap_between <= max_gap) or \
                                         (time_diff_start >= 0 and time_diff_start <= time_threshold)
                    current_buffer_word_count = sum(len(str(item[
'text
']).split()) for item in buffer)
                    word_count_ok = (current_buffer_word_count + len(current_text.split())) <= max_words

                    if time_condition_met and word_count_ok:
                        buffer.append(current_item)
                    else:
                        merged_text = " ".join([str(item[
'text
']) for item in buffer])
                        start_time = float(buffer[0][
'start
'])
                        end_time = float(buffer[-1][
'end
']) + padding
                        duration = end_time - start_time
                        if duration < min_duration:
                            end_time = start_time + min_duration
                        merged_list.append({"text": merged_text, "start": start_time, "end": end_time})
                        buffer = [current_item]
            except (KeyError, TypeError, ValueError) as e:
                 self.logger.error(f"Error processing item during merge at index {i}: {e}. Item: {current_item}. Skipping item.", exc_info=False)
                 # Decide whether to flush buffer or just skip item
                 # Flushing buffer might be safer if error indicates data corruption
                 if buffer:
                    merged_text = " ".join([str(item[
'text
']) for item in buffer])
                    start_time = float(buffer[0][
'start
'])
                    end_time = float(buffer[-1][
'end
']) + padding
                    duration = end_time - start_time
                    if duration < min_duration:
                        end_time = start_time + min_duration
                    merged_list.append({"text": merged_text, "start": start_time, "end": end_time})
                 buffer = [] # Reset buffer after error
                 continue # Skip the problematic item

        if buffer:
            merged_text = " ".join([str(item[
'text
']) for item in buffer])
            start_time = float(buffer[0][
'start
'])
            end_time = float(buffer[-1][
'end
']) + padding
            duration = end_time - start_time
            if duration < min_duration:
                end_time = start_time + min_duration
            merged_list.append({"text": merged_text, "start": start_time, "end": end_time})

        self.logger.info(f"Merging complete. Produced {len(merged_list)} subtitle entries.")
        return merged_list

    def run(self, input_video: str, output_srt_path: str, mapping_json_path: str):
        self.logger.info(f"--- Starting AVT Pipeline ---")
        self.logger.info(f"Input Video: {input_video}")
        self.logger.info(f"Output SRT: {output_srt_path}")
        self.logger.info(f"Idiom Mapping: {mapping_json_path}")

        start_time = time.time()
        output_dir = os.path.dirname(output_srt_path) or "."
        # Output dir existence/writability checked in main()
        self.temp_audio_path = os.path.join(output_dir, self.config.get("TEMP_AUDIO_FILENAME"))

        try: # Main pipeline execution block (Point 9)
            # 1. Extract & Normalize Audio (Raises FFmpegError)
            self.logger.info("Step 1: Extracting and Normalizing Audio...")
            normalized_audio = self.ffmpeg.extract_and_normalize_audio(input_video, self.temp_audio_path)
            self.logger.info(f"Audio processed: {normalized_audio}")

            # 2. Transcribe Audio (Raises TranscriptionError, ModelLoadError)
            self.logger.info("Step 2: Transcribing Audio (Word-Level)...")
            word_entries = self.transcriber.get_word_level_timestamps(normalized_audio)
            self.logger.info(f"Transcription complete: {len(word_entries)} words found.")

            # 3. Initialize Idiom Mapper (Raises MappingError)
            self.logger.info("Step 3: Initializing Idiom Mapper...")
            # Pass config to mapper for MAPPING_FILE_REQUIRED check
            self.idiom_mapper = IdiomCsiMapper(mapping_json_path, self.config, self.logger)
            # 4. Apply Idiom Mapping
            self.logger.info("Step 4: Applying Idiom/CSI Mapping...")
            processed_entries = self.idiom_mapper.map_words_to_idioms(word_entries)

            # 5. Segment Entries
            self.logger.info("Step 5: Segmenting entries for translation...")
            # Input to segmenter is the list potentially containing mixed word-level entries (marked NMT)
            # and multi-word idiom entries (marked Idiom/CSI Mapping)
            segments = self.segmenter.segment(processed_entries)

            # 6. Translate NMT Segments (Raises TranslationError, ModelLoadError)
            self.logger.info("Step 6: Translating NMT Segments...")
            nmt_segment_indices = [i for i, seg in enumerate(segments) if seg.get("translation_source") == "NMT_Segment"]
            texts_to_translate = [segments[i]["original_english_text"] for i in nmt_segment_indices]

            if texts_to_translate:
                self.logger.info(f"Found {len(texts_to_translate)} segments needing NMT ({self.translator.error_strategy} strategy).")
                translated_texts = self.translator.translate_batch(texts_to_translate)
                # Validation of count happens inside translate_batch
                if len(translated_texts) != len(nmt_segment_indices):
                    # This should be caught by translate_batch, but double-check
                    raise TranslationError(f"Mismatch after translation. Expected {len(nmt_segment_indices)} translations, got {len(translated_texts)}.")
                
                for i, trans_text in zip(nmt_segment_indices, translated_texts):
                    # Update the segment dictionary with the translated text
                    segments[i]["translated_text"] = trans_text 
                    # Optionally update source to show translation status/strategy
                    # segments[i]["translation_source"] = f"NMT_Segment ({self.translator.error_strategy} applied)" if trans_text != texts_to_translate[nmt_segment_indices.index(i)] else "NMT_Segment (Translated)"
                    segments[i]["translation_source"] = "NMT_Segment (Translated)" # Keep it simple for now

                self.logger.info("NMT segment translation step complete.")
            else:
                self.logger.info("No segments required NMT translation.")

            # Prepare final list for SRT formatting (map segment structure to expected format)
            final_timed_entries = []
            for seg in segments:
                final_timed_entries.append({
                    "text": seg.get("translated_text") or seg.get("original_english_text"), # Use translated if available, else original/mapped
                    "start": seg["start_time"],
                    "end": seg["end_time"]
                    # Add other fields if needed by downstream processes like merging
                })

            # 7. Merge Entries (Optional - based on config)
            if not self.config.get("BYPASS_POST_TRANSLATION_MERGE"): 
                self.logger.info("Step 7: Merging Subtitle Entries (Post-Translation)...")
                # Pass the list of final segments to the merge function
                # Note: The merge logic might need review for segment-level merging effectiveness
                merged_entries = self.merge_close_timed_words(final_timed_entries) 
                if not merged_entries:
                    self.logger.warning("Merging resulted in zero subtitle entries. Output SRT will be empty.")
                entries_for_srt = merged_entries
            else:
                self.logger.info("Step 7: Skipping Post-Translation Merging as configured.")
                entries_for_srt = final_timed_entries # Use the direct segments

            # 8. Format to SRT (Raises SRTFormatError)
            self.logger.info("Step 8: Formatting to SRT...")
            srt_content = self.srt_formatter.create_srt_content(entries_for_srt)
            if not srt_content:
                 self.logger.warning("Final SRT content is empty.")

            # 9. Save SRT File Atomically (Point 9)
            self.logger.info(f"Step 9: Saving SRT file to: {output_srt_path}")
            temp_srt_path = output_srt_path + ".tmp"
            try:
                with open(temp_srt_path, 
'w
', encoding=
'utf-8
') as f:
                    f.write(srt_content)
                # Atomic rename
                os.rename(temp_srt_path, output_srt_path)
                self.logger.info(f"Successfully wrote SRT file: {output_srt_path}")
            except OSError as e:
                self.logger.error(f"Failed to write or rename SRT file {output_srt_path}: {e}")
                # Attempt cleanup of temp file
                if os.path.exists(temp_srt_path):
                    try: os.remove(temp_srt_path)
                    except OSError: pass
                raise PipelineError(f"Failed to save SRT file: {e}") from e

        # Specific exception handling (Point 9)
        except (ConfigError, FFmpegError, ModelLoadError, TranscriptionError, TranslationError, MappingError, SRTFormatError, FileNotFoundError, PermissionError, ValueError) as e:
            self.logger.error(f"Pipeline failed for video 
'{input_video}
'. Error Type: {type(e).__name__}. Message: {e}", exc_info=False) # Log specific error type
            self.logger.debug("Traceback:", exc_info=True) # Log full traceback at debug level
            raise # Re-raise the specific error for main() to catch
        except Exception as e: # Catch-all for truly unexpected errors
            self.logger.critical(f"Unexpected critical error in pipeline for video 
'{input_video}
': {e}", exc_info=True)
            raise PipelineError(f"Unexpected critical error: {e}") from e # Wrap in PipelineError
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

# --- Main Execution Block ---
def main():
    parser = argparse.ArgumentParser(
        description="Generates Indonesian SRT subtitles from an English video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--output", required=True, help="Path to save output SRT file.")
    parser.add_argument("--mapping", required=True, help="Path to JSON idiom/CSI mapping file.")
    # Add args to override config if needed, e.g.:
    # parser.add_argument("--log-level", default=DEFAULT_APP_CONFIG[
'LOG_LEVEL
'], help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    # parser.add_argument("--model-name", default=DEFAULT_APP_CONFIG[
'WHISPER_MODEL_NAME
'], help="Whisper model name")

    args = parser.parse_args()

    # Basic Config Setup
    config = DEFAULT_APP_CONFIG.copy()
    # Override config with args if implemented, e.g.:
    # config[
'LOG_LEVEL
'] = args.log_level
    # config[
'WHISPER_MODEL_NAME
'] = args.model_name

    # Setup logging early
    logger = get_logger(
        "main",
        config.get("LOG_LEVEL", "INFO"),
        config.get("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    try:
        # Input Validation (Point 2)
        logger.info("Validating inputs...")
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Input video not found: {args.video}")
        if not os.access(args.video, os.R_OK):
            raise PermissionError(f"Input video not readable: {args.video}")

        if not os.path.exists(args.mapping):
            raise FileNotFoundError(f"Mapping file not found: {args.mapping}")
        if not os.access(args.mapping, os.R_OK):
            raise PermissionError(f"Mapping file not readable: {args.mapping}")

        output_dir = os.path.dirname(args.output) or "."
        if not os.path.isdir(output_dir):
            logger.info(f"Output directory not found. Attempting to create: {output_dir}")
            try:
                os.makedirs(output_dir, exist_ok=True)
            except OSError as e:
                raise PermissionError(f"Failed to create output directory {output_dir}: {e}") from e
        if not os.access(output_dir, os.W_OK):
             raise PermissionError(f"Output directory is not writable: {output_dir}")
        logger.info("Input validation successful.")

        # Initialize and run pipeline
        logger.info("Initializing pipeline...")
        pipeline = AvtPipeline(config=config)
        pipeline.run(
            input_video=args.video,
            output_srt_path=args.output,
            mapping_json_path=args.mapping
        )
        logger.info("Pipeline completed successfully!")

    # Catch specific pipeline errors and known validation errors (Point 11)
    except (PipelineError, FileNotFoundError, PermissionError, ConfigError) as e:
        logger.error(f"Execution failed: {e}", exc_info=False) # Log concise error
        logger.debug("Traceback:", exc_info=True) # Debug level for full trace
        sys.exit(1) # Exit with error code
    except Exception as e:
        # Catch any other unexpected errors
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

