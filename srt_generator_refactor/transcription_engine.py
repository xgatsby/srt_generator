"""
TranscriptionEngine class for SRT Generator Phase 1 refactoring.

This class handles audio transcription using Whisper.
"""

import logging
import torch
import whisper
from typing import List, Dict, Union

class ModelLoadError(Exception):
    """Error loading ML models (Whisper, MarianMT)."""
    pass

class TranscriptionError(Exception):
    """Error during Whisper transcription."""
    pass

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
            # Raise specific error
            raise ModelLoadError(f"Failed to load Whisper model {self.model_name}: {e}") from e

    def _validate_word_entry(self, word_info: Dict, index: int) -> bool:
        """Validates structure and values of a single word entry."""
        if not isinstance(word_info, dict):
            self.logger.warning(f"Invalid word entry type at index {index}: {type(word_info)}. Skipping.")
            return False
        if not all(k in word_info for k in ('word', 'start', 'end')):  # Ganti 'text' menjadi 'word'
            self.logger.warning(f"Word entry missing required keys at index {index}: {word_info}. Skipping.")
            return False
        try:
            start = float(word_info['start'])
            end = float(word_info['end'])
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
            "temperature": float(self.config.get("WHISPER_TEMPERATURE")),  # Ensure float
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
            # Raise specific error
            raise TranscriptionError(f"Whisper transcription failed for {audio_path}: {e}") from e

        raw_word_entries = []
        if "words" in result and result["words"] and isinstance(result["words"], list):
            self.logger.info("Extracting words from top-level 'words' key.")
            raw_word_entries = result["words"]
        elif "segments" in result and isinstance(result["segments"], list):
            self.logger.info("Extracting words from 'segments'.")
            for segment in result["segments"]:
                if "words" in segment and segment["words"] and isinstance(segment["words"], list):
                    raw_word_entries.extend(segment["words"])
                elif segment.get("text","").strip():  # Fallback for segments without 'words'
                    raw_word_entries.append({
                        "word": segment["text"].strip(),  # Treat segment text as one "word"
                        "start": segment["start"],
                        "end": segment["end"]
                    })

        # Validate and format extracted entries
        word_entries = []
        for i, word_info in enumerate(raw_word_entries):
             if self._validate_word_entry(word_info, i):
                 word_entries.append({
                     "text": str(word_info["word"]).strip(),
                     "start": float(word_info["start"]),
                     "end": float(word_info["end"])
                 })

        # Raise error if no valid words found
        if not word_entries:
            self.logger.error(f"Transcription for {audio_path} yielded no valid word timestamps after processing.")
            self.logger.debug("--- DEBUG TE: WARNING - No valid word entries found after processing ---")
            raise TranscriptionError("Transcription completed but yielded no valid word timestamps.")

        self.logger.info(f"Extracted and validated {len(word_entries)} word-level entries from {audio_path}.")
        
        # Debug logging for word entries
        self.logger.debug("--- DEBUG TE: Raw Word Entries Begin ---")
        display_count = min(5, len(word_entries))
        for i in range(display_count):
            self.logger.debug(f"--- DEBUG TE: RawWordEntry {i}: {word_entries[i]} ---")
        
        if len(word_entries) > display_count:
            self.logger.debug(f"--- DEBUG TE: ... (total {len(word_entries)} entries) ---")
        
        self.logger.debug("--- DEBUG TE: Raw Word Entries End ---")
        
        return word_entries

