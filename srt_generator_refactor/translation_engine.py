"""
TranslationEngine class for SRT Generator Phase 1 refactoring.

This class handles translation of segments using MarianMT.
"""

import logging
import torch
from typing import List, Dict, Any
from transformers import MarianMTModel, MarianTokenizer

class ModelLoadError(Exception):
    """Error loading ML models (Whisper, MarianMT)."""
    pass

class TranslationError(Exception):
    """Error during MarianMT translation."""
    pass

class TranslationEngine:
    def __init__(self, config: dict, logger: logging.Logger):
        self.model_path = config.get("MARIANMT_MODEL_PATH_EN_ID")
        self.device = config.get("MARIANMT_DEVICE")
        self.batch_size = config.get("MARIANMT_BATCH_SIZE")
        self.error_strategy = config.get("TRANSLATION_ERROR_STRATEGY", "fail")
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
            raise ModelLoadError(f"Failed to load MarianMT model {self.model_path}: {e}") from e

    def translate_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        Translates segments that need NMT translation.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of segment dictionaries with translations
        """
        if not self.model or not self.tokenizer:
            raise ModelLoadError("MarianMT model/tokenizer not loaded. Cannot translate.")
        
        self.logger.info(f"Starting segment translation for {len(segments)} segments...")
        
        # Debug log entry with number of segments received
        self.logger.debug(f"--- DEBUG TE: Starting translation for {len(segments)} segments ---")
        
        # Identify segments that need translation
        segments_to_translate = []
        segment_indices = []
        
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                self.logger.warning(f"Skipping invalid segment at index {i}: {segment}")
                continue
                
            if segment.get("translation_source") == "NEEDS_NMT":
                segments_to_translate.append(segment["original_text"])
                segment_indices.append(i)
        
        self.logger.info(f"Found {len(segments_to_translate)} segments needing NMT translation.")
        
        # Debug log segments needing NMT
        self.logger.debug(f"--- DEBUG TE: Found {len(segments_to_translate)} segments needing NMT translation ---")
        
        # Log the texts to be translated (up to 5)
        display_count = min(5, len(segments_to_translate))
        if display_count > 0:
            for i in range(display_count):
                segment_idx = segment_indices[i]
                self.logger.debug(f"--- DEBUG TE: Segment {segment_idx} NMT Input: '{segments[segment_idx]['original_text']}' ---")
            
            if len(segments_to_translate) > display_count:
                self.logger.debug(f"--- DEBUG TE: ... (total {len(segments_to_translate)} segments for NMT) ---")
        
        # If no segments need translation, return the original segments
        if not segments_to_translate:
            self.logger.debug("--- DEBUG TE: No segments need translation, returning original segments ---")
            return segments
        
        # Translate the segments
        try:
            translated_texts = self.translate_batch(segments_to_translate)
            
            # Update the segments with translations
            for i, (idx, trans_text) in enumerate(zip(segment_indices, translated_texts)):
                segments[idx]["translated_text"] = trans_text
                segments[idx]["translation_source"] = "NMT_TRANSLATED"
                
                # Debug log translated segments (up to 5)
                if i < 5:
                    self.logger.debug(f"--- DEBUG TE: Segment {idx} NMT Output: '{trans_text}' ---")
            
            # Debug log summary of segments processed
            self.logger.debug(f"--- DEBUG TE: Successfully translated {len(translated_texts)} segments ---")
            
            self.logger.info(f"Successfully translated {len(translated_texts)} segments.")
            return segments
            
        except Exception as e:
            self.logger.error(f"Error during segment translation: {e}")
            raise TranslationError(f"Failed to translate segments: {e}") from e

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
                    # Handle error based on strategy
                    self.logger.error(f"Error translating batch starting at index {batch_indices[0]}: {batch_error}")
                    if self.error_strategy == 'fail':
                        raise TranslationError(f"Translation failed for batch starting at index {batch_indices[0]}: {batch_error}") from batch_error
                    elif self.error_strategy == 'fallback_english':
                        self.logger.warning(f"Falling back to English for failed batch (indices {batch_indices}).")
                        translated_texts.extend(batch_texts)  # Add original English texts
                    elif self.error_strategy == 'skip':
                        self.logger.warning(f"Skipping failed translation batch (indices {batch_indices}).")
                        translated_texts.extend(["" for _ in batch_texts])  # Add empty strings
                    else:  # Default to fail for unknown strategy
                        raise TranslationError(f"Unknown TRANSLATION_ERROR_STRATEGY '{self.error_strategy}'. Failing batch.")

            # Final validation
            if len(translated_texts) != len(texts):
                raise TranslationError(f"Mismatch in translation input/output count. Expected {len(texts)}, got {len(translated_texts)}.")

            self.logger.debug(f"Finished translating {len(texts)} texts.")
            return translated_texts

        except Exception as e:
            # Catch unexpected errors during the batching loop itself
            self.logger.error(f"Unexpected error during translation batch processing: {e}")
            raise TranslationError(f"Unexpected error during translation: {e}") from e

