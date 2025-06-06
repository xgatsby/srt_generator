"""
IdiomCsiMapper class for SRT Generator Phase 1 refactoring.

This class maps idioms and culturally specific items (CSI) in segments or word entries.
"""

import os
import json
import re
import logging
from typing import List, Dict, Any

class MappingError(Exception):
    """Error related to idiom mapping file or format."""
    pass

class IdiomCsiMapper:
    def __init__(self, mapping_path: str, config: dict, logger: logging.Logger):
        self.mapping_path = mapping_path
        self.config = config
        self.logger = logger
        self.mapping = {}
        self.is_required = self.config.get("MAPPING_FILE_REQUIRED", False)
        self._load_mapping()

    def _load_mapping(self):
        self.logger.info(f"Loading idiom/CSI mapping from: {self.mapping_path}")
        try:
            if not os.path.exists(self.mapping_path):
                if self.is_required:
                    raise MappingError(f"Required mapping file not found: {self.mapping_path}")
                else:
                    self.logger.warning(f"Mapping file not found: {self.mapping_path}. Idiom mapping will be disabled.")
                    return  # Proceed with empty mapping

            with open(self.mapping_path, 'r', encoding='utf-8') as f:
                raw_mapping = json.load(f)

            # Validate format
            if not isinstance(raw_mapping, dict):
                raise MappingError(f"Mapping file content is not a valid JSON dictionary: {self.mapping_path}")

            # Clean keys (strip whitespace)
            self.mapping = {key.strip(): value for key, value in raw_mapping.items()}
            self.logger.info(f"Loaded {len(self.mapping)} idiom/CSI rules from {self.mapping_path}.")

        except json.JSONDecodeError as e:
            log_msg = f"Error decoding JSON mapping file {self.mapping_path}: {e}"
            if self.is_required:
                raise MappingError(log_msg) from e  # Fail if required
            else:
                self.logger.error(log_msg + " Idiom mapping disabled.")
                self.mapping = {}
        except Exception as e:
            log_msg = f"Error loading mapping file {self.mapping_path}: {e}"
            if self.is_required:
                raise MappingError(log_msg) from e  # Fail if required
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
            self.logger.debug("--- DEBUG IM: No idiom mapping available. All segments will proceed to NMT ---")
            return segments
        
        self.logger.info(f"Applying idiom/CSI mapping to {len(segments)} segments...")
        
        # Debug log entry with number of segments received
        self.logger.debug(f"--- DEBUG IM: Starting idiom mapping for {len(segments)} segments with {len(self.mapping)} idiom rules ---")
        
        mapped_segments = []
        idiom_mapped_count = 0
        
        for segment_idx, segment in enumerate(segments):
            # Skip segments that don't have the required fields
            if not isinstance(segment, dict) or not all(k in segment for k in ("original_text", "start_time", "end_time")):
                self.logger.warning(f"Skipping invalid segment during idiom mapping: {segment}")
                mapped_segments.append(segment)
                continue
            
            # Normalize the segment text for matching
            normalized_text = self._normalize_text(segment["original_text"])
            
            # Debug log segment text and normalized form (for first 5 segments)
            if segment_idx < 5:
                self.logger.debug(f"--- DEBUG IM: Segment {segment_idx}: original_text='{segment['original_text']}', normalized_text='{normalized_text}' ---")
            
            # Check if the entire segment matches an idiom
            found_match = False
            idiom_comparisons = 0
            
            for idiom_key, idiom_translation in self.mapping.items():
                normalized_idiom = self._normalize_text(idiom_key)
                
                # Debug log comparison between segment text and idiom keys (for first 5 segments and first 5 idioms)
                if segment_idx < 5 and idiom_comparisons < 5:
                    self.logger.debug(f"--- DEBUG IM: Comparing seg_norm='{normalized_text}' WITH idiom_norm='{normalized_idiom}' (original='{idiom_key}') ---")
                
                idiom_comparisons += 1
                
                if normalized_text == normalized_idiom:
                    # Match found, update the segment
                    self.logger.debug(f"Mapped idiom for segment: '{segment['original_text']}' -> '{idiom_translation}'")
                    
                    # Debug log successful idiom match at INFO level
                    self.logger.info(f"Idiom match found for segment {segment_idx}: '{segment['original_text']}' -> '{idiom_translation}'")
                    
                    segment["translated_text"] = idiom_translation
                    segment["translation_source"] = "IDIOM_MAPPED"
                    found_match = True
                    idiom_mapped_count += 1
                    break
            
            # If no match found, keep the segment as is (NEEDS_NMT)
            if not found_match:
                self.logger.debug(f"No idiom match for segment: '{segment['original_text']}'")
                
                # Debug log segments proceeding to NMT (for first 5 segments)
                if segment_idx < 5:
                    self.logger.debug(f"--- DEBUG IM: No idiom match for segment {segment_idx}. Will proceed to NMT ---")
            
            mapped_segments.append(segment)
        
        # Debug log summary of segments processed
        self.logger.debug(f"--- DEBUG IM: Idiom mapping complete. Processed {len(segments)} segments, {idiom_mapped_count} were mapped to idioms, {len(segments) - idiom_mapped_count} will proceed to NMT ---")
        
        self.logger.info(f"Idiom mapping complete. {idiom_mapped_count} segments mapped.")
        return mapped_segments

    def map_words_to_idioms(self, word_entries: List[Dict]) -> List[Dict]:
        """
        Maps word entries to idioms based on matching phrases.
        
        Args:
            word_entries: List of word entry dictionaries
            
        Returns:
            List of word entry dictionaries with idioms mapped
        """
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
        return output_entries

