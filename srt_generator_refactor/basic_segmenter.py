"""
BasicSegmenter class for SRT Generator Phase 1 refactoring.

This class segments word entries into coherent segments based on pause duration, max words, and punctuation.
"""

import logging
from typing import List, Dict, Any

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
            self.logger.debug("--- DEBUG BS: No word entries provided for segmentation ---")
            return []
        
        self.logger.info(f"Starting basic segmentation for {len(word_entries)} word entries.")
        
        # Debug log entry with total word entries count and config
        self.logger.debug(f"--- DEBUG BS: Starting segmentation with {len(word_entries)} word entries, max_pause_sec={self.max_pause_sec}, max_words={self.max_words} ---")
        
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
            
            # Debug log segment creation details
            self.logger.debug(f"--- DEBUG BS: Creating segment with original_text='{original_text}', start_time={start_time:.3f}, end_time={end_time:.3f}, duration={duration:.3f}, num_words={len(words)} ---")
            
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
                # Debug log first word of a new segment
                self.logger.debug(f"--- DEBUG BS: Starting new segment with first word '{word['text']}' ---")
                continue
            
            # Check if we need to create a new segment
            create_new_segment = False
            
            # Check pause duration between current word and previous word
            prev_word = current_segment_words[-1]
            pause_duration = word["start"] - prev_word["end"]
            
            # Debug log current word evaluation details
            ends_with_punctuation = prev_word["text"].strip().endswith((".", "?", "!"))
            self.logger.debug(f"--- DEBUG BS: Evaluating word '{word['text']}'. Pause={pause_duration:.3f}s, Current segment words={len(current_segment_words)}, Previous word ends with punctuation={ends_with_punctuation} ---")
            
            if pause_duration > self.max_pause_sec:
                self.logger.debug(f"--- DEBUG BS: Creating new segment due to pause: {pause_duration:.3f}s > {self.max_pause_sec:.3f}s ---")
                create_new_segment = True
            
            # Check if current segment has reached max words
            elif len(current_segment_words) >= self.max_words:
                self.logger.debug(f"--- DEBUG BS: Creating new segment due to max words: {len(current_segment_words)} >= {self.max_words} ---")
                create_new_segment = True
            
            # Optional: Check if previous word ends with strong punctuation
            elif prev_word["text"].strip().endswith((".", "?", "!")):
                self.logger.debug(f"--- DEBUG BS: Creating new segment due to punctuation: '{prev_word['text']}' ---")
                create_new_segment = True
            
            # If we need to create a new segment, do it and start a new one
            if create_new_segment:
                # Debug log the list of word texts in current segment
                word_texts = [word["text"] for word in current_segment_words]
                self.logger.debug(f"--- DEBUG BS: Finalizing segment with words: {word_texts} ---")
                
                segment = create_segment(current_segment_words)
                if segment:
                    segments.append(segment)
                current_segment_words = [word]  # Start a new segment with the current word
                
                # Debug log starting a new segment
                self.logger.debug(f"--- DEBUG BS: Starting new segment with word '{word['text']}' ---")
            else:
                # Add the current word to the current segment
                current_segment_words.append(word)
                # Debug log appending a word
                self.logger.debug(f"--- DEBUG BS: Appending word '{word['text']}' to current segment (now {len(current_segment_words)} words) ---")
        
        # Don't forget to add the last segment if there are words left
        if current_segment_words:
            # Debug log the list of word texts in the last segment
            word_texts = [word["text"] for word in current_segment_words]
            self.logger.debug(f"--- DEBUG BS: Finalizing last segment with words: {word_texts} ---")
            
            segment = create_segment(current_segment_words)
            if segment:
                segments.append(segment)
        
        self.logger.info(f"Basic segmentation complete. Created {len(segments)} segments.")
        
        # Debug log segment summary at the end
        self.logger.debug("--- DEBUG BS: Segments Summary Begin ---")
        display_count = min(5, len(segments))
        for i in range(display_count):
            segment = segments[i]
            self.logger.debug(f"--- DEBUG BS: Segment {i}: original_text='{segment['original_text']}', start_time={segment['start_time']:.3f}, end_time={segment['end_time']:.3f} ---")
        
        if len(segments) > display_count:
            self.logger.debug(f"--- DEBUG BS: ... (total {len(segments)} segments) ---")
        
        if not segments:
            self.logger.debug("--- DEBUG BS: WARNING - No segments created ---")
            
        self.logger.debug("--- DEBUG BS: Segments Summary End ---")
        
        return segments

