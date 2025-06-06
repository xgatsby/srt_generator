"""
SRTFormatter class for SRT Generator Phase 1 refactoring.

This class formats segments into SRT blocks.
"""

import logging
import math
from typing import List, Dict, Any

class SRTFormatError(Exception):
    """Error during SRT formatting or timestamp issues."""
    pass

class SRTFormatter:
    def __init__(self, config: dict, logger: logging.Logger):
        self.max_chars_per_line = config.get("SRT_MAX_CHARS_PER_LINE")
        self.max_lines_per_block = config.get("SRT_MAX_LINES_PER_BLOCK")
        self.min_duration_per_block = config.get("SRT_MIN_DURATION_PER_BLOCK_SEC", 1.0)
        self.minimal_block_pause = config.get("SRT_MINIMAL_BLOCK_PAUSE_SEC", 0.05)
        self.config = config
        self.logger = logger

    def _format_timestamp(self, seconds: float) -> str:
        # Validate timestamp
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

    def _format_text_for_srt(self, text: str) -> List[str]:
        """
        Format text into lines that fit within max_chars_per_line.
        
        Args:
            text: Text to format
            
        Returns:
            List of formatted lines
        """
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
            
        return lines

    def create_srt_from_segments(self, segments: List[Dict]) -> str:
        """
        Create SRT content from segments.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            SRT content as a string
        """
        if not isinstance(segments, list):
            raise SRTFormatError(f"Input segments must be a list, got {type(segments)}")
        
        self.logger.info(f"Creating SRT content from {len(segments)} segments...")
        
        srt_blocks = []
        block_index = 1  # SRT block counter
        
        for segment in segments:
            # Skip segments that don't have the required fields or have empty text
            if not isinstance(segment, dict) or not all(k in segment for k in ("translated_text", "start_time", "end_time")):
                self.logger.warning(f"Skipping invalid segment during SRT formatting: {segment}")
                continue
                
            if not segment["translated_text"] or not str(segment["translated_text"]).strip():
                self.logger.debug(f"Skipping empty text segment: {segment}")
                continue
            
            # Format the text into lines
            formatted_lines = self._format_text_for_srt(str(segment["translated_text"]))
            if not formatted_lines:
                self.logger.debug(f"Skipping segment with no formatted lines: {segment}")
                continue
            
            # Determine if we need to split this segment into multiple SRT blocks
            num_target_lines_per_block = self.max_lines_per_block
            
            if len(formatted_lines) <= num_target_lines_per_block:
                # Simple case: one SRT block for this segment
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                
                # Ensure minimum duration
                if end_time - start_time < self.min_duration_per_block:
                    end_time = start_time + self.min_duration_per_block
                
                start_time_str = self._format_timestamp(start_time)
                end_time_str = self._format_timestamp(end_time)
                
                block_text = "\n".join(formatted_lines)
                srt_block = f"{block_index}\n{start_time_str} --> {end_time_str}\n{block_text}\n"
                srt_blocks.append(srt_block)
                block_index += 1
            else:
                # Complex case: split into multiple SRT blocks
                num_srt_blocks = math.ceil(len(formatted_lines) / num_target_lines_per_block)
                duration_per_block = segment["end_time"] - segment["start_time"]
                if duration_per_block <= 0:
                    duration_per_block = self.min_duration_per_block
                duration_per_block_approx = duration_per_block / num_srt_blocks
                
                current_block_start_time = segment["start_time"]
                
                for i in range(num_srt_blocks):
                    start_idx = i * num_target_lines_per_block
                    end_idx = min((i + 1) * num_target_lines_per_block, len(formatted_lines))
                    current_block_lines = formatted_lines[start_idx:end_idx]
                    
                    # Calculate end time for this block
                    is_last_block = (i == num_srt_blocks - 1)
                    if is_last_block:
                        current_block_end_time = segment["end_time"]
                    else:
                        current_block_end_time = current_block_start_time + duration_per_block_approx
                    
                    # Ensure minimum duration
                    if current_block_end_time - current_block_start_time < self.min_duration_per_block:
                        current_block_end_time = current_block_start_time + self.min_duration_per_block
                    
                    start_time_str = self._format_timestamp(current_block_start_time)
                    end_time_str = self._format_timestamp(current_block_end_time)
                    
                    block_text = "\n".join(current_block_lines)
                    srt_block = f"{block_index}\n{start_time_str} --> {end_time_str}\n{block_text}\n"
                    srt_blocks.append(srt_block)
                    block_index += 1
                    
                    # Set start time for next block
                    if not is_last_block:
                        current_block_start_time = current_block_end_time + self.minimal_block_pause
        
        self.logger.info(f"SRT formatting complete. Created {len(srt_blocks)} SRT blocks.")
        return "\n".join(srt_blocks)

    def create_srt_content(self, timed_entries: List[Dict]) -> str:
        """
        Create SRT content from timed entries (legacy method).
        
        Args:
            timed_entries: List of timed entry dictionaries
            
        Returns:
            SRT content as a string
        """
        # Validate input type
        if not isinstance(timed_entries, list):
            raise SRTFormatError(f"Input timed_entries must be a list, got {type(timed_entries)}")

        srt_blocks = []
        for i, entry in enumerate(timed_entries):
            try:
                # Validate entry structure and content
                if not isinstance(entry, dict):
                    self.logger.warning(f"Skipping invalid entry at index {i}: Not a dictionary ({type(entry)}).")
                    continue
                if not all(k in entry for k in ('text', 'start', 'end')):
                    self.logger.warning(f"Skipping invalid entry at index {i}: Missing keys ({entry.keys()}).")
                    continue
                if not entry['text'] or not str(entry['text']).strip():
                    self.logger.debug(f"Skipping empty text entry at index {i}")
                    continue

                start_time_str = self._format_timestamp(entry['start'])
                end_time_str = self._format_timestamp(entry['end'])

                # Basic duration check
                if entry['end'] < entry['start']:
                     self.logger.warning(f"Correcting negative duration at index {i}: start={entry['start']}, end={entry['end']}. Setting end = start.")
                     end_time_str = start_time_str # Or use start + minimal duration

                formatted_text = "\n".join(self._format_text_for_srt(str(entry['text'])))

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

