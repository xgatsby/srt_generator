"""
Test script for SRT Generator Phase 1 refactoring.

This script tests the refactored SRT generator by creating mock data and running it through the pipeline.
"""

import logging
import unittest
from typing import List, Dict

# Import components to test
from srt_generator_refactor.basic_segmenter import BasicSegmenter
from srt_generator_refactor.idiom_mapper import IdiomCsiMapper
from srt_generator_refactor.translation_engine import TranslationEngine
from srt_generator_refactor.srt_formatter import SRTFormatter
from srt_generator_refactor.config import DEFAULT_APP_CONFIG

# Mock data and utilities
def get_mock_logger():
    """Create a mock logger for testing."""
    logger = logging.getLogger("test")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def get_mock_word_entries() -> List[Dict]:
    """Create mock word entries for testing."""
    return [
        {"text": "This", "start": 0.0, "end": 0.5},
        {"text": "is", "start": 0.6, "end": 0.9},
        {"text": "a", "start": 1.0, "end": 1.2},
        {"text": "test", "start": 1.3, "end": 1.8},
        {"text": "of", "start": 1.9, "end": 2.1},
        {"text": "the", "start": 2.2, "end": 2.4},
        {"text": "segmentation", "start": 2.5, "end": 3.2},
        {"text": "system.", "start": 3.3, "end": 4.0},
        # Long pause to trigger new segment
        {"text": "It", "start": 5.0, "end": 5.2},
        {"text": "should", "start": 5.3, "end": 5.6},
        {"text": "create", "start": 5.7, "end": 6.0},
        {"text": "multiple", "start": 6.1, "end": 6.5},
        {"text": "segments", "start": 6.6, "end": 7.0},
        {"text": "based", "start": 7.1, "end": 7.4},
        {"text": "on", "start": 7.5, "end": 7.7},
        {"text": "pauses", "start": 7.8, "end": 8.2},
        {"text": "and", "start": 8.3, "end": 8.5},
        {"text": "punctuation.", "start": 8.6, "end": 9.2},
    ]

def get_mock_mapping():
    """Create mock idiom mapping for testing."""
    return {
        "this is a test": "Ini adalah ujian",
        "multiple segments": "segmen berganda"
    }

class TestBasicSegmenter(unittest.TestCase):
    """Test the BasicSegmenter class."""
    
    def setUp(self):
        self.config = DEFAULT_APP_CONFIG.copy()
        self.logger = get_mock_logger()
        self.segmenter = BasicSegmenter(self.config, self.logger)
        self.word_entries = get_mock_word_entries()
    
    def test_segment_words(self):
        """Test segmenting words into segments."""
        segments = self.segmenter.segment_words(self.word_entries)
        
        # Check that segments were created
        self.assertGreater(len(segments), 0)
        
        # Check that each segment has the required fields
        for segment in segments:
            self.assertIn("original_words", segment)
            self.assertIn("original_text", segment)
            self.assertIn("start_time", segment)
            self.assertIn("end_time", segment)
            self.assertIn("duration", segment)
            self.assertIn("translation_source", segment)
            self.assertIn("translated_text", segment)
            
            # Check that duration is calculated correctly
            self.assertAlmostEqual(segment["duration"], segment["end_time"] - segment["start_time"])
            
            # Check that translation_source is set to NEEDS_NMT
            self.assertEqual(segment["translation_source"], "NEEDS_NMT")
            
            # Check that translated_text is None
            self.assertIsNone(segment["translated_text"])
        
        # Check that we have at least 2 segments due to the long pause
        self.assertGreaterEqual(len(segments), 2)
        
        # Print segments for inspection
        print(f"Created {len(segments)} segments:")
        for i, segment in enumerate(segments):
            print(f"Segment {i+1}: '{segment['original_text']}' ({segment['start_time']:.2f}s - {segment['end_time']:.2f}s)")

class TestIdiomMapper(unittest.TestCase):
    """Test the IdiomCsiMapper class."""
    
    def setUp(self):
        self.config = DEFAULT_APP_CONFIG.copy()
        self.logger = get_mock_logger()
        # Create a mock mapping file
        import tempfile
        import json
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json")
        json.dump(get_mock_mapping(), self.temp_file)
        self.temp_file.close()
        self.mapper = IdiomCsiMapper(self.temp_file.name, self.config, self.logger)
        
        # Create mock segments
        self.segmenter = BasicSegmenter(self.config, self.logger)
        self.segments = self.segmenter.segment_words(get_mock_word_entries())
    
    def tearDown(self):
        import os
        os.unlink(self.temp_file.name)
    
    def test_map_segments_for_idioms(self):
        """Test mapping segments to idioms."""
        mapped_segments = self.mapper.map_segments_for_idioms(self.segments)
        
        # Check that segments were mapped
        self.assertEqual(len(mapped_segments), len(self.segments))
        
        # Check if any segments were mapped to idioms
        idiom_mapped = False
        for segment in mapped_segments:
            if segment["translation_source"] == "IDIOM_MAPPED":
                idiom_mapped = True
                self.assertIsNotNone(segment["translated_text"])
        
        # Print mapped segments for inspection
        print(f"Mapped {len(mapped_segments)} segments:")
        for i, segment in enumerate(mapped_segments):
            source = segment["translation_source"]
            text = segment["translated_text"] if segment["translated_text"] else "None"
            print(f"Segment {i+1}: '{segment['original_text']}' -> '{text}' ({source})")

def run_tests():
    """Run the tests."""
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()

