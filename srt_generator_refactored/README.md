# SRT Generator Refactoring - Phase 1

This project implements Phase 1 of the SRT Generator refactoring, moving from word-by-word translation to segment-by-segment translation for improved quality.

## Overview

The refactored SRT Generator now uses a segment-based approach for translation, which provides better translation quality compared to the previous word-by-word approach. The main changes include:

1. Added new configuration parameters for segmentation and SRT formatting
2. Created a BasicSegmenter class that segments word entries based on pause duration, max words, and punctuation
3. Refactored IdiomCsiMapper to work with segments instead of word entries
4. Refactored TranslationEngine to translate segments instead of individual words
5. Refactored SRTFormatter to create SRT blocks from segments, with support for splitting long segments

## Directory Structure

```
srt_generator_refactor/
├── __init__.py
├── avt_pipeline.py
├── basic_segmenter.py
├── config.py
├── ffmpeg_manager.py
├── idiom_mapper.py
├── main.py
├── srt_formatter.py
├── transcription_engine.py
└── translation_engine.py
```

## Usage

To use the refactored SRT Generator, run the following command:

```bash
python -m srt_generator_refactor.main --video <path_to_video> --output <path_to_output_srt> --mapping <path_to_mapping_json>
```

### Arguments

- `--video`: Path to the input video file
- `--output`: Path to save the output SRT file
- `--mapping`: Path to the JSON idiom/CSI mapping file

## Configuration

The default configuration is defined in `config.py`. The new configuration parameters for Phase 1 are:

- `SEGMENT_MAX_PAUSE_SEC`: Maximum pause between words to end a segment (default: 0.8 seconds)
- `SEGMENT_MAX_WORDS`: Maximum words per segment before forced split (default: 12 words)
- `SRT_MINIMAL_BLOCK_PAUSE_SEC`: Minimal pause between split SRT blocks (default: 0.05 seconds)
- `SRT_MIN_DURATION_PER_BLOCK_SEC`: Minimum duration for any SRT block (default: 1.0 seconds)

## Pipeline Flow

1. Extract and normalize audio using FFmpeg
2. Transcribe audio using Whisper to get word-level timestamps
3. Segment words into coherent segments using BasicSegmenter
4. Apply idiom/CSI mapping to segments using IdiomCsiMapper
5. Translate segments using TranslationEngine
6. Format segments into SRT blocks using SRTFormatter
7. Save the final SRT file

## Dependencies

- Python 3.6+
- torch
- whisper
- transformers
- ffmpeg (command-line tool)

## Future Work

This implementation represents Phase 1 of the refactoring. Future phases may include:

- Improved segmentation based on linguistic features
- Better handling of long segments
- Enhanced idiom mapping
- Support for additional languages

