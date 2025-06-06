# SRT Generator Refactoring Requirements - Phase 1

## New Configuration Parameters
- `"SEGMENT_MAX_PAUSE_SEC": 0.8` (float: max pause between words to end a segment)
- `"SEGMENT_MAX_WORDS": 12` (int: max words per segment before forced split)
- `"SRT_MINIMAL_BLOCK_PAUSE_SEC": 0.05` (float: minimal pause between split SRT blocks)
- `"SRT_MIN_DURATION_PER_BLOCK_SEC": 1.0` (float: minimum duration for any SRT block)

## New Component: BasicSegmenter
- Input: List of word_entries (dictionaries with 'text', 'start', 'end' keys)
- Logic: Create segments based on pause duration, max words, and punctuation
- Output: List of segment dictionaries with:
  - original_words: List of original word entry dicts
  - original_text: Concatenated text
  - start_time: Start time of first word
  - end_time: End time of last word
  - duration: end_time - start_time
  - translation_source: "NEEDS_NMT" (initial status)
  - translated_text: None (to be filled by TranslationEngine)

## Modify AvtPipeline.run() Workflow
- After getting word_entries, call segmenter_basic to produce segments
- Pass segments to subsequent steps (idiom mapping, translation, SRT formatting)

## Modify IdiomCsiMapper
- Refactor to map_segments_for_idioms(self, segments: List[Dict]) -> List[Dict]
- Match entire segment['original_text'] against idioms
- If match found:
  - Set segment['translated_text'] to idiom's translation
  - Set segment['translation_source'] to "IDIOM_MAPPED"
- If no match, segment['translation_source'] remains "NEEDS_NMT"

## Modify TranslationEngine
- Refactor to translate_segments(self, segments: List[Dict]) -> List[Dict]
- Create batch of texts to translate from segments with "NEEDS_NMT" status
- Translate batch using MarianMT
- Populate segment['translated_text'] with NMT output
- Update segment['translation_source'] to "NMT_TRANSLATED"
- Handle translation errors based on TRANSLATION_ERROR_STRATEGY

## Modify SRTFormatter
- Refactor to create_srt_from_segments(self, segments: List[Dict]) -> str
- For each segment:
  - Format translated_text using _format_text_for_srt
  - Handle splitting for long segments:
    - If formatted_lines <= SRT_MAX_LINES_PER_BLOCK: Create one SRT block
    - If formatted_lines > SRT_MAX_LINES_PER_BLOCK: Split into multiple blocks
  - Ensure minimum duration (SRT_MIN_DURATION_PER_BLOCK_SEC)
  - Add minimal pause between blocks (SRT_MINIMAL_BLOCK_PAUSE_SEC)

## Maintain Existing Features
- Logging (info, debug, warning, error)
- Custom exception handling
- argparse main execution block
- Cleanup of temporary files

