# SRT Subtitle Generator (Segment-Based - Phase 1)

This project is an SRT subtitle generator that processes English videos to produce Indonesian subtitles. This version implements "Phase 1" of a significant refactoring, transitioning from a less effective word-by-word translation approach to a more professional segment-by-segment translation for improved quality.

## Overview

The core functionality involves:
1.  Extracting audio from a video file using FFmpeg.
2.  Transcribing the audio to text with word-level timestamps using OpenAI Whisper (medium model).
3.  Segmenting the transcribed words into coherent phrases or sentences using a custom `BasicSegmenter`.
4.  Applying idiom/CSI (Culturally Specific Items) mapping to these segments from a user-provided JSON file.
5.  Translating the English segments (those not mapped as idioms) to Indonesian using the Helsinki-NLP MarianMT model.
6.  Formatting the translated segments into SRT subtitle blocks, including logic for splitting long segments and adhering to basic SRT rules.

## Project Structure

The project is organized into a Python package `srt_generator_refactor`:


srt-generator-paperspace/
├── srt_generator_refactor/
│ ├── init.py
│ ├── avt_pipeline.py # Main pipeline orchestration
│ ├── basic_segmenter.py # Logic for segmenting word entries
│ ├── config.py # Default configuration
│ ├── ffmpeg_manager.py # FFmpeg interaction
│ ├── idiom_mapper.py # Idiom/CSI mapping
│ ├── main.py # Command-line entry point
│ ├── srt_formatter.py # SRT block creation and formatting
│ ├── transcription_engine.py # Whisper transcription
│ └── translation_engine.py # MarianMT translation
├── .gitignore
├── input.json # Example mapping file (user-provided)
└── README.md # This file


## Prerequisites

*   Python 3.7+
*   FFmpeg installed and available in your system's PATH.
*   Git (for cloning and managing versions).

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/xgatsby/srt_generator.git
    cd srt_generator
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python dependencies:**
    (Ensure you have a `requirements.txt` file in your project root based on our discussions)
    ```bash
    pip install -r requirements.txt
    ```
    A typical `requirements.txt` might include:
    ```txt
    torch
    torchaudio
    # openai-whisper (or git+https://github.com/openai/whisper.git)
    git+https://github.com/openai/whisper.git
    transformers
    sentencepiece
    sacremoses
    ```

## Configuration

Default configuration parameters are defined in `srt_generator_refactor/config.py`. Key parameters for Phase 1 include:

*   `"WHISPER_MODEL_NAME": "medium"`
*   `"MARIANMT_BATCH_SIZE": 4`
*   `"SEGMENT_MAX_PAUSE_SEC": 0.8`
*   `"SEGMENT_MAX_WORDS": 12`
*   `"SRT_MINIMAL_BLOCK_PAUSE_SEC": 0.05`
*   `"SRT_MIN_DURATION_PER_BLOCK_SEC": 1.0`

You can modify `config.py` directly for persistent changes or a_n_t_i_s_i_p_a_s_i_k_a_n command-line overrides in future versions.

## Usage

Run the script from the root directory of the project (`srt-generator-paperspace/`) using the Python module execution flag `-m`:

```bash
python -m srt_generator_refactor.main --video path/to/your/video.mp4 --output path/to/your/output.srt --mapping path/to/your/input.json

Arguments:
--video: Path to the input English video file.
--output: Path where the generated Indonesian SRT file will be saved.
--mapping: Path to your JSON file containing idiom/CSI mappings (English phrase -> Indonesian translation).
Current Status (Phase 1)
Successfully refactored to a segment-based translation pipeline.
Basic segmentation logic implemented.
Initial support for splitting long SRT segments.
Future Work (Phase 2 and beyond)
Integration of advanced NLP libraries (e.g., spaCy) for more robust sentence segmentation.
Smarter logic for splitting long translated segments into well-timed SRT blocks, including Characters Per Second (CPS) considerations.
Enhanced idiom/terminology management.
More sophisticated error handling and reporting for problematic segments.
Contributing
(Details if you plan for others to contribute - for now, this can be omitted or a simple statement)
Currently, this is a personal project.
License
(Specify a license if you wish, e.g., MIT License - for now, can be omitted)
This project is unlicensed.