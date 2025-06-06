"""
Main entry point for SRT Generator Phase 1 refactoring.

This script provides the command-line interface for the SRT generator.
"""

import os
import sys
import argparse
import logging

# Import custom exceptions and components
from srt_generator_refactor.avt_pipeline import (
    AvtPipeline, PipelineError, ConfigError, get_logger
)

# Import default configuration
from srt_generator_refactor.config import DEFAULT_APP_CONFIG

def main():
    parser = argparse.ArgumentParser(
        description="Generates Indonesian SRT subtitles from an English video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--output", required=True, help="Path to save output SRT file.")
    parser.add_argument("--mapping", required=True, help="Path to JSON idiom/CSI mapping file.")
    # Add args to override config if needed, e.g.:
    # parser.add_argument("--log-level", default=DEFAULT_APP_CONFIG['LOG_LEVEL'], help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    # parser.add_argument("--model-name", default=DEFAULT_APP_CONFIG['WHISPER_MODEL_NAME'], help="Whisper model name")

    args = parser.parse_args()

    # Basic Config Setup
    config = DEFAULT_APP_CONFIG.copy()
    # Override config with args if implemented, e.g.:
    # config['LOG_LEVEL'] = args.log_level
    # config['WHISPER_MODEL_NAME'] = args.model_name

    # Setup logging early
    logger = get_logger(
        "main",
        config.get("LOG_LEVEL", "INFO"),
        config.get("LOG_FORMAT", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    try:
        # Input Validation
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

    # Catch specific pipeline errors and known validation errors
    except (PipelineError, FileNotFoundError, PermissionError, ConfigError) as e:
        logger.error(f"Execution failed: {e}", exc_info=False)  # Log concise error
        logger.debug("Traceback:", exc_info=True)  # Debug level for full trace
        sys.exit(1)  # Exit with error code
    except Exception as e:
        # Catch any other unexpected errors
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

