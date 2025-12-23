#!/usr/bin/env python3
"""
Wrapper script to run evaluation with automatic retry on segmentation faults.

This script will automatically retry the evaluation when it encounters:
- Segmentation faults (exit code 139 or -11)
- CUDA errors
- Out of memory errors
- Other unexpected crashes

The evaluation will resume from the last checkpoint automatically.
"""

import sys
import subprocess
import time
import signal
import argparse
import logging
from pathlib import Path

# Set up module logger
logger = logging.getLogger(__name__)


def setup_logging():
    """Configure logging for the retry wrapper."""
    import logging
    
    # Configure standard Python logging for the retry wrapper
    # We don't need to configure fastNLP logger here as it's configured in the main script
    
    # Set up stderr logging for the retry wrapper itself
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Get the root logger and configure if needed
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)


def run_with_retry(command, max_retries=10, retry_delay=5):
    """
    Run a command with automatic retry on failures.
    
    Args:
        command: List of command arguments (e.g., ['python', 'script.py', '--arg'])
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    attempt = 0
    total_attempts = max_retries + 1  # Initial attempt + retries
    
    logger.info("="*80)
    logger.info(f"Starting evaluation with automatic retry (max retries: {max_retries})")
    logger.info(f"Command: {' '.join(command)}")
    logger.info("="*80)
    
    while attempt < total_attempts:
        attempt += 1
        
        logger.info("")
        logger.info(f"{'#'*80}")
        logger.info(f"Attempt {attempt}/{total_attempts}")
        logger.info(f"{'#'*80}")
        
        try:
            # Run the command
            logger.info(f"Executing: {' '.join(command)}")
            start_time = time.time()
            
            result = subprocess.run(
                command,
                stdout=sys.stdout,
                stderr=sys.stderr,
                text=True
            )
            
            elapsed_time = time.time() - start_time
            exit_code = result.returncode
            
            logger.info("")
            logger.info(f"Process finished with exit code: {exit_code}")
            logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")
            
            # Check exit code
            if exit_code == 0:
                logger.info("="*80)
                logger.info("Evaluation completed successfully!")
                logger.info("="*80)
                return 0
            
            # Check for segmentation fault
            elif exit_code == 139 or exit_code == -11:
                logger.error("="*80)
                logger.error(f"SEGMENTATION FAULT DETECTED (exit code: {exit_code})")
                logger.error("="*80)
                
            # Check for other errors
            elif exit_code < 0:
                signal_name = signal.Signals(-exit_code).name if -exit_code in [s.value for s in signal.Signals] else "UNKNOWN"
                logger.error("="*80)
                logger.error(f"Process killed by signal: {signal_name} (exit code: {exit_code})")
                logger.error("="*80)
                
            else:
                logger.error("="*80)
                logger.error(f"Process failed with exit code: {exit_code}")
                logger.error("="*80)
            
            # Check if we should retry
            if attempt < total_attempts:
                logger.warning(f"Will retry in {retry_delay} seconds...")
                logger.warning(f"Remaining attempts: {total_attempts - attempt}")
                time.sleep(retry_delay)
            else:
                logger.error("="*80)
                logger.error("Maximum retry attempts reached!")
                logger.error(f"Process failed after {total_attempts} attempts")
                logger.error("="*80)
                return exit_code
                
        except KeyboardInterrupt:
            logger.warning("")
            logger.warning("="*80)
            logger.warning("Interrupted by user (Ctrl+C)")
            logger.warning("="*80)
            return 130
            
        except Exception as e:
            logger.error("")
            logger.error("="*80)
            logger.error(f"Unexpected error while running command: {e}")
            logger.error("="*80)
            
            if attempt < total_attempts:
                logger.warning(f"Will retry in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("Maximum retry attempts reached!")
                return 1
    
    return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run evaluation with automatic retry on failures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings (10 retries)
  python run_with_retry.py python evaluate_softcot.py --base_model_id ... --task_name gsm8k
  
  # Run with custom retry settings
  python run_with_retry.py --max-retries 5 --retry-delay 10 python evaluate_softcot.py ...
        """
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=10,
        help='Maximum number of retry attempts (default: 10)'
    )
    
    parser.add_argument(
        '--retry-delay',
        type=int,
        default=5,
        help='Delay in seconds between retries (default: 5)'
    )
    
    parser.add_argument(
        'command',
        nargs=argparse.REMAINDER,
        help='Command to execute (e.g., python evaluate_softcot.py ...)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Validate command
    if not args.command:
        logger.error("Error: No command specified!")
        parser.print_help()
        return 1
    
    # Run with retry
    exit_code = run_with_retry(
        args.command,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay
    )
    
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

