"""
Checkpoint Manager for handling evaluation progress and recovery from failures.
This module provides robust checkpoint/resume functionality to handle segmentation faults
and other unexpected failures during long-running evaluations.
"""

import os
import json
import fcntl
import time
from typing import Dict, Any, Optional
from pathlib import Path
from fastNLP import logger


class CheckpointManager:
    """
    Manages checkpoint files for evaluation progress tracking and recovery.
    
    Features:
    - Atomic writes with file locking to prevent corruption
    - Progress tracking per sample
    - Automatic recovery from last successful checkpoint
    - Results aggregation across multiple runs
    """
    
    def __init__(self, checkpoint_dir: str, run_identifier: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
            run_identifier: Unique identifier for this evaluation run
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_identifier = run_identifier
        self.checkpoint_file = self.checkpoint_dir / f"{run_identifier}.checkpoint.json"
        self.results_file = self.checkpoint_dir / f"{run_identifier}.results.jsonl"
        
        logger.info(f"Checkpoint manager initialized for run: {run_identifier}")
        logger.info(f"Checkpoint file: {self.checkpoint_file}")
        logger.info(f"Results file: {self.results_file}")
        
    def _atomic_write(self, file_path: Path, data: Dict[str, Any]):
        """
        Atomically write data to file with file locking to prevent corruption.
        
        Args:
            file_path: Path to the file
            data: Dictionary data to write
        """
        temp_file = file_path.with_suffix('.tmp')
        
        try:
            # Write to temporary file first
            with open(temp_file, 'w', encoding='utf-8') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
                # Lock is automatically released when file is closed
            
            # Atomic rename (POSIX guarantees atomicity)
            temp_file.replace(file_path)
            logger.debug(f"Successfully wrote checkpoint to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to write checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise
    
    def _atomic_read(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Read data from file with file locking.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary data or None if file doesn't exist
        """
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Acquire shared lock for reading
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                data = json.load(f)
                # Lock is automatically released when file is closed
            return data
        except Exception as e:
            logger.error(f"Failed to read checkpoint from {file_path}: {e}")
            return None
    
    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load checkpoint data from file.
        
        Returns:
            Dictionary containing checkpoint data with keys:
                - last_completed_idx: Index of last successfully processed sample
                - total_samples: Total number of samples in dataset
                - correct_count: Number of correct predictions so far
                - start_time: Timestamp when evaluation started
                - last_update_time: Timestamp of last checkpoint
        """
        checkpoint = self._atomic_read(self.checkpoint_file)
        
        if checkpoint is None:
            logger.info("No existing checkpoint found, starting fresh evaluation")
            return {
                'last_completed_idx': -1,
                'total_samples': 0,
                'correct_count': 0,
                'start_time': time.time(),
                'last_update_time': time.time(),
                'results': []
            }
        
        logger.info(f"Loaded checkpoint: last_completed_idx={checkpoint['last_completed_idx']}, "
                   f"correct_count={checkpoint['correct_count']}/{checkpoint['total_samples']}")
        return checkpoint
    
    def save_checkpoint(self, idx: int, total_samples: int, correct_count: int, 
                       result: Optional[Dict[str, Any]] = None):
        """
        Save checkpoint after processing a sample.
        
        Args:
            idx: Index of the sample just processed
            total_samples: Total number of samples in dataset
            correct_count: Current count of correct predictions
            result: Optional dictionary containing result details for this sample
        """
        checkpoint = self._atomic_read(self.checkpoint_file)
        
        if checkpoint is None:
            checkpoint = {
                'last_completed_idx': -1,
                'total_samples': total_samples,
                'correct_count': 0,
                'start_time': time.time(),
                'last_update_time': time.time(),
                'results': []
            }
        
        # Update checkpoint
        checkpoint['last_completed_idx'] = idx
        checkpoint['total_samples'] = total_samples
        checkpoint['correct_count'] = correct_count
        checkpoint['last_update_time'] = time.time()
        
        if result is not None:
            checkpoint['results'].append(result)
        
        # Save checkpoint atomically
        self._atomic_write(self.checkpoint_file, checkpoint)
        
        logger.debug(f"Checkpoint saved: sample {idx+1}/{total_samples} completed, "
                    f"accuracy: {correct_count}/{idx+1} = {correct_count/(idx+1)*100:.2f}%")
    
    def save_result(self, idx: int, result: Dict[str, Any]):
        """
        Append a result to the results file (JSONL format).
        
        Args:
            idx: Index of the sample
            result: Dictionary containing result details
        """
        result['idx'] = idx
        result['timestamp'] = time.time()
        
        try:
            with open(self.results_file, 'a', encoding='utf-8') as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                # Lock is automatically released when file is closed
            
            logger.debug(f"Result saved for sample {idx}")
            
        except Exception as e:
            logger.error(f"Failed to save result for sample {idx}: {e}")
    
    def get_resume_index(self) -> int:
        """
        Get the index to resume from (next sample to process).
        
        Returns:
            Index of the next sample to process (0 if starting fresh)
        """
        checkpoint = self.load_checkpoint()
        resume_idx = checkpoint['last_completed_idx'] + 1
        
        if resume_idx > 0:
            logger.info(f"Resuming evaluation from sample index: {resume_idx}")
        else:
            logger.info("Starting evaluation from the beginning")
        
        return resume_idx
    
    def get_correct_count(self) -> int:
        """
        Get the current count of correct predictions from checkpoint.
        
        Returns:
            Number of correct predictions so far
        """
        checkpoint = self.load_checkpoint()
        return checkpoint['correct_count']
    
    def is_completed(self, total_samples: int) -> bool:
        """
        Check if evaluation is already completed.
        
        Args:
            total_samples: Total number of samples in dataset
            
        Returns:
            True if all samples have been processed
        """
        checkpoint = self.load_checkpoint()
        completed = checkpoint['last_completed_idx'] >= total_samples - 1
        
        if completed:
            logger.info(f"Evaluation already completed: {checkpoint['last_completed_idx']+1}/{total_samples} samples")
        
        return completed
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get evaluation statistics from checkpoint.
        
        Returns:
            Dictionary containing statistics
        """
        checkpoint = self.load_checkpoint()
        
        if checkpoint['last_completed_idx'] < 0:
            return {
                'samples_processed': 0,
                'accuracy': 0.0,
                'elapsed_time': 0.0
            }
        
        samples_processed = checkpoint['last_completed_idx'] + 1
        accuracy = checkpoint['correct_count'] / samples_processed if samples_processed > 0 else 0.0
        elapsed_time = checkpoint['last_update_time'] - checkpoint['start_time']
        
        return {
            'samples_processed': samples_processed,
            'correct_count': checkpoint['correct_count'],
            'accuracy': accuracy,
            'elapsed_time': elapsed_time,
            'avg_time_per_sample': elapsed_time / samples_processed if samples_processed > 0 else 0.0
        }
    
    def cleanup(self, keep_results: bool = True):
        """
        Clean up checkpoint files.
        
        Args:
            keep_results: If True, keep the results file, only delete checkpoint
        """
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info(f"Deleted checkpoint file: {self.checkpoint_file}")
        
        if not keep_results and self.results_file.exists():
            self.results_file.unlink()
            logger.info(f"Deleted results file: {self.results_file}")
    
    def finalize(self):
        """
        Finalize the evaluation by logging final statistics.
        """
        stats = self.get_statistics()
        checkpoint = self.load_checkpoint()
        
        logger.info("=" * 80)
        logger.info(f"Evaluation completed for run: {self.run_identifier}")
        logger.info(f"Total samples processed: {stats['samples_processed']}/{checkpoint['total_samples']}")
        logger.info(f"Correct predictions: {stats['correct_count']}")
        logger.info(f"Final accuracy: {stats['accuracy']*100:.2f}%")
        logger.info(f"Total elapsed time: {stats['elapsed_time']:.2f} seconds")
        logger.info(f"Average time per sample: {stats['avg_time_per_sample']:.2f} seconds")
        logger.info("=" * 80)

