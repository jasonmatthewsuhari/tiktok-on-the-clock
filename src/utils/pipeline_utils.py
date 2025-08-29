"""
Common utility functions for pipeline operations.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
import glob

def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validate that a dataframe meets basic requirements.
    
    Args:
        df: Dataframe to validate
        required_columns: List of columns that must be present
        
    Returns:
        bool: True if valid, False otherwise
    """
    if df is None or df.empty:
        logging.error("Dataframe is None or empty")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            return False
    
    logging.info(f"Dataframe validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True

def save_pipeline_metadata(output_dir: str, stage_name: str, metadata: Dict[str, Any]):
    """
    Save metadata about a pipeline stage execution.
    
    Args:
        output_dir: Directory to save metadata
        stage_name: Name of the pipeline stage
        metadata: Dictionary of metadata to save
    """
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metadata_file = output_path / f"{stage_name}_metadata.json"
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
            
        logging.info(f"Saved metadata for {stage_name} to {metadata_file}")
        
    except Exception as e:
        logging.warning(f"Failed to save metadata for {stage_name}: {e}")

def load_pipeline_metadata(output_dir: str, stage_name: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata from a previous pipeline stage execution.
    
    Args:
        output_dir: Directory containing metadata
        stage_name: Name of the pipeline stage
        
    Returns:
        Dictionary of metadata or None if not found
    """
    try:
        metadata_file = Path(output_dir) / f"{stage_name}_metadata.json"
        
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logging.info(f"Loaded metadata for {stage_name} from {metadata_file}")
            return metadata
        else:
            logging.info(f"No metadata found for {stage_name}")
            return None
            
    except Exception as e:
        logging.warning(f"Failed to load metadata for {stage_name}: {e}")
        return None

def get_dataframe_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about a dataframe.
    
    Args:
        df: Dataframe to analyze
        
    Returns:
        Dictionary containing dataframe information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'null_counts': df.isnull().sum().to_dict(),
        'duplicate_count': df.duplicated().sum(),
        'numeric_columns': df.select_dtypes(include=['number']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    # Add basic statistics for numeric columns
    if info['numeric_columns']:
        info['numeric_stats'] = df[info['numeric_columns']].describe().to_dict()
    
    return info

def log_dataframe_summary(df: pd.DataFrame, stage_name: str):
    """
    Log a comprehensive summary of a dataframe.
    
    Args:
        df: Dataframe to summarize
        stage_name: Name of the current stage for logging context
    """
    info = get_dataframe_info(df)
    
    logging.info(f"=== {stage_name} Data Summary ===")
    logging.info(f"Shape: {info['shape']}")
    logging.info(f"Memory usage: {info['memory_usage'] / 1024:.2f} KB")
    logging.info(f"Null values: {sum(info['null_counts'].values())}")
    logging.info(f"Duplicates: {info['duplicate_count']}")
    logging.info(f"Columns: {info['columns']}")
    
    if info['numeric_stats']:
        logging.info("Numeric column statistics:")
        for col, stats in info['numeric_stats'].items():
            logging.info(f"  {col}: mean={stats.get('mean', 'N/A'):.2f}, std={stats.get('std', 'N/A'):.2f}")

def ensure_file_path(file_path: str, create_parent: bool = True) -> Path:
    """
    Ensure a file path exists and optionally create parent directories.
    
    Args:
        file_path: Path to ensure
        create_parent: Whether to create parent directories
        
    Returns:
        Path object
    """
    path = Path(file_path)
    
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    return path

def add_timestamp_to_filename(file_path: str, timestamp: Optional[str] = None) -> str:
    """
    Add a timestamp to a filename.
    
    Args:
        file_path: Original file path
        timestamp: Optional timestamp string (if None, current time will be used)
        
    Returns:
        str: New file path with timestamp
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    path = Path(file_path)
    new_name = f"{path.stem}_{timestamp}{path.suffix}"
    return str(path.parent / new_name)

def find_latest_file_by_pattern(pattern: str) -> Optional[str]:
    """
    Find the most recent file matching a glob pattern.
    
    Args:
        pattern: Glob pattern to search for files
        
    Returns:
        Optional[str]: Path to the most recent file, or None if none found
    """
    files = glob.glob(pattern)
    if not files:
        return None
    
    # Sort by modification time (newest first)
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return files[0]

def get_pipeline_execution_id() -> str:
    """
    Generate a unique execution ID for the current pipeline run.
    
    Returns:
        str: Execution ID in format YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_execution_directory(base_dir: str, execution_id: str) -> str:
    """
    Create a directory for the current pipeline execution.
    
    Args:
        base_dir: Base directory for pipeline outputs
        execution_id: Unique execution identifier
        
    Returns:
        str: Path to the created execution directory
    """
    execution_dir = Path(base_dir) / f"execution_{execution_id}"
    execution_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created execution directory: {execution_dir}")
    return str(execution_dir)
