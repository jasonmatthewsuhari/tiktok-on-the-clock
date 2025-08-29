"""
Pipeline Stage 1: Take Input CSV
This module handles reading and initial processing of CSV input data.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

def run(config: Dict[str, Any]) -> bool:
    """
    Execute the first pipeline stage.
    
    Args:
        config: Configuration dictionary containing:
            - input_file: Path to input CSV file
            - output_file: Path to output CSV file
            
    Returns:
        bool: True if successful, False otherwise
    """
    input_file = config['input_file']
    output_file = config['output_file']
    execution_output_dir = config.get('execution_output_dir')
    
    # Generate timestamp for metadata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use execution directory if provided, otherwise use timestamped filename
    if execution_output_dir:
        output_path = Path(execution_output_dir)
        output_file = str(output_path / "stage1_output.csv")
    else:
        output_path = Path(output_file)
        output_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
        output_file = str(output_path.parent / output_name)
    
    logging.info(f"Stage 1: Reading input from {input_file}")
    logging.info(f"Stage 1: Output will be saved to {output_file}")
    
    # Read CSV file
    df = pd.read_csv(input_file, encoding='latin-1', sep=';')
    logging.info(f"Loaded {len(df)} rows from {input_file}")
    
    # Save processed data
    df.to_csv(output_file, sep=';', index=False, encoding='latin-1')
    logging.info(f"Stage 1: Saved {len(df)} rows to {output_file}")
    
    # Log summary statistics
    logging.info(f"Stage 1 Summary:")
    logging.info(f"  - Total rows: {len(df)}")
    logging.info(f"  - Columns: {list(df.columns)}")
    logging.info(f"  - Data types: {df.dtypes.to_dict()}")
    
    return True

if __name__ == "__main__":
    # Test the module independently
    test_config = {
        'input_file': 'data/input.csv',
        'output_file': 'data/stage1_output.csv'
    }
    success = run(test_config)
    print(f"Stage 1 execution: {'SUCCESS' if success else 'FAILED'}")
