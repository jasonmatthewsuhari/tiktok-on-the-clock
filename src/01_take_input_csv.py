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
    
    # Read CSV file (new schema uses comma separation)
    df = pd.read_csv(input_file, encoding='utf-8')
    logging.info(f"Loaded {len(df)} rows from {input_file}")
    
    # Map old column names to new schema for compatibility
    column_mapping = {
        'name_y': 'business_name',
        'name_x': 'author_name', 
        # text, rating, pics stay the same
    }
    
    # Rename columns if they exist
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist, create if missing
    required_columns = ['business_name', 'author_name', 'text', 'rating', 'pics']
    for col in required_columns:
        if col not in df.columns:
            df[col] = ''
            logging.warning(f"Missing column '{col}' - created empty column")
    
    # Process category field if available (convert from JSON array to string)
    if 'category' in df.columns:
        df['category'] = df['category'].astype(str).str.replace(r'[\[\]\']', '', regex=True)
        df['category'] = df['category'].str.replace(',', ';').fillna('unknown')
    else:
        df['category'] = 'unknown'
    
    # Extract temporal features from Unix timestamp
    if 'time' in df.columns:
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['review_datetime'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
        df['hour_of_day'] = df['review_datetime'].dt.hour
        df['day_of_week'] = df['review_datetime'].dt.dayofweek  # 0=Monday
        df['month'] = df['review_datetime'].dt.month
        df['year'] = df['review_datetime'].dt.year
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        logging.info("Extracted temporal features: hour_of_day, day_of_week, month, year, is_weekend")
    
    # Process business response field
    if 'resp' in df.columns:
        df['has_business_response'] = (~df['resp'].isna() & (df['resp'] != '')).astype(int)
        df['response_length'] = df['resp'].astype(str).str.len().fillna(0)
        logging.info("Extracted response features: has_business_response, response_length")
    
    # Extract text quality features
    if 'text' in df.columns:
        df['text_length'] = df['text'].astype(str).str.len()
        df['word_count'] = df['text'].astype(str).str.split().str.len()
        df['exclamation_count'] = df['text'].astype(str).str.count('!')
        df['caps_ratio'] = df['text'].astype(str).str.count(r'[A-Z]') / df['text_length'].clip(lower=1)
        logging.info("Extracted text features: text_length, word_count, exclamation_count, caps_ratio")
    
    # Derive business intelligence features  
    if 'avg_rating' in df.columns and 'rating' in df.columns:
        df['rating_deviation'] = abs(df['rating'] - df['avg_rating'])
        logging.info("Calculated rating_deviation feature")
    
    # Clean numeric fields
    numeric_columns = ['rating', 'avg_rating', 'num_of_reviews', 'latitude', 'longitude', 
                      'hour_of_day', 'day_of_week', 'month', 'year', 'text_length', 'word_count']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean text data to remove problematic characters
    text_columns = ['text', 'business_name', 'author_name', 'category']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\x00-\x7F]+', ' ', regex=True)  # Remove non-ASCII
    
    # Save processed data with UTF-8 encoding
    df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
    logging.info(f"Stage 1: Saved {len(df)} rows to {output_file}")
    
    # Log summary statistics
    logging.info(f"Stage 1 Summary:")
    logging.info(f"  - Total rows: {len(df)}")
    logging.info(f"  - Original columns: {len(df.columns)}")
    logging.info(f"  - Key columns available: {[col for col in required_columns if col in df.columns]}")
    logging.info(f"  - Rich features: avg_rating={('avg_rating' in df.columns)}, category={('category' in df.columns)}, location={('state' in df.columns)}")
    
    return True

if __name__ == "__main__":
    # Test the module independently
    test_config = {
        'input_file': 'data/input.csv',
        'output_file': 'data/stage1_output.csv'
    }
    success = run(test_config)
    print(f"Stage 1 execution: {'SUCCESS' if success else 'FAILED'}")
