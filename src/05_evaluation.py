"""
Pipeline Stage 5: Evaluation
This module evaluates the pipeline performance by comparing relevance-filtered results against original labels.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import glob
from datetime import datetime
import json

def find_latest_file(pattern: str) -> str:
    """
    Find the most recent file matching a pattern.
    
    Args:
        pattern: Glob pattern to search for files
        
    Returns:
        str: Path to the most recent file
    """
    files = glob.glob(pattern)
    # Sort by modification time (newest first)
    files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
    return files[0]

def load_original_data(input_file: str) -> pd.DataFrame:
    """
    Load the original dataset.
    
    Args:
        input_file: Path to original input file
        
    Returns:
        DataFrame: Original dataset with all labels
    """
    # Load data with UTF-8 encoding and comma separator
    df = pd.read_csv(input_file, encoding='utf-8')
    logging.info(f"Loaded original data with UTF-8 encoding and comma separator")
    
    return df

def calculate_metrics(original_df: pd.DataFrame, filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate evaluation metrics comparing filtered results to original labels.
    
    Args:
        original_df: Original dataset with true labels
        filtered_df: Filtered dataset from pipeline
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Check what label values exist in the dataset
    unique_labels = original_df['Label'].value_counts()
    logging.info(f"Original dataset label distribution: {dict(unique_labels)}")
    
    # Map common label variations to standard format
    label_mapping = {
        'valid': 'Yes',
        'invalid': 'No',
        'error': 'No',  # Treat errors as invalid/No
        'Valid': 'Yes', 
        'Invalid': 'No',
        'Error': 'No',
        'VALID': 'Yes',
        'INVALID': 'No',
        'ERROR': 'No',
        '1': 'Yes',
        '0': 'No',
        1: 'Yes',
        0: 'No',
        True: 'Yes',
        False: 'No'
    }
    
    # Apply label mapping
    original_df['Label_normalized'] = original_df['Label'].map(label_mapping).fillna(original_df['Label'])
    
    # Get original label counts with normalized labels
    original_yes = len(original_df[original_df['Label_normalized'] == 'Yes'])
    original_no = len(original_df[original_df['Label_normalized'] == 'No'])
    original_total = len(original_df)
    
    logging.info(f"After normalization - Yes: {original_yes}, No: {original_no}, Total: {original_total}")
    
    # Get filtered counts (everything kept should theoretically be "Yes")
    filtered_total = len(filtered_df)
    
    # Use a more reliable matching strategy - try multiple identifiers
    logging.info("Attempting to match records using available identifiers...")
    
    # Try matching by index first (if available)
    if 'Unnamed: 0' in original_df.columns and 'Unnamed: 0' in filtered_df.columns:
        logging.info("Using index-based matching...")
        original_labels = dict(zip(original_df['Unnamed: 0'], original_df['Label_normalized']))
        
        kept_yes = 0
        kept_no = 0
        matched_count = 0
        
        for _, row in filtered_df.iterrows():
            idx = row['Unnamed: 0']
            if idx in original_labels:
                original_label = original_labels[idx]
                matched_count += 1
                if original_label == 'Yes':
                    kept_yes += 1
                elif original_label == 'No':
                    kept_no += 1
        
        logging.info(f"Matched {matched_count}/{len(filtered_df)} records using index")
        
    # Fallback to user_id + business combo if index matching fails
    elif 'user_id' in original_df.columns and 'gmap_id' in original_df.columns:
        logging.info("Using user_id + gmap_id matching...")
        original_labels = {}
        for _, row in original_df.iterrows():
            key = f"{row['user_id']}_{row['gmap_id']}"
            original_labels[key] = row['Label_normalized']
        
        kept_yes = 0
        kept_no = 0
        matched_count = 0
        
        for _, row in filtered_df.iterrows():
            key = f"{row['user_id']}_{row['gmap_id']}"
            if key in original_labels:
                original_label = original_labels[key]
                matched_count += 1
        if original_label == 'Yes':
            kept_yes += 1
        elif original_label == 'No':
            kept_no += 1
        
        logging.info(f"Matched {matched_count}/{len(filtered_df)} records using user_id + gmap_id")
        
    # Final fallback to text matching with fuzzy matching
    else:
        logging.warning("Using text-based matching (may be unreliable due to text processing)")
        original_text_labels = {}
        for _, row in original_df.iterrows():
            # Use first 50 chars of text as key to handle minor modifications
            text_key = str(row['text']).strip()[:50]
            original_text_labels[text_key] = row['Label_normalized']
        
        kept_yes = 0
        kept_no = 0
        matched_count = 0
        
        for _, row in filtered_df.iterrows():
            text_key = str(row['text']).strip()[:50]
            if text_key in original_text_labels:
                original_label = original_text_labels[text_key]
                matched_count += 1
                if original_label == 'Yes':
                    kept_yes += 1
                elif original_label == 'No':
                    kept_no += 1
            else:
                # If no match found, assume it was originally "No" (conservative)
                kept_no += 1
        
        logging.info(f"Matched {matched_count}/{len(filtered_df)} records using text prefix")
        if matched_count < len(filtered_df) * 0.8:  # If less than 80% matched
            logging.warning(f"Low match rate ({matched_count/len(filtered_df):.1%}) - evaluation metrics may be unreliable")
    
    # Calculate metrics
    # True Positives: Correctly kept "Yes" items
    true_positives = kept_yes
    
    # False Positives: Incorrectly kept "No" items  
    false_positives = kept_no
    
    # False Negatives: "Yes" items that were filtered out
    false_negatives = original_yes - kept_yes
    
    # True Negatives: "No" items that were correctly filtered out
    true_negatives = original_no - kept_no
    
    # Calculate standard metrics with proper error handling
    accuracy = (true_positives + true_negatives) / original_total if original_total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate retention rates with proper error handling
    overall_retention = filtered_total / original_total if original_total > 0 else 0
    yes_retention = kept_yes / original_yes if original_yes > 0 else 0
    no_retention = kept_no / original_no if original_no > 0 else 0
    
    metrics = {
        'original_stats': {
            'total_reviews': original_total,
            'yes_labels': original_yes,
            'no_labels': original_no,
            'yes_percentage': (original_yes / original_total * 100)
        },
        'filtered_stats': {
            'total_kept': filtered_total,
            'kept_yes': kept_yes,
            'kept_no': kept_no,
            'overall_retention_rate': overall_retention * 100
        },
        'confusion_matrix': {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        },
        'performance_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        },
        'retention_metrics': {
            'yes_retention_rate': yes_retention * 100,
            'no_retention_rate': no_retention * 100,
            'selectivity': (1 - overall_retention) * 100  # How much we filtered out
        }
    }
    
    return metrics

def generate_evaluation_report(metrics: Dict[str, Any], execution_id: str) -> str:
    """
    Generate a human-readable evaluation report.
    
    Args:
        metrics: Evaluation metrics dictionary
        execution_id: Pipeline execution ID
        
    Returns:
        Formatted report string
    """
    report = f"""
=== PIPELINE EVALUATION REPORT ===
Execution ID: {execution_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--- ORIGINAL DATASET STATISTICS ---
Total Reviews: {metrics.get('original_stats', {}).get('total_reviews', 'N/A'):,}
"Yes" Labels: {metrics.get('original_stats', {}).get('yes_labels', 'N/A'):,} ({metrics.get('original_stats', {}).get('yes_percentage', 0):.1f}%)
"No" Labels: {metrics.get('original_stats', {}).get('no_labels', 'N/A'):,}

--- FILTERING RESULTS ---
Reviews Kept: {metrics.get('filtered_stats', {}).get('total_kept', 'N/A'):,}
Overall Retention Rate: {metrics.get('filtered_stats', {}).get('overall_retention_rate', 0):.1f}%
"""

    if 'confusion_matrix' in metrics:
        cm = metrics['confusion_matrix']
        pm = metrics['performance_metrics']
        rm = metrics['retention_metrics']
        
        report += f"""
--- DETAILED PERFORMANCE ANALYSIS ---
Confusion Matrix:
  True Positives (Correctly kept "Yes"):  {cm['true_positives']:,}
  False Positives (Incorrectly kept "No"): {cm['false_positives']:,}
  True Negatives (Correctly filtered "No"): {cm['true_negatives']:,}
  False Negatives (Incorrectly filtered "Yes"): {cm['false_negatives']:,}

Performance Metrics:
  Accuracy:  {pm['accuracy']:.3f} ({pm['accuracy']*100:.1f}%)
  Precision: {pm['precision']:.3f} ({pm['precision']*100:.1f}%)
  Recall:    {pm['recall']:.3f} ({pm['recall']*100:.1f}%)
  F1-Score:  {pm['f1_score']:.3f}

Retention Analysis:
  "Yes" Labels Retained: {rm['yes_retention_rate']:.1f}%
  "No" Labels Retained:  {rm['no_retention_rate']:.1f}%
  Selectivity (Filtered Out): {rm['selectivity']:.1f}%

--- INTERPRETATION ---
"""
        
        # Add interpretation based on metrics
        if pm['accuracy'] >= 0.9:
            report += "🟢 NOICE: Pipeline shows very high accuracy (≥90%)\n"
        elif pm['accuracy'] >= 0.8:
            report += "🟡 GOOD: Pipeline shows good accuracy (≥80%)\n"
        elif pm['accuracy'] >= 0.7:
            report += "🟠 FAIR: Pipeline shows fair accuracy (≥70%)\n"
        else:
            report += "🔴 POOR: Pipeline accuracy below 70% - needs improvement\n"
        
        if pm['precision'] >= 0.9:
            report += "🟢 HIGH PRECISION: Very few false positives\n"
        elif pm['precision'] >= 0.8:
            report += "🟡 GOOD PRECISION: Low false positive rate\n"
        else:
            report += "🔴 LOW PRECISION: High false positive rate - filters too leniently\n"
        
        if pm['recall'] >= 0.9:
            report += "🟢 HIGH RECALL: Captures most positive cases\n"
        elif pm['recall'] >= 0.8:
            report += "🟡 GOOD RECALL: Captures most positive cases\n"
        else:
            report += "🔴 LOW RECALL: Missing many positive cases - filters too strictly\n"
    
    report += "\n=== END REPORT ==="
    return report

def run(config: Dict[str, Any]) -> bool:
    """
    Execute the evaluation stage.
    
    Args:
        config: Configuration dictionary containing:
            - input_file: Path to original input CSV file
            - stage3_output: Path to stage 3 output (optional, will auto-detect)
            - execution_output_dir: Directory for this execution's outputs
            - execution_id: Unique execution identifier
            
    Returns:
        bool: True if successful, False otherwise
    """
    # Get configuration
    original_input = config['input_file']
    execution_output_dir = config['execution_output_dir']
    execution_id = config.get('execution_id', datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    logging.info(f"Stage 5: Starting evaluation")
    logging.info(f"Original input file: {original_input}")
    logging.info(f"Execution output directory: {execution_output_dir}")
    
    # Find the previous stage output file
    if execution_output_dir:
        # Look for previous stage output
        previous_stage = config.get('previous_stage', '04_relevance_check')
        stage_number = previous_stage.split('_')[0] if previous_stage else '04'
        previous_output_file = f"stage{stage_number[-1]}_output.csv"
        
        stage4_file = str(Path(execution_output_dir) / previous_output_file)
        logging.info(f"Using previous stage ({previous_stage}) output: {stage4_file}")
    else:
        # Auto-detect latest stage 4 output
        stage4_pattern = config.get('stage4_output', 'data/stage4_output_*.csv')
        stage4_file = find_latest_file(stage4_pattern)
        logging.info(f"Using Stage 4 output: {stage4_file}")
    
    # Load original dataset
    logging.info(f"Loading original dataset from {original_input}")
    original_df = load_original_data(original_input)
    
    # Load relevance processed dataset
    logging.info(f"Loading relevance processed dataset from {stage4_file}")
    filtered_df = pd.read_csv(stage4_file, sep=';', encoding='utf-8')
    
    # Calculate evaluation metrics
    logging.info("Calculating evaluation metrics...")
    metrics = calculate_metrics(original_df, filtered_df)
    
    # Generate evaluation report
    report = generate_evaluation_report(metrics, execution_id)
    
    # Save results to execution directory
    output_dir = Path(execution_output_dir)
    
    # Save metrics as JSON
    metrics_file = output_dir / "evaluation_metrics.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Save human-readable report
    report_file = output_dir / "evaluation_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Log the report to console
    logging.info("Evaluation completed successfully!")
    logging.info(f"Saved metrics to: {metrics_file}")
    logging.info(f"Saved report to: {report_file}")
    
    # Print key metrics to console
    pm = metrics['performance_metrics']
    logging.info(f"Key Results - Accuracy: {pm['accuracy']:.3f}, Precision: {pm['precision']:.3f}, Recall: {pm['recall']:.3f}, F1: {pm['f1_score']:.3f}")
    
    print(report)  # Print report to console for immediate viewing
    
    return True

if __name__ == "__main__":
    # Test the module independently
    test_config = {
        'input_file': 'data/input.csv',
        'stage2_output': 'data/stage2_output_*.csv',
        'execution_output_dir': 'data/output/test/',
        'execution_id': 'test_run'
    }
    success = run(test_config)
    print(f"Stage 3 execution: {'SUCCESS' if success else 'FAILED'}")
