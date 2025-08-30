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
    # Load data with UTF-8 encoding and semicolon separator
    df = pd.read_csv(input_file, sep=';', encoding='utf-8')
    logging.info(f"Loaded original data with UTF-8 encoding and semicolon separator")
    
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
    # Get original label counts
    original_yes = len(original_df[original_df['Label'] == 'Yes'])
    original_no = len(original_df[original_df['Label'] == 'No'])
    original_total = len(original_df)
    
    # Get filtered counts (everything kept should theoretically be "Yes")
    filtered_total = len(filtered_df)
    
    # Create a mapping from text to original label for comparison
    original_text_labels = dict(zip(original_df['text'], original_df['Label']))
    
    # Check how many of the filtered items were originally "Yes" vs "No"
    kept_yes = 0
    kept_no = 0
    
    for text in filtered_df['text']:
        original_label = original_text_labels[text]
        if original_label == 'Yes':
            kept_yes += 1
        elif original_label == 'No':
            kept_no += 1
    
    # Calculate metrics
    # True Positives: Correctly kept "Yes" items
    true_positives = kept_yes
    
    # False Positives: Incorrectly kept "No" items  
    false_positives = kept_no
    
    # False Negatives: "Yes" items that were filtered out
    false_negatives = original_yes - kept_yes
    
    # True Negatives: "No" items that were correctly filtered out
    true_negatives = original_no - kept_no
    
    # Calculate standard metrics
    accuracy = (true_positives + true_negatives) / original_total
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Calculate retention rates
    overall_retention = filtered_total / original_total
    yes_retention = kept_yes / original_yes
    no_retention = kept_no / original_no
    
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
    
    # Find the stage 4 output file
    if execution_output_dir:
        # Look for stage4_output.csv in the same execution directory
        stage4_file = str(Path(execution_output_dir) / "stage4_output.csv")
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
