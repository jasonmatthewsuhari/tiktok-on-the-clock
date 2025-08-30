"""
Pipeline Stage 4: Relevance Check
This module uses cross-encoders to check whether review text is relevant to business descriptions.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import json
import pickle
import hashlib
from sentence_transformers import CrossEncoder
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def generate_relevance_hash(df: pd.DataFrame, config: Dict[str, Any]) -> str:
    """
    Generate a hash based on data characteristics and config to identify cached relevance scores.
    
    Args:
        df: Input dataframe
        config: Relevance checking configuration
        
    Returns:
        str: Hash string for relevance model identification
    """
    # Create a signature of the data and config
    data_signature = {
        'num_rows': len(df),
        'text_columns': ['text', 'description'],
        'relevance_config': config.get('relevance_config', {}),
        'sample_texts': df['text'].head(3).tolist() if 'text' in df.columns else [],
        'sample_descriptions': df['description'].head(3).tolist() if 'description' in df.columns else []
    }
    
    # Convert to string and hash
    signature_str = json.dumps(data_signature, sort_keys=True)
    return hashlib.md5(signature_str.encode()).hexdigest()[:12]

def save_relevance_artifacts(relevance_scores: np.ndarray, 
                           embeddings: Dict[str, np.ndarray],
                           relevance_hash: str) -> Dict[str, str]:
    """
    Save relevance checking artifacts to the models/ directory.
    
    Returns:
        Dict containing paths to saved artifacts
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    artifacts = {}
    
    # Save relevance scores
    scores_path = models_dir / f"relevance_scores_{relevance_hash}.pkl"
    with open(scores_path, 'wb') as f:
        pickle.dump(relevance_scores, f)
    artifacts['scores'] = str(scores_path)
    logging.info(f"Saved relevance scores to {scores_path}")
    
    # Save embeddings if provided
    if embeddings:
        embeddings_path = models_dir / f"embeddings_{relevance_hash}.pkl"
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        artifacts['embeddings'] = str(embeddings_path)
        logging.info(f"Saved embeddings to {embeddings_path}")
    
    # Save metadata
    metadata = {
        'relevance_hash': relevance_hash,
        'num_scores': len(relevance_scores),
        'score_stats': {
            'mean': float(np.mean(relevance_scores)),
            'std': float(np.std(relevance_scores)),
            'min': float(np.min(relevance_scores)),
            'max': float(np.max(relevance_scores))
        },
        'artifact_paths': artifacts,
        'created_at': datetime.now().isoformat()
    }
    
    metadata_path = models_dir / f"relevance_metadata_{relevance_hash}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    artifacts['metadata'] = str(metadata_path)
    logging.info(f"Saved relevance metadata to {metadata_path}")
    
    return artifacts

def load_relevance_artifacts(relevance_hash: str) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load relevance checking artifacts from the models/ directory.
    
    Returns:
        Tuple of (relevance_scores, embeddings, metadata) or None if not found
    """
    models_dir = Path("models")
    metadata_path = models_dir / f"relevance_metadata_{relevance_hash}.json"
    
    if not metadata_path.exists():
        return None, None, None
    
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load relevance scores
        scores_path = models_dir / f"relevance_scores_{relevance_hash}.pkl"
        if scores_path.exists():
            with open(scores_path, 'rb') as f:
                relevance_scores = pickle.load(f)
        else:
            return None, None, None
        
        # Load embeddings if they exist
        embeddings = {}
        embeddings_path = models_dir / f"embeddings_{relevance_hash}.pkl"
        if embeddings_path.exists():
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
        
        logging.info(f"Loaded cached relevance artifacts for hash: {relevance_hash}")
        return relevance_scores, embeddings, metadata
        
    except Exception as e:
        logging.error(f"Error loading relevance artifacts: {e}")
        return None, None, None

def check_relevance_cross_encoder(texts: List[str], descriptions: List[str], 
                                config: Dict[str, Any]) -> np.ndarray:
    """
    Use cross-encoder to check relevance between review texts and business descriptions.
    
    Args:
        texts: List of review texts
        descriptions: List of business descriptions
        config: Relevance checking configuration
        
    Returns:
        Array of relevance scores (0-1, higher = more relevant)
    """
    model_name = config.get('cross_encoder_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
    batch_size = config.get('batch_size', 32)
    
    logging.info(f"Loading cross-encoder model: {model_name}")
    cross_encoder = CrossEncoder(model_name)
    
    # Prepare text pairs for cross-encoder
    text_pairs = []
    for text, desc in zip(texts, descriptions):
        # Clean and truncate if necessary
        text = str(text)[:512] if text else ""
        desc = str(desc)[:512] if desc else ""
        text_pairs.append([text, desc])
    
    logging.info(f"Computing relevance scores for {len(text_pairs)} text-description pairs")
    
    # Compute relevance scores in batches
    relevance_scores = []
    for i in range(0, len(text_pairs), batch_size):
        batch = text_pairs[i:i+batch_size]
        batch_scores = cross_encoder.predict(batch)
        relevance_scores.extend(batch_scores)
        
        if i % (batch_size * 10) == 0:
            logging.info(f"Processed {i + len(batch)}/{len(text_pairs)} pairs")
    
    # Convert to numpy array and normalize to 0-1 range
    relevance_scores = np.array(relevance_scores)
    
    # Apply sigmoid to ensure 0-1 range
    relevance_scores = 1 / (1 + np.exp(-relevance_scores))
    
    return relevance_scores

def check_relevance_sentence_similarity(texts: List[str], descriptions: List[str],
                                      config: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Use sentence embeddings and cosine similarity to check relevance.
    
    Args:
        texts: List of review texts
        descriptions: List of business descriptions
        config: Relevance checking configuration
        
    Returns:
        Tuple of (relevance_scores, embeddings_dict)
    """
    model_name = config.get('sentence_model', 'all-MiniLM-L6-v2')
    
    logging.info(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Clean texts
    clean_texts = [str(text)[:512] if text else "" for text in texts]
    clean_descriptions = [str(desc)[:512] if desc else "" for desc in descriptions]
    
    logging.info("Computing embeddings for review texts")
    text_embeddings = model.encode(clean_texts, batch_size=32, show_progress_bar=True)
    
    logging.info("Computing embeddings for business descriptions")
    desc_embeddings = model.encode(clean_descriptions, batch_size=32, show_progress_bar=True)
    
    # Compute cosine similarity between each text and its corresponding description
    relevance_scores = []
    for i in range(len(text_embeddings)):
        similarity = cosine_similarity(
            text_embeddings[i].reshape(1, -1),
            desc_embeddings[i].reshape(1, -1)
        )[0][0]
        relevance_scores.append(similarity)
    
    relevance_scores = np.array(relevance_scores)
    
    # Store embeddings for potential future use
    embeddings = {
        'text_embeddings': text_embeddings,
        'desc_embeddings': desc_embeddings
    }
    
    return relevance_scores, embeddings

def run(config: Dict[str, Any]) -> bool:
    """
    Execute the relevance checking pipeline stage.
    
    Args:
        config: Configuration dictionary containing:
            - input_file: Path to input CSV file from stage 3
            - output_file: Path to output CSV file with relevance scores
            - execution_output_dir: Directory for execution outputs
            - relevance_config: Relevance checking configuration
            
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get configuration parameters
        input_file = config.get('input_file')
        output_file = config.get('output_file', 'data/stage4_output.csv')
        execution_output_dir = config.get('execution_output_dir')
        execution_id = config.get('execution_id')
        
        # Relevance checking configuration
        relevance_config = config.get('relevance_config', {})
        method = relevance_config.get('method', 'cross_encoder')  # 'cross_encoder' or 'sentence_similarity'
        relevance_threshold = relevance_config.get('threshold', 0.5)
        use_cache = relevance_config.get('use_cache', True)
        
        # Generate timestamp for metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine input and output paths
        if execution_output_dir:
            # Look for previous stage output
            previous_stage = config.get('previous_stage', '02_rule_based_filtering')
            stage_number = previous_stage.split('_')[0] if previous_stage else '03'
            previous_output_file = f"stage{stage_number[-1]}_output.csv"
            
            stage_input_file = Path(execution_output_dir) / previous_output_file
            input_file = str(stage_input_file)
            logging.info(f"Stage 4: Using previous stage ({previous_stage}) output: {input_file}")
            output_file = str(Path(execution_output_dir) / "stage4_output.csv")
            model_output_dir = Path(execution_output_dir)
        else:
            # Use timestamped output filename
            output_path = Path(output_file)
            output_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
            output_file = str(output_path.parent / output_name)
            model_output_dir = output_path.parent
        
        logging.info(f"Stage 4: Loading data from {input_file}")
        logging.info(f"Stage 4: Relevance output will be saved to {output_file}")
        
        # Load data from stage 3
        df = pd.read_csv(input_file, sep=';', encoding='utf-8')
        initial_count = len(df)
        logging.info(f"Loaded {initial_count} reviews from stage 3")
        
        # Check if we have the required columns
        required_columns = ['text', 'description']
        available_columns = list(df.columns)
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            logging.error(f"Available columns: {available_columns}")
            return False
        
        # Clean and prepare data
        df['text'] = df['text'].fillna('').astype(str)
        df['description'] = df['description'].fillna('').astype(str)
        
        # Filter out rows with empty text or description
        valid_mask = (df['text'].str.len() > 0) & (df['description'].str.len() > 0)
        df_valid = df[valid_mask].copy()
        df_invalid = df[~valid_mask].copy()
        
        logging.info(f"Valid text-description pairs: {len(df_valid)}")
        logging.info(f"Invalid pairs (empty text/description): {len(df_invalid)}")
        
        if len(df_valid) == 0:
            logging.warning("No valid text-description pairs found, adding dummy relevance scores")
            df['relevance_score'] = 0.0
            df['is_relevant'] = 0
            df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
            return True
        
        # Generate relevance hash for caching
        relevance_hash = generate_relevance_hash(df_valid, config)
        logging.info(f"Generated relevance hash: {relevance_hash}")
        
        # Check for cached results
        if use_cache:
            cached_scores, cached_embeddings, cached_metadata = load_relevance_artifacts(relevance_hash)
            if cached_scores is not None and len(cached_scores) == len(df_valid):
                logging.info("Found cached relevance scores! Using existing results...")
                relevance_scores = cached_scores
                embeddings = cached_embeddings or {}
                skip_computation = True
            else:
                skip_computation = False
        else:
            skip_computation = False
        
        if not skip_computation:
            logging.info(f"Computing relevance scores using method: {method}")
            
            if method == 'cross_encoder':
                relevance_scores = check_relevance_cross_encoder(
                    df_valid['text'].tolist(),
                    df_valid['description'].tolist(),
                    relevance_config
                )
                embeddings = {}
                
            elif method == 'sentence_similarity':
                relevance_scores, embeddings = check_relevance_sentence_similarity(
                    df_valid['text'].tolist(),
                    df_valid['description'].tolist(),
                    relevance_config
                )
                
            else:
                logging.error(f"Unknown relevance method: {method}")
                return False
            
            # Save results to cache
            if use_cache:
                logging.info("Saving relevance results to cache...")
                save_relevance_artifacts(relevance_scores, embeddings, relevance_hash)
        
        # Add relevance scores to dataframe
        df_valid['relevance_score'] = relevance_scores
        
        # Use dynamic threshold - remove bottom 50% least relevant
        if len(relevance_scores) > 0:
            dynamic_threshold = np.percentile(relevance_scores, 50)  # 50th percentile (median)
            logging.info(f"Dynamic relevance threshold (50th percentile): {dynamic_threshold:.4f}")
            logging.info(f"Original static threshold was: {relevance_threshold:.4f}")
        else:
            dynamic_threshold = relevance_threshold
            
        df_valid['is_relevant'] = (relevance_scores >= dynamic_threshold).astype(int)
        
        # For invalid pairs, set low relevance scores
        if len(df_invalid) > 0:
            df_invalid['relevance_score'] = 0.0
            df_invalid['is_relevant'] = 0
        
        # Combine results
        df_final = pd.concat([df_valid, df_invalid], ignore_index=True)
        
        # Sort back to original order if needed
        if 'Unnamed: 0' in df_final.columns:
            df_final = df_final.sort_values('Unnamed: 0').reset_index(drop=True)
        
        # Calculate statistics
        relevant_count = int(df_final['is_relevant'].sum())
        irrelevant_count = len(df_final) - relevant_count
        relevance_rate = relevant_count / len(df_final) * 100
        
        # Show examples of relevant and irrelevant pairs
        logging.info("\n" + "="*80)
        logging.info("📊 RELEVANCE ANALYSIS EXAMPLES")
        logging.info("="*80)
        
        # Get examples (sort by relevance score for better display)
        df_sorted = df_final.sort_values('relevance_score', ascending=False)
        relevant_examples = df_sorted[df_sorted['is_relevant'] == 1].head(10)
        irrelevant_examples = df_sorted[df_sorted['is_relevant'] == 0].tail(10)  # Get lowest scoring irrelevant
        
        if len(relevant_examples) > 0:
            threshold_used = dynamic_threshold if len(relevance_scores) > 0 else relevance_threshold
            logging.info(f"\n✅ TOP 10 RELEVANT PAIRS (score >= {threshold_used:.4f}):")
            logging.info("-" * 60)
            for i, (_, row) in enumerate(relevant_examples.iterrows(), 1):
                score = row['relevance_score']
                text = str(row['text'])[:100] + "..." if len(str(row['text'])) > 100 else str(row['text'])
                description = str(row['description'])[:80] + "..." if len(str(row['description'])) > 80 else str(row['description'])
                business = row['business_name']
                
                logging.info(f"{i:2d}. Score: {score:.4f} | Business: {business}")
                logging.info(f"    Review: {text}")
                logging.info(f"    Description: {description}")
                logging.info("")
        
        if len(irrelevant_examples) > 0:
            threshold_used = dynamic_threshold if len(relevance_scores) > 0 else relevance_threshold
            logging.info(f"\n❌ TOP 10 IRRELEVANT PAIRS (score < {threshold_used:.4f}):")
            logging.info("-" * 60)
            for i, (_, row) in enumerate(irrelevant_examples.iterrows(), 1):
                score = row['relevance_score']
                text = str(row['text'])[:100] + "..." if len(str(row['text'])) > 100 else str(row['text'])
                description = str(row['description'])[:80] + "..." if len(str(row['description'])) > 80 else str(row['description'])
                business = row['business_name']
                
                logging.info(f"{i:2d}. Score: {score:.4f} | Business: {business}")
                logging.info(f"    Review: {text}")
                logging.info(f"    Description: {description}")
                logging.info("")
        
        logging.info("="*80)
        
        # Save results
        df_final.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        logging.info(f"Stage 4: Saved {len(df_final)} rows with relevance scores to {output_file}")
        
        # Save relevance metrics
        metrics_path = model_output_dir / "relevance_metrics.json"
        relevance_metrics = {
            'execution_id': execution_id,
            'timestamp': timestamp,
            'relevance_hash': relevance_hash,
            'method': method,
            'threshold': relevance_threshold,
            'total_reviews': len(df_final),
            'valid_pairs': len(df_valid),
            'relevant_reviews': relevant_count,
            'irrelevant_reviews': irrelevant_count,
            'relevance_rate': relevance_rate,
            'score_statistics': {
                'mean': float(np.mean(df_final['relevance_score'])),
                'std': float(np.std(df_final['relevance_score'])),
                'min': float(np.min(df_final['relevance_score'])),
                'max': float(np.max(df_final['relevance_score'])),
                'median': float(np.median(df_final['relevance_score']))
            },
            'used_cached_results': skip_computation
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(relevance_metrics, f, indent=2)
        logging.info(f"Saved relevance metrics to {metrics_path}")
        
        # Log summary statistics
        logging.info(f"Stage 4 Summary:")
        logging.info(f"  - Total reviews processed: {len(df_final)}")
        logging.info(f"  - Method used: {method}")
        logging.info(f"  - Relevance threshold: {relevance_threshold}")
        logging.info(f"  - Relevant reviews: {relevant_count} ({relevance_rate:.1f}%)")
        logging.info(f"  - Mean relevance score: {np.mean(df_final['relevance_score']):.3f}")
        logging.info(f"  - Used cached results: {skip_computation}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in stage 4 (relevance check): {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Test the module independently
    test_config = {
        'input_file': 'data/stage3_output.csv',
        'output_file': 'data/stage4_output.csv',
        'relevance_config': {
            'method': 'cross_encoder',
            'threshold': 0.5,
            'batch_size': 16
        }
    }
    success = run(test_config)
    print(f"Stage 4 execution: {'SUCCESS' if success else 'FAILED'}")
