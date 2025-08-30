"""
Pipeline Stage 3: Advanced Multi-Stage Review Classification
Implements a cascade architecture: weak supervision → fast filter → boosted model → text encoder → calibration
"""

import torch
import pandas as pd
import numpy as np
import logging
import pickle
import json
import time
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
import hashlib
import re

# ML Libraries
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
import lightgbm as lgb

# Text processing
from sentence_transformers import SentenceTransformer

def log_progress(message, force_flush=True):
    """Log message with immediate flush to both console and file"""
    logging.info(message)
    if force_flush:
        for handler in logging.getLogger().handlers:
            handler.flush()
        sys.stdout.flush()

def format_time(seconds):
    """Format seconds into readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"

def create_progress_bar(current, total, width=50):
    """Create a visual progress bar"""
    progress = current / total
    filled = int(width * progress)
    bar = '█' * filled + '░' * (width - filled)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}%"

# =============================================================================
# 1. WEAK SUPERVISION - LABEL FUNCTIONS
# =============================================================================

class LabelFunctionEngine:
    """Implements label functions for weak supervision"""
    
    def __init__(self):
        self.label_functions = []
        self._setup_label_functions()
    
    def _setup_label_functions(self):
        """Define label functions for automatic labeling"""
        
        # LF1: Ad/Promo detection
        def lf_promo_signals(row):
            text = str(row.get('text', '')).lower()
            promo_words = ['discount', 'promo', 'code', 'dm me', 'whatsapp', 'telegram', 
                          'collab', 'sponsored', 'partnership', 'follow me', 'check out my']
            url_count = len(re.findall(r'http[s]?://|www\.|\.[com|net|org]', text))
            phone_count = len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text))
            
            score = url_count * 0.4 + phone_count * 0.3
            for word in promo_words:
                if word in text:
                    score += 0.2
            
            if score >= 0.5:
                return 1, 0.8  # junk, high confidence
            return None, 0.0
        
        # LF2: Irrelevant content detection
        def lf_irrelevant_content(row):
            text = str(row.get('text', ''))
            
            # Too short
            if len(text.strip()) < 8:
                return 1, 0.7  # junk, medium confidence
            
            # Emoji only
            emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
            clean_text = re.sub(emoji_pattern, '', text).strip()
            if len(clean_text) < 3:
                return 1, 0.6  # junk, medium confidence
            
            return None, 0.0
        
        # LF3: Rating mismatch detection
        def lf_rating_mismatch(row):
            rating = row.get('rating', 3)
            text = str(row.get('text', '')).lower()
            
            positive_words = ['great', 'love', 'awesome', 'amazing', 'excellent', 'perfect']
            negative_words = ['worst', 'terrible', 'awful', 'hate', 'horrible', 'disgusting']
            
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            
            # 1-2 star with many positive words
            if rating <= 2 and pos_count >= 2 and neg_count == 0:
                return 1, 0.6  # junk, medium confidence
            
            # 4-5 star with many negative words
            if rating >= 4 and neg_count >= 2 and pos_count == 0:
                return 1, 0.6  # junk, medium confidence
            
            return None, 0.0
        
        # LF4: Owner response signals
        def lf_owner_response(row):
            resp_text = str(row.get('resp', '')).lower()  # Use 'resp' column
            spam_signals = ['spam', 'promotion', 'fake', 'bot', 'automated']
            
            for signal in spam_signals:
                if signal in resp_text:
                    return 1, 0.7  # junk, high confidence
            
            return None, 0.0
        
        # LF5: User behavior patterns
        def lf_user_patterns(row):
            # This would require aggregated user stats
            # For now, basic heuristics
            text = str(row.get('text', ''))
            
            # Very generic reviews
            generic_phrases = ['good place', 'nice place', 'ok place', 'decent food', 'average service']
            if any(phrase in text.lower() for phrase in generic_phrases) and len(text) < 30:
                return 1, 0.4  # junk, low confidence
            
            return None, 0.0
        
        self.label_functions = [
            lf_promo_signals,
            lf_irrelevant_content, 
            lf_rating_mismatch,
            lf_owner_response,
            lf_user_patterns
        ]
    
    def apply_label_functions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all label functions and aggregate results"""
        log_progress("Applying weak supervision label functions...")
        
        lf_results = []
        lf_confidences = []
        
        for i, row in df.iterrows():
            votes = []
            confs = []
            
            for lf in self.label_functions:
                label, confidence = lf(row)
                if label is not None:
                    votes.append(label)
                    confs.append(confidence)
            
            if votes:
                # Weighted majority vote
                if confs:
                    weighted_vote = np.average(votes, weights=confs)
                    avg_confidence = np.mean(confs)
                else:
                    weighted_vote = np.mean(votes)
                    avg_confidence = 0.5
                
                final_label = 1 if weighted_vote > 0.5 else 0
                lf_results.append(final_label)
                lf_confidences.append(avg_confidence)
            else:
                # No label functions fired - neutral
                lf_results.append(0)  # assume relevant if unclear
                lf_confidences.append(0.1)  # low confidence
        
        df['weak_label'] = lf_results
        df['weak_confidence'] = lf_confidences
        
        log_progress(f"Weak supervision complete. Labels: {np.bincount(lf_results)}")
        return df

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

class AdvancedFeatureExtractor:
    """Extracts comprehensive features for the cascade model"""
    
    def __init__(self):
        self.text_vectorizer_word = None
        self.text_vectorizer_char = None
        self.svd_transformer = None
        self.target_encoders = {}
        self.fitted = False
    
    def extract_text_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Extract hashed text features"""
        texts = df['text'].fillna('').astype(str)
        
        if fit or self.text_vectorizer_word is None:
            # Word n-grams
            self.text_vectorizer_word = HashingVectorizer(
                ngram_range=(1, 2),
                analyzer='word',
                n_features=2**16,  # Smaller for demo, scale to 2**18 in prod
                alternate_sign=False,
                lowercase=True,
                stop_words='english'
            )
            
            # Char n-grams  
            self.text_vectorizer_char = HashingVectorizer(
                ngram_range=(3, 5),
                analyzer='char',
                n_features=2**16,
                alternate_sign=False,
                lowercase=True
            )
        
        word_features = self.text_vectorizer_word.fit_transform(texts) if fit else self.text_vectorizer_word.transform(texts)
        char_features = self.text_vectorizer_char.fit_transform(texts) if fit else self.text_vectorizer_char.transform(texts)
        
        # Combine sparse matrices
        from scipy.sparse import hstack
        combined_features = hstack([word_features, char_features])
        
        # Optional: Add SVD for dense representation
        if fit or self.svd_transformer is None:
            if fit:
                self.svd_transformer = TruncatedSVD(n_components=256, random_state=42)
                svd_features = self.svd_transformer.fit_transform(combined_features)
            else:
                svd_features = self.svd_transformer.transform(combined_features)
        else:
            svd_features = self.svd_transformer.transform(combined_features)
        
        return combined_features, svd_features
    
    def extract_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract regex-based count features"""
        features = df.copy()
        
        # Text analysis (text column should always exist)
        text_col = features['text'].fillna('') if 'text' in features.columns else pd.Series([''] * len(features))
        features['text_len'] = text_col.str.len()
        features['word_count'] = text_col.str.split().str.len()
        features['exclamation_count'] = text_col.str.count('!')
        features['caps_ratio'] = text_col.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
        )
        
        # URL and contact detection
        features['url_count'] = text_col.str.count(r'http[s]?://|www\.|\.com|\.net|\.org')
        features['phone_count'] = text_col.str.count(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        features['email_count'] = text_col.str.count(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Emoji counting (simplified)
        emoji_pattern = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]'
        features['emoji_count'] = text_col.str.count(emoji_pattern)
        
        # Response features (your data has 'resp' column)
        if 'resp' in features.columns:
            features['response_len'] = features['resp'].fillna('').str.len()
        elif 'resp_text' in features.columns:
            features['response_len'] = features['resp_text'].fillna('').str.len()
        else:
            log_progress("  Warning: No response column found, using default values")
            features['response_len'] = 0
        features['has_response'] = (features['response_len'] > 0).astype(int)
        
        # Pics feature (your data has 'pics' column)
        if 'pics' in features.columns:
            # Count non-empty pics entries
            pics_col = features['pics'].fillna('')
            features['pics_count'] = pics_col.apply(
                lambda x: 1 if str(x).strip() and str(x).strip() != 'nan' else 0
            )
        elif 'pics_urls' in features.columns:
            # Fallback to pics_urls if exists
            pics_col = features['pics_urls'].fillna('')
            features['pics_count'] = pics_col.apply(
                lambda x: len(str(x).split(',')) if str(x) and str(x) != '[]' and str(x) != '' else 0
            )
        else:
            log_progress("  Warning: No pics column found, using default values")
            features['pics_count'] = 0
        
        return features
    
    def extract_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process categorical features with hashing for high-cardinality"""
        features = df.copy()
        
        # Price ordinal encoding (handle missing price column)
        if 'price' in features.columns:
            price_map = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
            features['price_ordinal'] = features['price'].fillna('').map(price_map).fillna(0)
        else:
            log_progress("  Warning: 'price' column not found, using default values")
            features['price_ordinal'] = 0
        
        # State - keep as is for now (could hash if too many)
        if 'state' in features.columns:
            features['state'] = features['state'].fillna('unknown')
        else:
            log_progress("  Warning: 'state' column not found, using default values")
            features['state'] = 'unknown'
        
        # Category processing - take first category
        def extract_main_category(cat_json):
            try:
                if pd.isna(cat_json) or cat_json == '':
                    return 'unknown'
                # Simple extraction - take first word before comma/semicolon
                main_cat = str(cat_json).split(',')[0].split(';')[0].strip()
                return main_cat.lower()[:20]  # Truncate long categories
            except:
                return 'unknown'
        
        if 'category' in features.columns:
            features['main_category'] = features['category'].apply(extract_main_category)
        else:
            log_progress("  Warning: 'category' column not found, using default values")
            features['main_category'] = 'unknown'
        
        return features
    
    def fit_transform(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fit all feature extractors and transform data"""
        log_progress("Extracting comprehensive features...")
        
        # Count features
        count_features = self.extract_count_features(df)
        
        # Categorical features
        cat_features = self.extract_categorical_features(count_features)
        
        # Text features
        sparse_text, dense_text = self.extract_text_features(cat_features, fit=True)
        
        # Select numeric features (handle missing columns gracefully)
        # Updated for your actual schema
        base_numeric_cols = ['rating', 'avg_rating', 'num_of_reviews', 'latitude', 'longitude',
                            'text_len', 'word_count', 'exclamation_count', 'caps_ratio', 
                            'url_count', 'phone_count', 'email_count', 'emoji_count', 
                            'response_len', 'has_response', 'pics_count', 'price_ordinal']
        
        # Only use columns that exist, fill missing ones with defaults
        numeric_cols = []
        for col in base_numeric_cols:
            if col in cat_features.columns:
                numeric_cols.append(col)
            else:
                # Add missing columns with default values
                if col in ['rating', 'avg_rating']:
                    cat_features[col] = 3.0  # Default neutral rating
                elif col == 'num_of_reviews':
                    cat_features[col] = 1  # Default to 1 review
                else:
                    cat_features[col] = 0  # Default to 0 for counts
                numeric_cols.append(col)
                log_progress(f"  Warning: Missing column '{col}', using default values")
        
        numeric_features = cat_features[numeric_cols].fillna(0).values
        
        # Categorical features for high-level model
        categorical_cols = ['state', 'main_category']
        categorical_features = cat_features[categorical_cols]
        
        self.fitted = True
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'sparse_text': sparse_text,
            'dense_text': dense_text,
            'count_features': cat_features[numeric_cols],  # Use cat_features which has all columns
            'processed_df': cat_features
        }
    
    def transform(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Transform new data using fitted extractors"""
        if not self.fitted:
            raise ValueError("Must fit feature extractor first")
        
        # Count features
        count_features = self.extract_count_features(df)
        
        # Categorical features
        cat_features = self.extract_categorical_features(count_features)
        
        # Text features
        sparse_text, dense_text = self.extract_text_features(cat_features, fit=False)
        
        # Select numeric features (use same logic as fit_transform)
        # Updated for your actual schema
        base_numeric_cols = ['rating', 'avg_rating', 'num_of_reviews', 'latitude', 'longitude',
                            'text_len', 'word_count', 'exclamation_count', 'caps_ratio', 
                            'url_count', 'phone_count', 'email_count', 'emoji_count', 
                            'response_len', 'has_response', 'pics_count', 'price_ordinal']
        
        # Only use columns that exist, fill missing ones with defaults
        numeric_cols = []
        for col in base_numeric_cols:
            if col in cat_features.columns:
                numeric_cols.append(col)
            else:
                # Add missing columns with default values
                if col in ['rating', 'avg_rating']:
                    cat_features[col] = 3.0  # Default neutral rating
                elif col == 'num_of_reviews':
                    cat_features[col] = 1  # Default to 1 review
                else:
                    cat_features[col] = 0  # Default to 0 for counts
                numeric_cols.append(col)
        
        numeric_features = cat_features[numeric_cols].fillna(0).values
        
        # Categorical features for high-level model
        categorical_cols = ['state', 'main_category']
        categorical_features = cat_features[categorical_cols]
        
        return {
            'numeric': numeric_features,
            'categorical': categorical_features,
            'sparse_text': sparse_text,
            'dense_text': dense_text,
            'count_features': cat_features[numeric_cols],  # Use cat_features which has all columns
            'processed_df': cat_features
        }

# =============================================================================
# 3. STAGE 1: FAST FILTER
# =============================================================================

class Stage1FastFilter:
    """Fast linear model for obvious relevant/junk classification"""
    
    def __init__(self):
        self.text_model = None
        self.rules_model = None
        self.calibrator_text = None
        self.calibrator_rules = None
        self.thresholds = {'tau_hi': 0.8, 'tau_lo': 0.2}
    
    def fit(self, features: Dict[str, Any], y: np.ndarray):
        """Fit Stage 1 models"""
        log_progress("Training Stage 1: Fast Filter...")
        
        # Model 1a: Text-only linear model
        self.text_model = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=1e-4,
            l1_ratio=0.15,
            early_stopping=True,
            n_iter_no_change=5,
            class_weight='balanced',
            random_state=42
        )
        
        log_progress("  Training text-based SGD model...")
        self.text_model.fit(features['sparse_text'], y)
        
        # Model 1b: Rules/counts linear model
        self.rules_model = SGDClassifier(
            loss='log_loss',
            penalty='elasticnet',
            alpha=1e-4,
            l1_ratio=0.15,
            early_stopping=True,
            n_iter_no_change=5,
            class_weight='balanced',
            random_state=42
        )
        
        log_progress("  Training rules-based SGD model...")
        self.rules_model.fit(features['numeric'], y)
        
        # Calibrate both models
        log_progress("  Calibrating Stage 1 models...")
        self.calibrator_text = CalibratedClassifierCV(self.text_model, method='isotonic', cv=3)
        self.calibrator_text.fit(features['sparse_text'], y)
        
        self.calibrator_rules = CalibratedClassifierCV(self.rules_model, method='isotonic', cv=3)
        self.calibrator_rules.fit(features['numeric'], y)
        
        log_progress("Stage 1 training complete!")
    
    def predict_proba(self, features: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Get probabilities from both Stage 1 models"""
        text_proba = self.calibrator_text.predict_proba(features['sparse_text'])[:, 1]
        rules_proba = self.calibrator_rules.predict_proba(features['numeric'])[:, 1]
        return text_proba, rules_proba
    
    def gate_predictions(self, text_proba: np.ndarray, rules_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply gating logic to determine which samples need Stage 2"""
        combined_proba = (text_proba + rules_proba) / 2
        
        # Gating decisions
        confident_junk = combined_proba >= self.thresholds['tau_hi']
        confident_relevant = combined_proba <= self.thresholds['tau_lo']
        needs_stage2 = ~(confident_junk | confident_relevant)
        
        decisions = np.where(confident_junk, 1, np.where(confident_relevant, 0, -1))  # -1 = uncertain
        
        return decisions, needs_stage2

# =============================================================================
# 4. STAGE 2: BOOSTED MODEL
# =============================================================================

class Stage2BoostedModel:
    """High-accuracy boosted model for ambiguous cases"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.calibrator = None
    
    def fit(self, features: Dict[str, Any], y: np.ndarray, cat_features: Optional[List[str]] = None):
        """Fit Stage 2 LightGBM model"""
        log_progress("Training Stage 2: Boosted Model...")
        
        # Combine features for LightGBM
        # Numeric + dense text features
        X_combined = np.concatenate([
            features['numeric'],
            features['dense_text']
        ], axis=1)
        
        # Create feature names (updated for your schema)
        numeric_names = ['rating', 'avg_rating', 'num_of_reviews', 'latitude', 'longitude',
                        'text_len', 'word_count', 'exclamation_count', 'caps_ratio', 
                        'url_count', 'phone_count', 'email_count', 'emoji_count', 
                        'response_len', 'has_response', 'pics_count', 'price_ordinal']
        text_names = [f'text_svd_{i}' for i in range(features['dense_text'].shape[1])]
        self.feature_names = numeric_names + text_names
        
        # LightGBM parameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.08,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 50,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'is_unbalance': True
        }
        
        # Train with early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=self.feature_names)
        
        log_progress("  Training LightGBM model...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
        )
        
        # Calibrate
        log_progress("  Calibrating Stage 2 model...")
        val_pred = self.model.predict(X_val)
        self.calibrator = CalibratedClassifierCV(self.model, method='isotonic', cv=3)
        # Note: LightGBM calibration requires wrapper - simplified for demo
        from sklearn.isotonic import IsotonicRegression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(val_pred, y_val)
        
        log_progress("Stage 2 training complete!")
    
    def predict_proba(self, features: Dict[str, Any]) -> np.ndarray:
        """Predict probabilities for Stage 2"""
        X_combined = np.concatenate([
            features['numeric'],
            features['dense_text']
        ], axis=1)
        
        raw_pred = self.model.predict(X_combined)
        calibrated_pred = self.calibrator.transform(raw_pred)
        return calibrated_pred

# =============================================================================
# 5. STAGE 3: TEXT ENCODER (OPTIONAL)
# =============================================================================

class Stage3TextEncoder:
    """Sentence embedding model for final borderline cases"""
    
    def __init__(self):
        self.sentence_model = None
        self.classifier = None
        self.delta = 0.1  # borderline threshold
    
    def fit(self, df: pd.DataFrame, y: np.ndarray, stage2_proba: np.ndarray):
        """Fit Stage 3 model on borderline cases only"""
        log_progress("Training Stage 3: Text Encoder...")
        
        # Identify borderline cases
        borderline_mask = np.abs(stage2_proba - 0.5) <= self.delta
        
        if np.sum(borderline_mask) < 50:
            log_progress("  Not enough borderline cases for Stage 3 training")
            return
        
        borderline_texts = df.loc[borderline_mask, 'text'].values
        borderline_y = y[borderline_mask]
        
        log_progress(f"  Training on {len(borderline_texts)} borderline cases...")
        
        # Load sentence transformer
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Get embeddings
        embeddings = self.sentence_model.encode(borderline_texts)
        
        # Train simple classifier
        self.classifier = LogisticRegression(max_iter=400, class_weight='balanced')
        self.classifier.fit(embeddings, borderline_y)
        
        log_progress("Stage 3 training complete!")
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict probabilities for borderline texts"""
        if self.sentence_model is None or self.classifier is None:
            return np.array([0.5] * len(texts))  # Return neutral if not trained
        
        embeddings = self.sentence_model.encode(texts)
        return self.classifier.predict_proba(embeddings)[:, 1]

# =============================================================================
# 6. MAIN CASCADE MODEL
# =============================================================================

class CascadeReviewClassifier:
    """Main cascade model orchestrating all stages"""
    
    def __init__(self):
        self.label_engine = LabelFunctionEngine()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.stage1 = Stage1FastFilter()
        self.stage2 = Stage2BoostedModel()
        self.stage3 = Stage3TextEncoder()
        self.fitted = False
    
    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Fit the complete cascade model"""
        start_time = time.time()
        log_progress("="*80)
        log_progress("TRAINING CASCADE REVIEW CLASSIFIER")
        log_progress("="*80)
        
        # Step 1: Weak supervision if no labels provided
        if y is None:
            log_progress("No ground truth labels provided - using weak supervision")
            df = self.label_engine.apply_label_functions(df)
            y = df['weak_label'].values
        else:
            log_progress("Using provided ground truth labels")
        
        log_progress(f"Training on {len(df)} samples with {np.bincount(y)} distribution")
        
        # Step 2: Feature extraction
        features = self.feature_extractor.fit_transform(df, y)
        
        # Step 3: Train Stage 1
        self.stage1.fit(features, y)
        
        # Step 4: Train Stage 2 on full data (will be gated during inference)
        self.stage2.fit(features, y)
        
        # Step 5: Get Stage 2 predictions for Stage 3 training
        stage2_proba = self.stage2.predict_proba(features)
        
        # Step 6: Train Stage 3 on borderline cases
        self.stage3.fit(df, y, stage2_proba)
        
        self.fitted = True
        total_time = time.time() - start_time
        
        log_progress("="*80)
        log_progress(f"CASCADE TRAINING COMPLETE! Total time: {format_time(total_time)}")
        log_progress("="*80)
    
    def predict_proba(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Run inference through the cascade"""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        log_progress(f"Running cascade inference on {len(df)} samples...")
        
        # Extract features
        features = self.feature_extractor.transform(df)
        
        # Stage 1: Fast filter
        text_proba, rules_proba = self.stage1.predict_proba(features)
        decisions, needs_stage2 = self.stage1.gate_predictions(text_proba, rules_proba)
        
        stage1_confident = np.sum(decisions != -1)
        stage2_needed = np.sum(needs_stage2)
        
        log_progress(f"  Stage 1: {stage1_confident}/{len(df)} confident, {stage2_needed} need Stage 2")
        
        # Initialize final probabilities
        final_proba = (text_proba + rules_proba) / 2
        stage_used = np.where(decisions != -1, 1, 2)
        
        # Stage 2: For uncertain cases
        if stage2_needed > 0:
            stage2_indices = np.where(needs_stage2)[0]
            stage2_features = {
                'numeric': features['numeric'][stage2_indices],
                'dense_text': features['dense_text'][stage2_indices]
            }
            stage2_proba = self.stage2.predict_proba(stage2_features)
            final_proba[stage2_indices] = stage2_proba
            
            # Stage 3: For borderline cases from Stage 2
            borderline_mask = np.abs(stage2_proba - 0.5) <= self.stage3.delta
            if np.sum(borderline_mask) > 0:
                borderline_indices = stage2_indices[borderline_mask]
                borderline_texts = df.iloc[borderline_indices]['text'].tolist()
                
                if len(borderline_texts) > 0:
                    stage3_proba = self.stage3.predict_proba(borderline_texts)
                    final_proba[borderline_indices] = stage3_proba
                    stage_used[borderline_indices] = 3
                    
                    log_progress(f"  Stage 3: {len(borderline_indices)} borderline cases processed")
        
        return {
            'probabilities': final_proba,
            'stage_used': stage_used,
            'stage1_decisions': decisions,
            'needs_stage2': needs_stage2
        }
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions"""
        result = self.predict_proba(df)
        return (result['probabilities'] > threshold).astype(int)

# =============================================================================
# 7. INTEGRATION WITH EXISTING PIPELINE
# =============================================================================

def run(config: Dict[str, Any]) -> bool:
    """Main entry point for Stage 3 - integrates with existing pipeline"""
    try:
        log_progress("="*80)
        log_progress("STAGE 3: ADVANCED CASCADE MODEL")
        log_progress("="*80)
        
        # Get configuration
        execution_output_dir = config.get('execution_output_dir')
        model_config = config.get('model_config', {})
        
        # Determine input and output paths
        if execution_output_dir:
            previous_stage = config.get('previous_stage', '02_rule_based_filtering')
            stage_number = previous_stage.split('_')[0] if previous_stage else '02'
            previous_output_file = f"stage{stage_number[-1]}_output.csv"
            
            input_file = str(Path(execution_output_dir) / previous_output_file)
            output_file = str(Path(execution_output_dir) / "stage3_output.csv")
            log_progress(f"Input: {input_file}")
            log_progress(f"Output: {output_file}")
        else:
            input_file = config.get('input_file', 'data/stage2_output.csv')
            output_file = config.get('output_file', 'data/stage3_output.csv')
        
        # Load data
        log_progress(f"Loading data from {input_file}")
        df = pd.read_csv(input_file, sep=';', encoding='utf-8')
        log_progress(f"Loaded {len(df)} reviews")
        
        # Check for existing labels (your data has 'Label' column)
        has_labels = False
        y = None
        
        if 'Label' in df.columns:
            # Map your labels: valid->0 (relevant), invalid->1 (junk)
            label_mapping = {
                'valid': 0,
                'invalid': 1,
                'Valid': 0,
                'Invalid': 1,
                'VALID': 0,
                'INVALID': 1
            }
            df['model_label'] = df['Label'].map(label_mapping)
            # Only use rows with valid labels
            valid_mask = df['model_label'].notna()
            if valid_mask.sum() > 0:
                df = df[valid_mask].copy()
                y = df['model_label'].values.astype(int)
                has_labels = True
                log_progress(f"Found ground truth labels: {np.bincount(y)} (0=relevant, 1=junk)")
        elif 'model_label' in df.columns:
            y = df['model_label'].values
            has_labels = True
        
        # Initialize and train cascade model
        cascade = CascadeReviewClassifier()
        cascade.fit(df, y)
        
        # Run inference
        results = cascade.predict_proba(df)
        
        # Add results to dataframe
        df['model_prediction'] = cascade.predict(df)
        df['model_probability'] = results['probabilities']
        df['stage_used'] = results['stage_used']
        
        # Performance summary
        stage_counts = np.bincount(results['stage_used'], minlength=4)
        log_progress("="*60)
        log_progress("PROCESSING SUMMARY:")
        log_progress(f"  Stage 1 (Fast Filter): {stage_counts[1]} samples")
        log_progress(f"  Stage 2 (Boosted Model): {stage_counts[2]} samples") 
        log_progress(f"  Stage 3 (Text Encoder): {stage_counts[3]} samples")
        log_progress(f"  Predicted Junk: {np.sum(df['model_prediction'] == 1)} ({np.mean(df['model_prediction'])*100:.1f}%)")
        log_progress("="*60)
        
        # Save results
        df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        log_progress(f"Results saved to {output_file}")
        
        return True
        
    except Exception as e:
        log_progress(f"Error in Stage 3: {e}")
        import traceback
        log_progress(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Test configuration
    test_config = {
        'execution_output_dir': '.',
        'model_config': {
            'epochs': 3,
            'batch_size': 16,
            'learning_rate': 2e-5
        }
    }
    
    log_progress("Testing Stage 3 cascade model...")
    success = run(test_config)
    log_progress(f"Stage 3 test: {'SUCCESS' if success else 'FAILED'}")
