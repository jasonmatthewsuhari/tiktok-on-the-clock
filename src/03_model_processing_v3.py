"""
Pipeline Stage 3: Rigorous ML-Based Review Classification
Focuses on sophisticated ML analysis of reviews that passed Stage 2's rule-based filtering.
No duplication of fast filtering - that's handled in Stage 2.
"""

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

# ML Libraries
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import lightgbm as lgb

# Text processing and embeddings
from sentence_transformers import SentenceTransformer

# Optional text analysis (with fallbacks)
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    
try:
    from textstat import flesch_reading_ease, automated_readability_index
    TEXTSTAT_AVAILABLE = True
except ImportError:
    TEXTSTAT_AVAILABLE = False

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
# 1. ADVANCED FEATURE ENGINEERING
# =============================================================================

class AdvancedTextFeatures(BaseEstimator, TransformerMixin):
    """Extract sophisticated text features for ML models"""
    
    def __init__(self):
        self.fitted = False
        
    def fit(self, X, y=None):
        self.fitted = True
        return self
    
    def transform(self, X):
        """Extract advanced text features from review text"""
        if isinstance(X, pd.Series):
            texts = X.fillna('').astype(str)
        else:
            texts = pd.Series(X).fillna('').astype(str)
        
        features = pd.DataFrame()
        
        # Basic text metrics
        features['text_length'] = texts.str.len()
        features['word_count'] = texts.str.split().str.len()
        features['sentence_count'] = texts.str.count(r'[.!?]+') + 1
        features['avg_word_length'] = texts.apply(lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0)
        
        # Readability scores (with fallbacks)
        if TEXTSTAT_AVAILABLE:
            features['flesch_score'] = texts.apply(lambda x: flesch_reading_ease(str(x)) if len(str(x)) > 10 else 50)
            features['reading_difficulty'] = texts.apply(lambda x: automated_readability_index(str(x)) if len(str(x)) > 10 else 5)
        else:
            # Simple approximations if textstat not available
            features['flesch_score'] = 50  # Neutral readability
            features['reading_difficulty'] = 5  # Average difficulty
        
        # Linguistic patterns
        features['caps_ratio'] = texts.apply(lambda x: sum(c.isupper() for c in str(x)) / max(len(str(x)), 1))
        features['punctuation_ratio'] = texts.apply(lambda x: sum(c in '!?.,;:' for c in str(x)) / max(len(str(x)), 1))
        features['exclamation_ratio'] = texts.apply(lambda x: str(x).count('!') / max(len(str(x)), 1))
        features['question_ratio'] = texts.apply(lambda x: str(x).count('?') / max(len(str(x)), 1))
        
        # Spam indicators
        features['url_count'] = texts.str.count(r'http[s]?://|www\.|\.com|\.net|\.org')
        features['email_count'] = texts.str.count(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        features['phone_count'] = texts.str.count(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
        features['mention_count'] = texts.str.count(r'@\w+')
        features['hashtag_count'] = texts.str.count(r'#\w+')
        
        # Promotional language
        promo_words = ['discount', 'sale', 'promo', 'code', 'deal', 'offer', 'free', 'win', 'contest', 'dm me', 'follow']
        features['promo_word_count'] = texts.apply(lambda x: sum(word in str(x).lower() for word in promo_words))
        
        # Emotional indicators
        positive_words = ['great', 'amazing', 'awesome', 'fantastic', 'excellent', 'love', 'perfect', 'wonderful']
        negative_words = ['terrible', 'awful', 'horrible', 'worst', 'hate', 'disgusting', 'bad', 'poor']
        features['positive_word_count'] = texts.apply(lambda x: sum(word in str(x).lower() for word in positive_words))
        features['negative_word_count'] = texts.apply(lambda x: sum(word in str(x).lower() for word in negative_words))
        features['emotion_ratio'] = (features['positive_word_count'] + features['negative_word_count']) / features['word_count'].clip(lower=1)
        
        # Repetition patterns
        features['repeated_chars'] = texts.apply(lambda x: sum(1 for i in range(len(str(x))-1) if str(x)[i] == str(x)[i+1]))
        features['repeated_words'] = texts.apply(lambda x: len(str(x).split()) - len(set(str(x).lower().split())))
        
        return features.fillna(0)

class BusinessContextFeatures(BaseEstimator, TransformerMixin):
    """Extract business and user context features"""
    
    def __init__(self):
        self.fitted = False
        self.user_stats = {}
        self.business_stats = {}
        
    def fit(self, X, y=None):
        """Learn user and business patterns from training data"""
        log_progress("Learning user and business patterns...")
        log_progress(f"Available columns: {list(X.columns)}")
        
        # Calculate user statistics
        if 'user_id' in X.columns:
            user_groups = X.groupby('user_id')
            self.user_stats = {
                'review_counts': user_groups.size().to_dict(),
                'avg_ratings': user_groups['rating'].mean().to_dict() if 'rating' in X.columns else {},
                'text_lengths': user_groups['text'].apply(lambda x: x.str.len().mean()).to_dict() if 'text' in X.columns else {}
            }
            
            # Business diversity - try multiple possible business name columns
            business_col = None
            for col in ['name_y', 'business_name', 'name', 'gmap_id']:
                if col in X.columns:
                    business_col = col
                    break
            
            if business_col:
                self.user_stats['business_diversity'] = user_groups[business_col].nunique().to_dict()
                log_progress(f"Using '{business_col}' for business diversity calculation")
            else:
                self.user_stats['business_diversity'] = {}
                log_progress("Warning: No business identifier column found for diversity calculation")
        
        # Calculate business statistics  
        business_col = None
        for col in ['name_y', 'business_name', 'name', 'gmap_id']:
            if col in X.columns:
                business_col = col
                break
        
        if business_col:
            business_groups = X.groupby(business_col)
            self.business_stats = {
                'review_counts': business_groups.size().to_dict(),
                'avg_ratings': business_groups['rating'].mean().to_dict() if 'rating' in X.columns else {},
                'rating_variance': business_groups['rating'].var().to_dict() if 'rating' in X.columns else {}
            }
            log_progress(f"Using '{business_col}' for business statistics")
        else:
            self.business_stats = {
                'review_counts': {},
                'avg_ratings': {},
                'rating_variance': {}
            }
            log_progress("Warning: No business identifier column found")
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Extract context features"""
        features = pd.DataFrame(index=X.index)
        
        # Rating-based features
        if 'rating' in X.columns and 'avg_rating' in X.columns:
            features['rating_deviation'] = X['rating'] - X['avg_rating']
            features['rating_extremeness'] = np.abs(features['rating_deviation'])
            features['is_extreme_rating'] = ((X['rating'] <= 2) | (X['rating'] >= 4)).astype(int)
        
        # Business features
        if 'num_of_reviews' in X.columns:
            features['business_popularity'] = np.log1p(X['num_of_reviews'])
            features['is_new_business'] = (X['num_of_reviews'] < 50).astype(int)
        
        # Geographic features
        if 'latitude' in X.columns and 'longitude' in X.columns:
            # Simple geographic clustering (could be enhanced)
            features['lat_rounded'] = X['latitude'].round(1)
            features['lon_rounded'] = X['longitude'].round(1)
        
        # Price features
        if 'price' in X.columns:
            price_mapping = {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}
            features['price_level'] = X['price'].map(price_mapping).fillna(0)
        
        # User behavior features (if available)
        if 'user_id' in X.columns and self.user_stats:
            features['user_review_count'] = X['user_id'].map(self.user_stats.get('review_counts', {})).fillna(1)
            features['user_avg_rating'] = X['user_id'].map(self.user_stats.get('avg_ratings', {})).fillna(3.5)
            features['user_business_diversity'] = X['user_id'].map(self.user_stats.get('business_diversity', {})).fillna(1)
            features['is_prolific_reviewer'] = (features['user_review_count'] > 50).astype(int)
        else:
            # Default values if no user data available
            features['user_review_count'] = 1
            features['user_avg_rating'] = 3.5
            features['user_business_diversity'] = 1
            features['is_prolific_reviewer'] = 0
        
        # Business context features (if available) - try multiple business column names
        business_col = None
        for col in ['name_y', 'business_name', 'name', 'gmap_id']:
            if col in X.columns:
                business_col = col
                break
        
        if business_col and self.business_stats:
            features['business_review_count'] = X[business_col].map(self.business_stats.get('review_counts', {})).fillna(1)
            features['business_rating_variance'] = X[business_col].map(self.business_stats.get('rating_variance', {})).fillna(0.5)
        else:
            # Default values if no business column found
            features['business_review_count'] = 1
            features['business_rating_variance'] = 0.5
        
        # Response features
        if 'resp' in X.columns:
            features['has_response'] = (X['resp'].fillna('').str.len() > 0).astype(int)
            features['response_length'] = X['resp'].fillna('').str.len()
        else:
            features['has_response'] = 0
            features['response_length'] = 0
        
        # Picture features
        if 'pics' in X.columns:
            features['has_pictures'] = (X['pics'].fillna('').astype(str) != '').astype(int)
        else:
            features['has_pictures'] = 0
        
        # Geographic features defaults
        if 'latitude' not in X.columns or 'longitude' not in X.columns:
            features['lat_rounded'] = 0
            features['lon_rounded'] = 0
        
        # Business features defaults
        if 'num_of_reviews' not in X.columns:
            features['business_popularity'] = 0
            features['is_new_business'] = 0
        
        # Price features defaults
        if 'price' not in X.columns:
            features['price_level'] = 0
        
        # Rating features defaults
        if 'rating' not in X.columns or 'avg_rating' not in X.columns:
            features['rating_deviation'] = 0
            features['rating_extremeness'] = 0
            features['is_extreme_rating'] = 0
        
        return features.fillna(0)

# =============================================================================
# 2. SOPHISTICATED TEXT REPRESENTATIONS
# =============================================================================

class MultiLevelTextEncoder:
    """Creates multiple text representations for ensemble learning"""
    
    def __init__(self):
        self.tfidf_word = None
        self.tfidf_char = None
        self.hashing_vectorizer = None
        self.svd_transformer = None
        self.lda_model = None
        self.sentence_model = None
        self.fitted = False
    
    def fit(self, texts, y=None):
        """Fit all text encoders"""
        log_progress("Training multi-level text encoders...")
        
        texts = pd.Series(texts).fillna('').astype(str)
        
        # 1. TF-IDF Word Level
        self.tfidf_word = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2,
            max_df=0.95
        )
        tfidf_word_features = self.tfidf_word.fit_transform(texts)
        
        # 2. TF-IDF Character Level  
        self.tfidf_char = TfidfVectorizer(
            max_features=2000,
            ngram_range=(3, 5),
            analyzer='char',
            min_df=2
        )
        tfidf_char_features = self.tfidf_char.fit_transform(texts)
        
        # 3. Hashing for robustness
        self.hashing_vectorizer = HashingVectorizer(
            n_features=2**14,
            ngram_range=(1, 2),
            stop_words='english'
        )
        hashing_features = self.hashing_vectorizer.fit_transform(texts)
        
        # 4. SVD for dimensionality reduction
        from scipy.sparse import hstack
        combined_sparse = hstack([tfidf_word_features, tfidf_char_features, hashing_features])
        
        # 4. SVD for dimensionality reduction (ensure components <= features)
        n_svd_components = min(300, combined_sparse.shape[1] - 1, combined_sparse.shape[0] - 1)
        if n_svd_components < 1:
            n_svd_components = 1
        self.svd_transformer = TruncatedSVD(n_components=n_svd_components, random_state=42)
        self.svd_features = self.svd_transformer.fit_transform(combined_sparse)
        
        # 5. Topic modeling with LDA
        try:
            self.lda_model = LatentDirichletAllocation(
                n_components=min(20, tfidf_word_features.shape[1], len(texts)),  # Ensure components <= features
                random_state=42,
                max_iter=10,
                learning_method='batch'
            )
            self.lda_features = self.lda_model.fit_transform(tfidf_word_features)
        except Exception as e:
            log_progress(f"Warning: LDA failed ({e}), using dummy features")
            self.lda_model = None
            self.lda_features = np.zeros((len(texts), 20))
        
        # 6. Sentence embeddings (for semantic understanding)
        log_progress("Loading sentence transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Sample for efficiency during training
        sample_size = min(1000, len(texts))
        sample_texts = texts.sample(sample_size, random_state=42)
        self.sentence_embeddings_sample = self.sentence_model.encode(sample_texts.tolist())
        
        self.fitted = True
        log_progress("Multi-level text encoding complete!")
        
    def transform(self, texts):
        """Transform texts into multiple representations"""
        if not self.fitted:
            raise ValueError("Must fit encoder first")
            
        texts = pd.Series(texts).fillna('').astype(str)
        
        # Get all representations
        tfidf_word_features = self.tfidf_word.transform(texts)
        tfidf_char_features = self.tfidf_char.transform(texts)
        hashing_features = self.hashing_vectorizer.transform(texts)
        
        # SVD features
        from scipy.sparse import hstack
        combined_sparse = hstack([tfidf_word_features, tfidf_char_features, hashing_features])
        svd_features = self.svd_transformer.transform(combined_sparse)
        
        # LDA features
        if self.lda_model is not None:
            lda_features = self.lda_model.transform(tfidf_word_features)
        else:
            lda_features = np.zeros((len(texts), 20))
        
        # Sentence embeddings
        sentence_features = self.sentence_model.encode(texts.tolist())
        
        return {
            'tfidf_word': tfidf_word_features,
            'tfidf_char': tfidf_char_features, 
            'hashing': hashing_features,
            'svd': svd_features,
            'lda': lda_features,
            'sentence': sentence_features,
            'combined_sparse': combined_sparse
        }

# =============================================================================
# 3. ENSEMBLE ML PIPELINE
# =============================================================================

class RigorousMLPipeline:
    """Sophisticated ML pipeline with multiple models and validation"""
    
    def __init__(self):
        self.text_encoder = MultiLevelTextEncoder()
        self.text_features = AdvancedTextFeatures()
        self.context_features = BusinessContextFeatures()
        self.scaler = RobustScaler()
        
        # Multiple model types
        self.models = {}
        self.ensemble_model = None
        self.calibrated_model = None
        
        # Model performance tracking
        self.training_metrics = {}
        self.feature_importance = {}
        
        self.fitted = False
    
    def _create_models(self):
        """Create diverse set of models for ensemble"""
        models = {
            'logistic': LogisticRegression(
                C=1.0, 
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                class_weight='balanced',
                random_state=42,
                verbose=-1
            )
        }
        return models
    
    def _prepare_features(self, df, fit=False):
        """Prepare comprehensive feature set"""
        log_progress("Preparing comprehensive features...")
        
        # Text representations
        if fit:
            self.text_encoder.fit(df['text'])
        text_reps = self.text_encoder.transform(df['text'])
        
        # Advanced text features
        if fit:
            self.text_features.fit(df['text'])
        advanced_text = self.text_features.transform(df['text'])
        
        # Business context features
        if fit:
            self.context_features.fit(df)
        context_feats = self.context_features.transform(df)
        
        # Combine dense features (ensure consistent dimensions)
        feature_components = []
        
        # Add text representations
        feature_components.append(text_reps['svd'])
        feature_components.append(text_reps['lda'])
        feature_components.append(text_reps['sentence'])
        
        # Add structured features
        feature_components.append(advanced_text.values)
        feature_components.append(context_feats.values)
        
        # Debug shapes
        log_progress(f"Feature shapes: SVD={text_reps['svd'].shape}, LDA={text_reps['lda'].shape}, "
                    f"Sentence={text_reps['sentence'].shape}, Text={advanced_text.shape}, Context={context_feats.shape}")
        
        dense_features = np.concatenate(feature_components, axis=1)
        
        # Scale dense features
        if fit:
            dense_features_scaled = self.scaler.fit_transform(dense_features)
        else:
            dense_features_scaled = self.scaler.transform(dense_features)
        
        return {
            'dense': dense_features_scaled,
            'sparse': text_reps['combined_sparse'],
            'text_features': advanced_text,
            'context_features': context_feats
        }
    
    def fit(self, df, y):
        """Fit the complete ML pipeline"""
        start_time = time.time()
        log_progress("="*80)
        log_progress("TRAINING RIGOROUS ML PIPELINE")
        log_progress("="*80)
        
        log_progress(f"Training on {len(df)} samples with {np.bincount(y)} distribution")
        
        # Prepare features
        features = self._prepare_features(df, fit=True)
        X_dense = features['dense']
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_dense, y, test_size=0.2, random_state=42, stratify=y
        )
        
        log_progress(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Create and train individual models
        self.models = self._create_models()
        
        for name, model in self.models.items():
            log_progress(f"Training {name} model...")
            
            try:
                # Cross-validation (adjust CV folds for small datasets)
                cv_folds = min(5, len(np.unique(y_train)))
                if cv_folds < 2:
                    cv_folds = 2
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Validation performance
                val_pred = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_pred)
                
            except Exception as e:
                log_progress(f"Warning: {name} model training failed: {e}")
                cv_scores = np.array([0.5])  # Random performance
                val_auc = 0.5
            
            self.training_metrics[name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'val_auc': val_auc
            }
            
            log_progress(f"  {name}: CV AUC = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, Val AUC = {val_auc:.4f}")
        
        # Create ensemble
        log_progress("Creating ensemble model...")
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft'
        )
        self.ensemble_model.fit(X_train, y_train)
        
        # Calibrate ensemble
        log_progress("Calibrating ensemble probabilities...")
        self.calibrated_model = CalibratedClassifierCV(
            self.ensemble_model, 
            method='isotonic', 
            cv=3
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # Final validation
        final_pred = self.calibrated_model.predict_proba(X_val)[:, 1]
        final_auc = roc_auc_score(y_val, final_pred)
        
        self.training_metrics['ensemble_calibrated'] = {
            'val_auc': final_auc
        }
        
        self.fitted = True
        total_time = time.time() - start_time
        
        log_progress("="*80)
        log_progress(f"RIGOROUS ML TRAINING COMPLETE! Total time: {format_time(total_time)}")
        log_progress(f"Final Ensemble AUC: {final_auc:.4f}")
        log_progress("="*80)
        
        return self
    
    def predict_proba(self, df):
        """Predict probabilities for new data"""
        if not self.fitted:
            raise ValueError("Must fit pipeline first")
        
        features = self._prepare_features(df, fit=False)
        X_dense = features['dense']
        
        return self.calibrated_model.predict_proba(X_dense)[:, 1]
    
    def predict(self, df, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(df)
        return (probabilities > threshold).astype(int)
    
    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        importance_dict = {}
        
        if 'random_forest' in self.models:
            importance_dict['random_forest'] = self.models['random_forest'].feature_importances_
        
        if 'lightgbm' in self.models:
            importance_dict['lightgbm'] = self.models['lightgbm'].feature_importances_
        
        return importance_dict
    
    def save_models(self, output_dir):
        """Save all models and components to pickle files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        log_progress(f"Saving models to {output_dir}...")
        
        saved_files = []
        
        try:
            # Save main components
            components = {
                'text_encoder.pkl': self.text_encoder,
                'text_features.pkl': self.text_features,
                'context_features.pkl': self.context_features,
                'scaler.pkl': self.scaler
            }
            
            for filename, component in components.items():
                filepath = output_dir / filename
                with open(filepath, 'wb') as f:
                    pickle.dump(component, f)
                saved_files.append(filename)
                log_progress(f"  ✓ Saved {filename}")
            
            # Save individual models
            if hasattr(self, 'models') and self.models:
                for name, model in self.models.items():
                    filename = f'model_{name}.pkl'
                    filepath = output_dir / filename
                    with open(filepath, 'wb') as f:
                        pickle.dump(model, f)
                    saved_files.append(filename)
                    log_progress(f"  ✓ Saved {filename}")
            
            # Save ensemble model
            if hasattr(self, 'ensemble_model') and self.ensemble_model is not None:
                filename = 'ensemble_model.pkl'
                filepath = output_dir / filename
                with open(filepath, 'wb') as f:
                    pickle.dump(self.ensemble_model, f)
                saved_files.append(filename)
                log_progress(f"  ✓ Saved {filename}")
            
            # Save calibrated model (MAIN MODEL FOR INFERENCE)
            if hasattr(self, 'calibrated_model') and self.calibrated_model is not None:
                filename = 'calibrated_model.pkl'
                filepath = output_dir / filename
                with open(filepath, 'wb') as f:
                    pickle.dump(self.calibrated_model, f)
                saved_files.append(filename)
                log_progress(f"  ✓ Saved {filename} (MAIN INFERENCE MODEL)")
            
            # Save training metrics
            if hasattr(self, 'training_metrics'):
                filename = 'training_metrics.json'
                filepath = output_dir / filename
                with open(filepath, 'w') as f:
                    json.dump(self.training_metrics, f, indent=2)
                saved_files.append(filename)
                log_progress(f"  ✓ Saved {filename}")
            
            # Save feature importance
            try:
                importance = self.get_feature_importance()
                if importance:
                    filename = 'feature_importance.pkl'
                    filepath = output_dir / filename
                    with open(filepath, 'wb') as f:
                        pickle.dump(importance, f)
                    saved_files.append(filename)
                    log_progress(f"  ✓ Saved {filename}")
            except Exception as e:
                log_progress(f"  Warning: Could not save feature importance: {e}")
            
            # Save complete pipeline state
            pipeline_state = {
                'fitted': self.fitted,
                'model_names': list(self.models.keys()) if hasattr(self, 'models') else [],
                'training_complete': hasattr(self, 'calibrated_model') and self.calibrated_model is not None
            }
            filename = 'pipeline_state.json'
            filepath = output_dir / filename
            with open(filepath, 'w') as f:
                json.dump(pipeline_state, f, indent=2)
            saved_files.append(filename)
            log_progress(f"  ✓ Saved {filename}")
            
            log_progress(f"SUCCESS: All {len(saved_files)} model files saved to {output_dir}")
            log_progress(f"Saved files: {', '.join(saved_files)}")
            
            # Verify all files were saved and are readable
            self._verify_saved_models(output_dir, saved_files)
            
        except Exception as e:
            log_progress(f"ERROR saving models: {e}")
            raise
    
    def _verify_saved_models(self, output_dir, expected_files):
        """Verify that all model files were saved correctly and are loadable"""
        log_progress("Verifying saved model files...")
        
        verified_files = []
        failed_files = []
        
        for filename in expected_files:
            filepath = output_dir / filename
            try:
                if filename.endswith('.pkl'):
                    # Try to load pickle file to verify it's valid
                    with open(filepath, 'rb') as f:
                        pickle.load(f)
                elif filename.endswith('.json'):
                    # Try to load JSON file to verify it's valid
                    with open(filepath, 'r') as f:
                        json.load(f)
                
                # Check file size
                file_size = filepath.stat().st_size
                if file_size > 0:
                    verified_files.append(f"{filename} ({file_size} bytes)")
                else:
                    failed_files.append(f"{filename} (empty file)")
                    
            except Exception as e:
                failed_files.append(f"{filename} (error: {e})")
        
        if failed_files:
            log_progress(f"WARNING: {len(failed_files)} files failed verification: {failed_files}")
        
        log_progress(f"✓ Verified {len(verified_files)} model files:")
        for file_info in verified_files:
            log_progress(f"    {file_info}")
        
        return len(failed_files) == 0
    
    @classmethod
    def load_models(cls, model_dir):
        """Load pre-trained models from pickle files"""
        model_dir = Path(model_dir)
        log_progress(f"Loading models from {model_dir}...")
        
        # Check if required files exist
        required_files = [
            'text_encoder.pkl', 'text_features.pkl', 'context_features.pkl', 
            'scaler.pkl', 'calibrated_model.pkl'
        ]
        
        missing_files = []
        for file in required_files:
            if not (model_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            raise FileNotFoundError(f"Missing required model files: {missing_files}")
        
        pipeline = cls()
        loaded_files = []
        
        try:
            # Load main components
            components = {
                'text_encoder.pkl': 'text_encoder',
                'text_features.pkl': 'text_features', 
                'context_features.pkl': 'context_features',
                'scaler.pkl': 'scaler'
            }
            
            for filename, attr_name in components.items():
                filepath = model_dir / filename
                with open(filepath, 'rb') as f:
                    setattr(pipeline, attr_name, pickle.load(f))
                loaded_files.append(filename)
                log_progress(f"  ✓ Loaded {filename}")
            
            # Load calibrated model (MAIN MODEL FOR INFERENCE)
            filepath = model_dir / 'calibrated_model.pkl'
            with open(filepath, 'rb') as f:
                pipeline.calibrated_model = pickle.load(f)
            loaded_files.append('calibrated_model.pkl')
            log_progress(f"  ✓ Loaded calibrated_model.pkl (MAIN INFERENCE MODEL)")
            
            # Load training metrics (optional)
            metrics_file = model_dir / 'training_metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    pipeline.training_metrics = json.load(f)
                loaded_files.append('training_metrics.json')
                log_progress(f"  ✓ Loaded training_metrics.json")
            else:
                pipeline.training_metrics = {}
                log_progress(f"  ! training_metrics.json not found (optional)")
            
            # Load individual models (optional, for analysis)
            model_files = list(model_dir.glob('model_*.pkl'))
            if model_files:
                pipeline.models = {}
                for model_file in model_files:
                    model_name = model_file.stem.replace('model_', '')
                    with open(model_file, 'rb') as f:
                        pipeline.models[model_name] = pickle.load(f)
                    loaded_files.append(model_file.name)
                    log_progress(f"  ✓ Loaded {model_file.name}")
            
            # Load ensemble model (optional)
            ensemble_file = model_dir / 'ensemble_model.pkl'
            if ensemble_file.exists():
                with open(ensemble_file, 'rb') as f:
                    pipeline.ensemble_model = pickle.load(f)
                loaded_files.append('ensemble_model.pkl')
                log_progress(f"  ✓ Loaded ensemble_model.pkl")
            
            pipeline.fitted = True
            
            log_progress(f"SUCCESS: Loaded {len(loaded_files)} model files from {model_dir}")
            log_progress(f"Loaded files: {', '.join(loaded_files)}")
            log_progress("Pipeline ready for inference!")
            
            return pipeline
            
        except Exception as e:
            log_progress(f"ERROR loading models: {e}")
            raise

# =============================================================================
# 4. INTEGRATION WITH PIPELINE
# =============================================================================

def run(config: Dict[str, Any]) -> bool:
    """Main entry point for Stage 3 - Rigorous ML Pipeline"""
    try:
        log_progress("="*80)
        log_progress("STAGE 3: RIGOROUS ML-BASED CLASSIFICATION")
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
            model_output_dir = Path(execution_output_dir) / "models"
        else:
            input_file = config.get('input_file', 'data/stage2_output.csv')
            output_file = config.get('output_file', 'data/stage3_output.csv')
            model_output_dir = Path("models")
        
        log_progress(f"Input: {input_file}")
        log_progress(f"Output: {output_file}")
        log_progress(f"Models will be saved to: {model_output_dir}")
        
        # Load data (Stage 2 output - rule-filtered data)
        log_progress(f"Loading Stage 2 filtered data from {input_file}")
        df = pd.read_csv(input_file, sep=';', encoding='utf-8')
        log_progress(f"Loaded {len(df)} reviews that passed rule-based filtering")
        
        # Check for existing labels
        has_labels = False
        y = None
        
        if 'Label' in df.columns:
            # Map labels: valid->0 (relevant), invalid->1 (junk)
            label_mapping = {
                'valid': 0, 'invalid': 1,
                'Valid': 0, 'Invalid': 1,
                'VALID': 0, 'INVALID': 1
            }
            df['model_label'] = df['Label'].map(label_mapping)
            # Only use rows with valid labels
            valid_mask = df['model_label'].notna()
            if valid_mask.sum() > 0:
                df_labeled = df[valid_mask].copy()
                y = df_labeled['model_label'].values.astype(int)
                has_labels = True
                log_progress(f"Found ground truth labels: {np.bincount(y)} (0=relevant, 1=junk)")
                
                # Use labeled data for training
                train_df = df_labeled
                train_y = y
            else:
                log_progress("No valid labels found")
                train_df = df
                train_y = None
        else:
            log_progress("No Label column found")
            train_df = df
            train_y = None
        
        # Check if we can load existing models
        required_model_files = [
            'calibrated_model.pkl', 'text_encoder.pkl', 'text_features.pkl', 
            'context_features.pkl', 'scaler.pkl'
        ]
        
        model_files_exist = all((model_output_dir / file).exists() for file in required_model_files)
        use_cached = model_config.get('use_cached_models', True)
        
        log_progress(f"Checking for cached models in: {model_output_dir}")
        log_progress(f"Required files: {required_model_files}")
        
        # Check each file and show details
        file_status = []
        total_size = 0
        
        for file in required_model_files:
            filepath = model_output_dir / file
            if filepath.exists():
                size = filepath.stat().st_size
                size_mb = size / (1024 * 1024)
                total_size += size
                file_status.append(f"[OK] {file} ({size_mb:.1f} MB)")
            else:
                file_status.append(f"[MISSING] {file}")
        
        log_progress("Model file status:")
        for status in file_status:
            log_progress(f"    {status}")
        
        missing_files = [f for f in required_model_files if not (model_output_dir / f).exists()]
        if missing_files:
            log_progress(f"Missing model files: {missing_files}")
        else:
            total_mb = total_size / (1024 * 1024)
            log_progress(f"All required model files found! Total size: {total_mb:.1f} MB")
        
        if model_files_exist and use_cached:
            log_progress("LOADING EXISTING TRAINED MODELS (CACHED)")
            log_progress("="*60)
            try:
                ml_pipeline = RigorousMLPipeline.load_models(model_output_dir)
                log_progress("="*60)
                log_progress("SUCCESS: LOADED CACHED MODELS - SKIPPING TRAINING")
                log_progress("="*60)
            except Exception as e:
                log_progress(f"FAILED to load cached models: {e}")
                log_progress("Will train new models instead...")
                model_files_exist = False
        elif not use_cached:
            log_progress("WARNING: use_cached_models=False - Will train new models")
        else:
            log_progress("No cached models found - Will train new models")
        
        if not model_files_exist:
            if train_y is None:
                log_progress("ERROR: No labels available for training and no cached models found")
                return False
            
            # Train new models
            log_progress("Training new ML pipeline...")
            ml_pipeline = RigorousMLPipeline()
            ml_pipeline.fit(train_df, train_y)
            
            # Save models
            ml_pipeline.save_models(model_output_dir)
        
        # Run inference on all data
        log_progress("Running ML inference on all reviews...")
        predictions = ml_pipeline.predict(df)
        probabilities = ml_pipeline.predict_proba(df)
        
        # Add results to dataframe
        df['ml_prediction'] = predictions
        df['ml_probability'] = probabilities
        df['ml_confidence'] = np.abs(probabilities - 0.5) * 2  # 0 = uncertain, 1 = very confident
        
        # Performance analysis
        if has_labels:
            from sklearn.metrics import classification_report, confusion_matrix
            
            # Only evaluate on labeled data
            labeled_mask = df['model_label'].notna()
            y_true = df.loc[labeled_mask, 'model_label'].values
            y_pred = df.loc[labeled_mask, 'ml_prediction'].values
            y_prob = df.loc[labeled_mask, 'ml_probability'].values
            
            # Generate detailed report
            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_prob)
            
            log_progress("="*60)
            log_progress("ML PERFORMANCE REPORT:")
            log_progress(f"  Accuracy: {report['accuracy']:.4f}")
            log_progress(f"  Precision (Class 1): {report['1']['precision']:.4f}")
            log_progress(f"  Recall (Class 1): {report['1']['recall']:.4f}")
            log_progress(f"  F1 Score (Class 1): {report['1']['f1-score']:.4f}")
            log_progress(f"  AUC Score: {auc_score:.4f}")
            log_progress(f"  Confusion Matrix: {cm.tolist()}")
            log_progress("="*60)
        
        # Summary statistics
        total_junk = np.sum(predictions == 1)
        total_relevant = np.sum(predictions == 0)
        avg_confidence = np.mean(df['ml_confidence'])
        
        log_progress("="*60)
        log_progress("PROCESSING SUMMARY:")
        log_progress(f"  Input (post Stage 2): {len(df)} reviews")
        log_progress(f"  Predicted Relevant: {total_relevant} ({total_relevant/len(df)*100:.1f}%)")
        log_progress(f"  Predicted Junk: {total_junk} ({total_junk/len(df)*100:.1f}%)")
        log_progress(f"  Average Confidence: {avg_confidence:.3f}")
        log_progress(f"  High Confidence (>0.8): {np.sum(df['ml_confidence'] > 0.8)} ({np.sum(df['ml_confidence'] > 0.8)/len(df)*100:.1f}%)")
        log_progress("="*60)
        
        # Save results
        df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        log_progress(f"Results saved to {output_file}")
        
        # Save summary report
        summary_file = Path(output_file).parent / "stage3_ml_report.json"
        summary_data = {
            'input_count': len(df),
            'predicted_relevant': int(total_relevant),
            'predicted_junk': int(total_junk),
            'average_confidence': float(avg_confidence),
            'high_confidence_count': int(np.sum(df['ml_confidence'] > 0.8)),
            'training_metrics': ml_pipeline.training_metrics if hasattr(ml_pipeline, 'training_metrics') else {}
        }
        
        if has_labels:
            summary_data.update({
                'performance': {
                    'accuracy': float(report['accuracy']),
                    'precision': float(report['1']['precision']),
                    'recall': float(report['1']['recall']),
                    'f1_score': float(report['1']['f1-score']),
                    'auc_score': float(auc_score)
                }
            })
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        log_progress(f"Summary report saved to {summary_file}")
        
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
            'use_cached_models': True
        }
    }
    
    log_progress("Testing Stage 3 rigorous ML pipeline...")
    success = run(test_config)
    log_progress(f"Stage 3 test: {'SUCCESS' if success else 'FAILED'}")
