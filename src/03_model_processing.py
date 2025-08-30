"""
Pipeline Stage 3: Model Processing
This module handles BERT-based model training and inference using structured and text features.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import pickle
import json
import hashlib
import time
import sys

def log_progress(message, force_flush=True):
    """Log message with immediate flush to both console and file"""
    logging.info(message)
    if force_flush:
        # Force flush all handlers
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

class ReviewDataset(Dataset):
    def __init__(self, texts, structured, labels, tokenizer, max_len):
        self.texts = texts
        self.structured = structured
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        structured = self.structured[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'structured': torch.tensor(structured, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BertWithStructured(nn.Module):
    def __init__(self, n_structured, n_classes=2):
        super(BertWithStructured, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        bert_hidden_size = self.bert.config.hidden_size
        self.fc1 = nn.Linear(bert_hidden_size + n_structured, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, n_classes)
    
    def forward(self, input_ids, attention_mask, structured):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = bert_out.pooler_output  # [CLS] token
        x = torch.cat((cls_output, structured), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def generate_model_hash(df: pd.DataFrame, model_config: Dict[str, Any], structured_features: list) -> str:
    """
    Generate a hash based on data characteristics and model config to identify cached models.
    
    Args:
        df: Input dataframe
        model_config: Model configuration parameters
        structured_features: List of structured feature names
        
    Returns:
        str: Hash string for model identification
    """
    # Create a signature of the data and model config
    data_signature = {
        'num_rows': len(df),
        'columns': sorted(df.columns.tolist()),
        'structured_features': sorted(structured_features),
        'label_distribution': df['model_label'].value_counts().to_dict() if 'model_label' in df.columns else {},
        'model_config': model_config
    }
    
    # Convert to string and hash
    signature_str = json.dumps(data_signature, sort_keys=True)
    return hashlib.md5(signature_str.encode()).hexdigest()[:12]  # Use first 12 chars for readability

def save_model_artifacts(model: BertWithStructured, 
                        scaler: Optional[StandardScaler], 
                        encoder: Optional[OneHotEncoder],
                        structured_features: list,
                        model_config: Dict[str, Any],
                        metrics: Dict[str, Any],
                        model_hash: str) -> Dict[str, str]:
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    artifacts = {}
    
    # Save PyTorch model
    model_path = models_dir / f"bert_model_{model_hash}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'n_structured': len(structured_features),
            'structured_features': structured_features,
            'n_classes': 2
        }
    }, model_path)
    artifacts['model'] = str(model_path)
    logging.info(f"Saved model to {model_path}")
    
    # Save scaler if exists
    if scaler is not None:
        scaler_path = models_dir / f"scaler_{model_hash}.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        artifacts['scaler'] = str(scaler_path)
        logging.info(f"Saved scaler to {scaler_path}")
    
    # Save encoder if exists  
    if encoder is not None:
        encoder_path = models_dir / f"encoder_{model_hash}.pkl"
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoder, f)
        artifacts['encoder'] = str(encoder_path)
        logging.info(f"Saved encoder to {encoder_path}")
    
    # Save metadata
    metadata = {
        'model_hash': model_hash,
        'structured_features': structured_features,
        'model_config': model_config,
        'metrics': metrics,
        'artifact_paths': artifacts,
        'created_at': datetime.now().isoformat()
    }
    
    metadata_path = models_dir / f"metadata_{model_hash}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    artifacts['metadata'] = str(metadata_path)
    logging.info(f"Saved metadata to {metadata_path}")
    
    return artifacts

def load_model_artifacts(model_hash: str) -> Optional[Tuple[BertWithStructured, Optional[StandardScaler], Optional[OneHotEncoder], Dict[str, Any]]]:
    """
    Load model artifacts from the models/ directory.
    
    Returns:
        Tuple of (model, scaler, encoder, metadata) or None if not found
    """
    models_dir = Path("models")
    metadata_path = models_dir / f"metadata_{model_hash}.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_path = models_dir / f"bert_model_{model_hash}.pth"
        if not model_path.exists():
            logging.error(f"Model file not found: {model_path}")
            return None
            
        checkpoint = torch.load(model_path, map_location='cpu')
        model_config = checkpoint['model_config']
        
        model = BertWithStructured(
            n_structured=model_config['n_structured'],
            n_classes=model_config['n_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded model from {model_path}")
        
        # Load scaler if exists
        scaler = None
        scaler_path = models_dir / f"scaler_{model_hash}.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            logging.info(f"Loaded scaler from {scaler_path}")
        
        # Load encoder if exists
        encoder = None
        encoder_path = models_dir / f"encoder_{model_hash}.pkl"
        if encoder_path.exists():
            with open(encoder_path, 'rb') as f:
                encoder = pickle.load(f)
            logging.info(f"Loaded encoder from {encoder_path}")
        
        return model, scaler, encoder, metadata
        
    except Exception as e:
        logging.error(f"Error loading model artifacts: {e}")
        return None

def run(config: Dict[str, Any]) -> bool:
    """
    Execute the model processing pipeline stage.
    
    Args:
        config: Configuration dictionary containing:
            - input_file: Path to input CSV file from stage 2
            - output_file: Path to output CSV file with predictions
            - execution_output_dir: Directory for execution outputs
            - model_config: Model configuration parameters
            
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get configuration parameters
        input_file = config.get('input_file')
        output_file = config.get('output_file', 'data/stage3_output.csv')
        execution_output_dir = config.get('execution_output_dir')
        execution_id = config.get('execution_id')
        
        # Model configuration
        model_config = config.get('model_config', {})
        epochs = model_config.get('epochs', 3)
        batch_size = model_config.get('batch_size', 16)
        max_len = model_config.get('max_len', 128)
        learning_rate = float(model_config.get('learning_rate', 2e-5))
        test_size = model_config.get('test_size', 0.2)
        
        # Generate timestamp for metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Determine input and output paths
        if execution_output_dir:
            # Look for previous stage output
            previous_stage = config.get('previous_stage', '02_rule_based_filtering')
            stage_number = previous_stage.split('_')[0] if previous_stage else '02'
            previous_output_file = f"stage{stage_number[-1]}_output.csv"
            
            stage_input_file = Path(execution_output_dir) / previous_output_file
            input_file = str(stage_input_file)
            logging.info(f"Stage 3: Using previous stage ({previous_stage}) output: {input_file}")
            output_file = str(Path(execution_output_dir) / "stage3_output.csv")
            model_output_dir = Path(execution_output_dir)
        else:
            # Use timestamped output filename
            output_path = Path(output_file)
            output_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
            output_file = str(output_path.parent / output_name)
            model_output_dir = output_path.parent
        
        logging.info(f"Stage 3: Loading data from {input_file}")
        logging.info(f"Stage 3: Model output will be saved to {output_file}")
        
        # Load data from stage 2
        df = pd.read_csv(input_file, sep=';', encoding='utf-8')
        initial_count = len(df)
        logging.info(f"Loaded {initial_count} reviews from stage 2")
        
        # Check if we have the required columns
        required_columns = ['text', 'rating']
        available_columns = list(df.columns)
        missing_columns = [col for col in required_columns if col not in available_columns]
        
        if missing_columns:
            logging.error(f"Missing required columns: {missing_columns}")
            logging.error(f"Available columns: {available_columns}")
            return False
        
        # Create labels based on business rules (example: rating < 3 = invalid)
        # You can modify this logic based on your specific requirements
        df['model_label'] = (df['rating'] < 3).astype(int)  # 0 = valid, 1 = invalid
        
        # Drop rows with missing text or labels
        df = df.dropna(subset=['text', 'model_label'])
        logging.info(f"After dropping missing values: {len(df)} reviews")
        
        if len(df) == 0:
            logging.error("No valid data available for model training")
            return False
        
        # Use Model_1.py exact feature configuration
        numeric_features = ['rating', 'avg_rating', 'num_of_reviews', 'pics']
        categorical_features = ['category']
        
        logging.info(f"Using Model_1.py feature configuration:")
        logging.info(f"Numeric features: {numeric_features}")
        logging.info(f"Categorical features: {categorical_features}")
        
        # Check if required features exist in the data
        available_numeric = [f for f in numeric_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
        missing_numeric = [f for f in numeric_features if f not in df.columns]
        missing_categorical = [f for f in categorical_features if f not in df.columns]
        
        if missing_numeric:
            logging.warning(f"Missing numeric features: {missing_numeric}")
        if missing_categorical:
            logging.warning(f"Missing categorical features: {missing_categorical}")
        
        # Convert to numeric, filling NaN with 0 for available features
        for col in available_numeric:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Convert to string, filling NaN with 'unknown' for available features  
        # Process exactly like Model_1.py (no special cleaning)
        for col in available_categorical:
            df[col] = df[col].astype(str).fillna('unknown')
            logging.info(f"Column '{col}' has {df[col].nunique()} unique values")
            logging.info(f"Sample {col} values: {df[col].unique()[:10]}")
        
        # Numeric scaler (exactly like Model_1.py)
        scaler = StandardScaler()
        if available_numeric:
            df[available_numeric] = scaler.fit_transform(df[available_numeric])
        
        # Categorical one-hot (exactly like Model_1.py) 
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        if available_categorical:
            cat_encoded = encoder.fit_transform(df[available_categorical])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(available_categorical))
            df = pd.concat([df.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)
        else:
            cat_encoded_df = pd.DataFrame()
        
        # Combine all structured features (exactly like Model_1.py)
        structured_features = available_numeric + list(cat_encoded_df.columns)
        
        logging.info(f"Available numeric features: {available_numeric}")
        logging.info(f"Available categorical features: {available_categorical}")
        logging.info(f"One-hot encoded categorical columns: {list(cat_encoded_df.columns)}")
        logging.info(f"Final structured features list: {structured_features}")
        logging.info(f"Total structured features: {len(structured_features)}")
        
        # Set device for model loading
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate model hash for caching
        model_hash = generate_model_hash(df, model_config, structured_features)
        logging.info(f"Generated model hash: {model_hash}")
        
        # Force retraining with improved training process
        logging.info("Starting fresh model training with current data...")
        cached_artifacts = None
        
        if cached_artifacts is not None:
            if 'loaded_from' not in str(cached_artifacts[3]):
                logging.info("Found cached model! Loading existing artifacts instead of training...")
            model, cached_scaler, cached_encoder, cached_metadata = cached_artifacts
            
            # Use cached preprocessors if they exist, otherwise create new ones
            if cached_scaler is not None and numeric_features:
                scaler = cached_scaler
                df[numeric_features] = scaler.transform(df[numeric_features])
                logging.info("Using cached scaler for numeric features")
            elif numeric_features:
                logging.warning("Cached scaler not found, creating new one")
                scaler = StandardScaler()
                df[numeric_features] = scaler.fit_transform(df[numeric_features])
            else:
                scaler = None
                
            if cached_encoder is not None and categorical_features:
                encoder = cached_encoder
                cat_encoded = encoder.transform(df[categorical_features])
                cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_features))
                df = pd.concat([df.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)
                logging.info("Using cached encoder for categorical features")
            elif categorical_features:
                logging.warning("Cached encoder not found, creating new one")
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                cat_encoded = encoder.fit_transform(df[categorical_features])
                cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_features))
                df = pd.concat([df.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)
            else:
                encoder = None
            
            # Set device and move model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            logging.info(f"Loaded cached model on device: {device}")
            
            # Skip to prediction generation
            skip_training = True
            metrics = cached_metadata.get('metrics', {})
            
        else:
            logging.info("No cached model found. Starting training process...")
            skip_training = False
            

        
        # Initialize BERT tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Prepare data for training
        X_text = df['text'].values
        X_struct = df[structured_features].values
        y = df['model_label'].values
        
        # Check class distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        logging.info(f"Class distribution: {dict(zip(unique_labels, counts))}")
        
        # Only proceed with training if model not cached
        if not skip_training:
            # Train/Test Split
            X_text_train, X_text_val, X_struct_train, X_struct_val, y_train, y_val = train_test_split(
                X_text, X_struct, y, test_size=test_size, random_state=42, stratify=y
            )
            
            logging.info(f"Training samples: {len(X_text_train)}")
            logging.info(f"Validation samples: {len(X_text_val)}")
            
            # Create datasets
            train_dataset = ReviewDataset(X_text_train, X_struct_train, y_train, tokenizer, max_len)
            val_dataset = ReviewDataset(X_text_val, X_struct_val, y_val, tokenizer, max_len)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Initialize model
            model = BertWithStructured(n_structured=len(structured_features))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            logging.info(f"Using device: {device}")
            logging.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
            
            # Enhanced training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, eps=1e-8)
            
            # Learning rate scheduler with warmup
            from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
            
            total_steps = len(train_loader) * epochs
            warmup_steps = min(100, total_steps // 10)  # 10% warmup or max 100 steps
            
            warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
            scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
            
            # Enhanced loss function with class weighting for imbalanced data
            class_counts = np.bincount(y_train)
            class_weights = len(y_train) / (len(class_counts) * class_counts)
            class_weights_tensor = torch.FloatTensor(class_weights).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            
            logging.info(f"Class weights for balanced training: {class_weights}")
            logging.info(f"Warmup steps: {warmup_steps}, Total steps: {total_steps}")
            
            # Training loop with live progress tracking
            log_progress(f"Starting enhanced training for {epochs} epochs")
            training_history = []
            
            # Training timing
            training_start_time = time.time()
            
            for epoch in range(epochs):
                epoch_start_time = time.time()
                model.train()
                total_loss = 0
                num_batches = 0
                
                log_progress(f"\nEpoch {epoch+1}/{epochs} - Training Phase")
                log_progress("=" * 60)
                
                for batch_idx, batch in enumerate(train_loader):
                    batch_start_time = time.time()
                    
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    structured = batch['structured'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask, structured)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    scheduler.step()  # Step the learning rate scheduler
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    batch_time = time.time() - batch_start_time
                    
                    # Live progress updates every 5% of batches or every 10 batches (whichever is smaller)
                    update_interval = min(max(1, len(train_loader) // 20), 10)
                    if batch_idx % update_interval == 0 or batch_idx == len(train_loader) - 1:
                        progress = (batch_idx + 1) / len(train_loader)
                        elapsed_epoch = time.time() - epoch_start_time
                        eta_epoch = elapsed_epoch / progress - elapsed_epoch if progress > 0 else 0
                        
                        current_lr = scheduler.get_last_lr()[0]
                        progress_bar = create_progress_bar(batch_idx + 1, len(train_loader), width=30)
                        
                        log_progress(f"  {progress_bar} Batch {batch_idx+1}/{len(train_loader)} | "
                                   f"Loss: {loss.item():.4f} | LR: {current_lr:.2e} | "
                                   f"Time: {format_time(batch_time)} | ETA: {format_time(eta_epoch)}")
                
                avg_loss = total_loss / num_batches
                epoch_train_time = time.time() - epoch_start_time
                
                # Validation evaluation every epoch
                log_progress(f"\nEpoch {epoch+1}/{epochs} - Validation Phase")
                log_progress("-" * 60)
                
                val_start_time = time.time()
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(val_loader):
                        val_input_ids = val_batch['input_ids'].to(device)
                        val_attention_mask = val_batch['attention_mask'].to(device)
                        val_structured = val_batch['structured'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        
                        val_outputs = model(val_input_ids, val_attention_mask, val_structured)
                        val_loss += criterion(val_outputs, val_labels).item()
                        
                        _, predicted = torch.max(val_outputs.data, 1)
                        val_total += val_labels.size(0)
                        val_correct += (predicted == val_labels).sum().item()
                        
                        # Show validation progress every 20% or every 5 batches
                        val_update_interval = min(max(1, len(val_loader) // 5), 5)
                        if val_batch_idx % val_update_interval == 0 or val_batch_idx == len(val_loader) - 1:
                            val_progress_bar = create_progress_bar(val_batch_idx + 1, len(val_loader), width=30)
                            current_val_acc = val_correct / val_total if val_total > 0 else 0
                            log_progress(f"  {val_progress_bar} Val Batch {val_batch_idx+1}/{len(val_loader)} | "
                                       f"Acc: {current_val_acc:.4f}")
                
                val_accuracy = val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)
                val_time = time.time() - val_start_time
                epoch_total_time = time.time() - epoch_start_time
                
                # Calculate overall training progress
                total_training_time = time.time() - training_start_time
                overall_progress = (epoch + 1) / epochs
                eta_total = total_training_time / overall_progress - total_training_time if overall_progress > 0 else 0
                
                training_history.append({
                    'epoch': epoch + 1, 
                    'train_loss': avg_loss, 
                    'val_loss': avg_val_loss,
                    'val_accuracy': val_accuracy,
                    'epoch_time': epoch_total_time
                })
                
                log_progress("=" * 60)
                log_progress(f"EPOCH {epoch+1}/{epochs} SUMMARY:")
                log_progress(f"  Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
                log_progress(f"  Epoch Time: {format_time(epoch_total_time)} (Train: {format_time(epoch_train_time)}, Val: {format_time(val_time)})")
                log_progress(f"  Overall Progress: {create_progress_bar(epoch + 1, epochs, width=40)}")
                log_progress(f"  Total Time: {format_time(total_training_time)} | ETA: {format_time(eta_total)}")
                log_progress("=" * 60)
                
                model.train()  # Switch back to training mode
            
            # Training completion summary
            total_training_time = time.time() - training_start_time
            log_progress("\n" + "*" * 60)
            log_progress("*** TRAINING COMPLETED SUCCESSFULLY! ***")
            log_progress(f"Total Training Time: {format_time(total_training_time)}")
            log_progress(f"Final Validation Accuracy: {val_accuracy:.4f}")
            log_progress(f"Best Epoch: {max(training_history, key=lambda x: x['val_accuracy'])['epoch']}")
            log_progress("*" * 60 + "\n")
            
            # Evaluation
            log_progress("Starting final model evaluation on validation set...")
            model.eval()
            all_preds = []
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    structured = batch['structured'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask, structured)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
            
            # Generate classification report
            report = classification_report(all_labels, all_preds, output_dict=True)
            logging.info("Classification Report:")
            logging.info(classification_report(all_labels, all_preds))
            
            # Create metrics for saving
            metrics = {
                'classification_report': report,
                'training_history': training_history
            }
            
            # Save the newly trained model to cache
            logging.info("Saving trained model to cache...")
            save_model_artifacts(model, scaler, encoder, structured_features, model_config, metrics, model_hash)
        else:
            logging.info("Using cached model, skipping training and evaluation steps")
        

        
        # Generate predictions for the entire dataset
        full_dataset = ReviewDataset(X_text, X_struct, y, tokenizer, max_len)
        full_loader = DataLoader(full_dataset, batch_size=batch_size)
        
        all_full_preds = []
        all_full_probs = []
        
        model.eval()
        with torch.no_grad():
            for batch in full_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                structured = batch['structured'].to(device)
                
                outputs = model(input_ids, attention_mask, structured)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_full_preds.extend(preds.cpu().numpy())
                all_full_probs.extend(probs.cpu().numpy())
        
        # Add predictions to dataframe
        df['model_prediction'] = all_full_preds
        df['model_confidence'] = [prob[pred] for prob, pred in zip(all_full_probs, all_full_preds)]
        df['model_prob_invalid'] = [prob[1] for prob in all_full_probs]
        
        # Save results
        df.to_csv(output_file, sep=';', index=False, encoding='utf-8')
        logging.info(f"Stage 3: Saved {len(df)} rows with predictions to {output_file}")
        
        # Save evaluation metrics to execution directory
        metrics_path = model_output_dir / "model_metrics.json"
        execution_metrics = {
            'execution_id': execution_id,
            'timestamp': timestamp,
            'model_hash': model_hash,
            'total_samples': len(df),
            'structured_features': structured_features,
            'model_config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'max_len': max_len,
                'learning_rate': learning_rate,
                'test_size': test_size
            },
            'class_distribution': dict(zip([str(x) for x in unique_labels], [int(x) for x in counts])),
            'cached_model_used': skip_training,
            'metrics': metrics
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(execution_metrics, f, indent=2)
        logging.info(f"Saved execution metrics to {metrics_path}")
        
        # Log summary statistics
        logging.info(f"Stage 3 Summary:")
        logging.info(f"  - Total samples processed: {len(df)}")
        logging.info(f"  - Structured features: {len(structured_features)}")
        logging.info(f"  - Model hash: {model_hash}")
        logging.info(f"  - Used cached model: {skip_training}")
        if not skip_training and 'classification_report' in metrics:
            logging.info(f"  - Model accuracy: {metrics['classification_report']['accuracy']:.4f}")
            logging.info(f"  - Training completed in {epochs} epochs")
        elif skip_training and 'classification_report' in metrics:
            logging.info(f"  - Cached model accuracy: {metrics['classification_report']['accuracy']:.4f}")
        
        return True
        
    except Exception as e:
        logging.error(f"Error in stage 3 (model processing): {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Test the module independently
    test_config = {
        'input_file': 'data/stage2_output.csv',
        'output_file': 'data/stage3_output.csv',
        'model_config': {
            'epochs': 1,
            'batch_size': 8,
            'max_len': 64
        }
    }
    success = run(test_config)
    print(f"Stage 3 execution: {'SUCCESS' if success else 'FAILED'}")
