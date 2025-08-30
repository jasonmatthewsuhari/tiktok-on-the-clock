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
        
        # Prepare features
        numeric_features = []
        categorical_features = []
        
        # Check for available numeric features (expanded for new schema + temporal + text quality)
        potential_numeric = ['rating', 'avg_rating', 'num_of_reviews', 'pics', 'latitude', 'longitude',
                            'hour_of_day', 'day_of_week', 'month', 'year', 'is_weekend', 
                            'has_business_response', 'response_length', 'text_length', 'word_count', 
                            'exclamation_count', 'caps_ratio', 'rating_deviation']
        for col in potential_numeric:
            if col in df.columns:
                # Convert to numeric, filling NaN with 0
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                numeric_features.append(col)
        
        # Check for available categorical features (expanded for new schema)
        potential_categorical = ['category', 'rating_category', 'state', 'price']
        for col in potential_categorical:
            if col in df.columns:
                df[col] = df[col].astype(str).fillna('unknown')
                # Clean category field if it contains JSON-like data
                if col == 'category':
                    df[col] = df[col].str.replace(r'[\[\]\']', '', regex=True)
                    df[col] = df[col].str.replace(',', ';').fillna('unknown')
                categorical_features.append(col)
        
        logging.info(f"Numeric features: {numeric_features}")
        logging.info(f"Categorical features: {categorical_features}")
        
        # Preprocessing
        structured_features = []
        
        # Scale numeric features
        if numeric_features:
            scaler = StandardScaler()
            df[numeric_features] = scaler.fit_transform(df[numeric_features])
            structured_features.extend(numeric_features)
        else:
            scaler = None
            
        # One-hot encode categorical features
        if categorical_features:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_encoded = encoder.fit_transform(df[categorical_features])
            cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_features))
            df = pd.concat([df.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)
            structured_features.extend(list(cat_encoded_df.columns))
        else:
            encoder = None
        
        if not structured_features:
            logging.warning("No structured features available, using dummy feature")
            df['dummy_feature'] = 0.0
            structured_features = ['dummy_feature']
        
        logging.info(f"Total structured features: {len(structured_features)}")
        
        # Generate model hash for caching
        model_hash = generate_model_hash(df, model_config, structured_features)
        logging.info(f"Generated model hash: {model_hash}")
        
        # Check if model already exists in cache
        cached_artifacts = load_model_artifacts(model_hash)
        
        if cached_artifacts is not None:
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
            
            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            logging.info(f"Starting training for {epochs} epochs")
            training_history = []
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                num_batches = 0
                
                for batch in train_loader:
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    structured = batch['structured'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids, attention_mask, structured)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                training_history.append({'epoch': epoch + 1, 'loss': avg_loss})
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Evaluation
            logging.info("Starting model evaluation")
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
