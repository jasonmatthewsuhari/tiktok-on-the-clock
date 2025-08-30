import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# --------------------------
# 1. Load Data
# --------------------------
df = pd.read_csv("data.csv")  # your CSV file
df = df.dropna(subset=['text', 'label'])  # ensure no missing text/label

# Binary encode label
df['label'] = df['label'].map({'valid':0, 'invalid':1})

# --------------------------
# 2. Preprocess numeric/categorical features
# --------------------------
numeric_features = ['rating', 'avg_rating', 'num_of_reviews', 'pics']
categorical_features = ['category']

# Numeric scaler
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# Categorical one-hot
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
cat_encoded = encoder.fit_transform(df[categorical_features])
cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(categorical_features))
df = pd.concat([df.reset_index(drop=True), cat_encoded_df.reset_index(drop=True)], axis=1)

# Combine all structured features
structured_features = numeric_features + list(cat_encoded_df.columns)

# --------------------------
# 3. BERT tokenizer
# --------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
MAX_LEN = 128

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

# --------------------------
# 4. Train/Test Split
# --------------------------
X_text = df['text'].values
X_struct = df[structured_features].values
y = df['label'].values

X_text_train, X_text_val, X_struct_train, X_struct_val, y_train, y_val = train_test_split(
    X_text, X_struct, y, test_size=0.2, random_state=42
)

train_dataset = ReviewDataset(X_text_train, X_struct_train, y_train, tokenizer, MAX_LEN)
val_dataset = ReviewDataset(X_text_val, X_struct_val, y_val, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# --------------------------
# 5. Model Definition
# --------------------------
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

model = BertWithStructured(n_structured=len(structured_features))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# --------------------------
# 6. Training Setup
# --------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# --------------------------
# 7. Training Loop
# --------------------------
EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
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
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

# --------------------------
# 8. Evaluation
# --------------------------
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        structured = batch['structured'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask, structured)
        preds = torch.argmax(outputs, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.metrics import classification_report
print(classification_report(all_labels, all_preds))
