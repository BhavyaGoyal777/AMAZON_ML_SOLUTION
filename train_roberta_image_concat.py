import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Configuration
class Config:
    DATA_DIR = "DATA/student_resource/dataset"
    TRAIN_IMAGE_EMB = "train_image_emb.npy"
    TEST_IMAGE_EMB = "test_image_emb.npy"

    TEXT_MODEL = "roberta-large"
    MAX_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_LENGTH = 512
    VAL_SPLIT = 0.15
    RANDOM_STATE = 42
    NUM_WORKERS = 4
    EARLY_STOPPING_PATIENCE = 3
    DROPOUT = 0.3
    GRADIENT_CLIP = 1.0
    DEVICE = device

config = Config()

# Load data
train_df = pd.read_csv(os.path.join(config.DATA_DIR, 'train.csv'))
test_df = pd.read_csv(os.path.join(config.DATA_DIR, 'test.csv'))
train_df['log_price'] = np.log1p(train_df['price'])

train_data, val_data = train_test_split(train_df, test_size=config.VAL_SPLIT, random_state=config.RANDOM_STATE)
train_data = train_data.reset_index(drop=True)
val_data = val_data.reset_index(drop=True)

print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_df)}")

# Load image embeddings
full_train_image_emb = np.load(config.TRAIN_IMAGE_EMB)
test_image_emb = np.load(config.TEST_IMAGE_EMB)

print(f"Train image emb: {full_train_image_emb.shape}")
print(f"Test image emb: {test_image_emb.shape}")

# Split image embeddings
train_idx = train_data.index.values
val_idx = val_data.index.values

train_image_emb = full_train_image_emb[train_idx]
val_image_emb = full_train_image_emb[val_idx]

print(f"Train image split: {train_image_emb.shape}")
print(f"Val image split: {val_image_emb.shape}")

# Dataset
class FusionDataset(Dataset):
    def __init__(self, df, image_emb, tokenizer, config, is_test=False):
        self.df = df
        self.image_emb = image_emb
        self.tokenizer = tokenizer
        self.config = config
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoding = self.tokenizer(
            row['catalog_content'],
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'image_emb': torch.from_numpy(self.image_emb[idx]).float()
        }
        if not self.is_test:
            item['label'] = torch.FloatTensor([row['log_price']])
        return item

# Model
class RoBERTaImageConcat(nn.Module):
    def __init__(self, model_name, image_dim=1280, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze all except FFN
        for name, param in self.encoder.named_parameters():
            if 'intermediate' in name or 'output.dense' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        text_dim = self.encoder.config.hidden_size

        # 3-layer regression head: concat(1024, 1280) = 2304
        self.regressor = nn.Sequential(
            nn.Linear(text_dim + image_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask, image_emb):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_hidden = outputs.last_hidden_state[:, 0, :]
        fused = torch.cat([text_hidden, image_emb], dim=-1)
        return self.regressor(fused).squeeze(-1)

# Prepare data
tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL)

train_dataset = FusionDataset(train_data, train_image_emb, tokenizer, config)
val_dataset = FusionDataset(val_data, val_image_emb, tokenizer, config)
test_dataset = FusionDataset(test_df, test_image_emb, tokenizer, config, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE*2, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE*2, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

print(f"Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

# Initialize model
model = RoBERTaImageConcat(config.TEXT_MODEL, image_dim=1280, dropout=config.DROPOUT).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total: {total_params:,} | Trainable: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

# Training setup
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.MAX_EPOCHS, eta_min=1e-6)
criterion = nn.SmoothL1Loss()
scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

def calculate_smape(predictions, actuals):
    predictions = np.expm1(predictions)
    actuals = np.expm1(actuals)
    numerator = np.abs(predictions - actuals)
    denominator = (np.abs(actuals) + np.abs(predictions)) / 2
    return np.mean(numerator / np.maximum(denominator, 1e-8)) * 100

def train_epoch(model, train_loader, optimizer, criterion, scaler, config):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_emb = batch['image_emb'].to(device)
        labels = batch['label'].to(device).squeeze()

        if scaler:
            with torch.cuda.amp.autocast():
                preds = model(input_ids, attention_mask, image_emb)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            preds = model(input_ids, attention_mask, image_emb)
            loss = criterion(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()

    return total_loss / len(train_loader)

@torch.no_grad()
def validate(model, val_loader, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0

    for batch in tqdm(val_loader, desc="Validation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_emb = batch['image_emb'].to(device)
        labels = batch['label'].to(device).squeeze()

        preds = model(input_ids, attention_mask, image_emb)
        loss = criterion(preds, labels)

        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    smape = calculate_smape(np.array(all_preds), np.array(all_labels))

    return avg_loss, smape

# Training loop
best_smape = float('inf')
patience_counter = 0

for epoch in range(config.MAX_EPOCHS):
    print(f"\nEpoch {epoch+1}/{config.MAX_EPOCHS} | LR: {optimizer.param_groups[0]['lr']:.2e}")

    train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, config)
    val_loss, val_smape = validate(model, val_loader, criterion)
    scheduler.step()

    print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | SMAPE: {val_smape:.2f}%")

    if val_smape < best_smape:
        best_smape = val_smape
        torch.save(model.state_dict(), 'best_roberta_image_concat.pth')
        print(f"✓ Best: {val_smape:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping")
            break

print(f"\n✓ Best SMAPE: {best_smape:.2f}%")

# Inference
model.load_state_dict(torch.load('best_roberta_image_concat.pth', map_location=device, weights_only=True))
model.eval()

all_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        image_emb = batch['image_emb'].to(device)
        preds = model(input_ids, attention_mask, image_emb)
        all_preds.extend(preds.cpu().numpy())

test_prices = np.expm1(np.array(all_preds))
submission = pd.DataFrame({'sample_id': test_df['sample_id'], 'price': test_prices})
submission.to_csv('submission_roberta_image_concat.csv', index=False)

print(f"\n✓ Saved: submission_roberta_image_concat.csv")
print(f"✓ SMAPE: {best_smape:.2f}%")
print(f"✓ Range: ${submission['price'].min():.2f} - ${submission['price'].max():.2f}")
