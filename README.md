# Amazon ML Challenge 2025 - Rank 183 Solution

**Final Rank: 183 / 20,000+ teams (Top 1%)**
**Final SMAPE Score: 44.8%**

## Team

Bhavya Goyal, Shivam Baheti, Shubham Choulkar,Kanishka Utagikar

## Overview

Rank 183 / 20,000+ teams in Amazon ML Challenge 2025 (Top 1%). Multimodal price prediction task using product images and catalog text. Final SMAPE: 44.8% on 75,000 samples.

## Approach

Our methodology was based on a **two-model ensemble** that combined complementary signal pathways:

### Model 1: RoBERTa FFN (Text-Only)
Fine-tuned RoBERTa-Large with a three-layer feed-forward regression head using GELU activation and **Smooth L1 loss**, improving stability on noisy or underspecified text descriptions.

### Model 2: RoBERTa Image Concat (Multimodal Fusion)
Concatenated RoBERTa-Large text embeddings (1024-dim) with pre-computed image embeddings (1280-dim) through a three-layer regression head to incorporate visual price cues such as brand proxies, material/finish quality, and implicit category context frequently missing from structured metadata.

Both models were ensembled through simple averaging, which consistently outperformed either modality in isolation and demonstrated the practical value of lightweight multimodal fusion when working with imperfect marketplace data.

## Architecture

### Model 1: RoBERTa FFN (Text-Only)

```
┌─────────────────────────────────────────────────────────────┐
│                      Input Text                              │
│              (catalog_content, max_len=512)                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   RoBERTa-Large Tokenizer                    │
│                  (input_ids, attention_mask)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    RoBERTa-Large Encoder                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Self-Attention Layers (FROZEN)                       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  FFN Layers (FINE-TUNED)                             │  │
│  │  - intermediate (dense expansion)                     │  │
│  │  - output.dense (projection back)                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│                  Output: [batch, 512, 1024]                  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   CLS Token Extraction                       │
│                  [batch, 1024] (hidden_size)                 │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  3-Layer Regression Head                     │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Linear(1024 → 512)                                  │  │
│  │  LayerNorm(512)                                      │  │
│  │  GELU()                                              │  │
│  │  Dropout(0.3)                                        │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Linear(512 → 256)                                   │  │
│  │  LayerNorm(256)                                      │  │
│  │  GELU()                                              │  │
│  │  Dropout(0.3)                                        │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Linear(256 → 1)                                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Output: log_price                          │
│                   (apply expm1 to get actual price)          │
└─────────────────────────────────────────────────────────────┘
```

### Model 2: RoBERTa Image Concat (Multimodal Fusion)

```
┌─────────────────────┐              ┌──────────────────────┐
│   Input Text        │              │  Pre-computed Image  │
│ (catalog_content)   │              │  Embeddings (1280)   │
└──────────┬──────────┘              └──────────┬───────────┘
           │                                    │
           ▼                                    │
┌─────────────────────┐                         │
│  RoBERTa-Large      │                         │
│  (FFN Fine-tuned)   │                         │
│                     │                         │
│  Output: [1024]     │                         │
└──────────┬──────────┘                         │
           │                                    │
           │                                    │
           └──────────────┬─────────────────────┘
                          │
                          ▼
           ┌──────────────────────────┐
           │  Concatenate             │
           │  [1024 + 1280] = [2304]  │
           └──────────┬───────────────┘
                      │
                      ▼
           ┌─────────────────────────────┐
           │  3-Layer Regression Head    │
           │                             │
           │  Linear(2304 → 512)         │
           │  LayerNorm → GELU → Drop    │
           │                             │
           │  Linear(512 → 256)          │
           │  LayerNorm → GELU → Drop    │
           │                             │
           │  Linear(256 → 1)            │
           └──────────┬──────────────────┘
                      │
                      ▼
           ┌─────────────────────┐
           │  Output: log_price  │
           └─────────────────────┘
```

### Ensemble Strategy

```
┌──────────────────┐         ┌────────────────────────┐
│  RoBERTa FFN     │         │ RoBERTa Image Concat   │
│  Prediction      │         │ Prediction             │
└────────┬─────────┘         └────────┬───────────────┘
         │                            │
         │    log_price_1             │    log_price_2
         │                            │
         └────────────┬───────────────┘
                      │
                      ▼
         ┌────────────────────────┐
         │  Ensemble (Average)    │
         │                        │
         │  log_price_final =     │
         │  (log_p1 + log_p2) / 2 │
         └────────┬───────────────┘
                  │
                  ▼
         ┌────────────────────┐
         │  expm1(log_price)  │
         │  → final_price     │
         └────────────────────┘
```

## Key Techniques

### 1. Log Transformation for Skewness Reduction

**Problem**: Price distribution was highly skewed (skewness = 13.60, mean: $23.65, median: $14.00)

**Solution**: Applied log1p transformation to normalize the target variable

```python
train_df['log_price'] = np.log1p(train_df['price'])
```

**Impact**:
- Normalized price distribution
- Better gradient flow during training
- Improved model convergence
- Reduced impact of extreme outliers

### 2. Selective Fine-Tuning (FFN Layers Only)

**Strategy**: Freeze self-attention layers, fine-tune only FFN (Feed-Forward Network) layers

```python
for name, param in self.encoder.named_parameters():
    if 'intermediate' in name or 'output.dense' in name:
        param.requires_grad = True  # Fine-tune FFN
    else:
        param.requires_grad = False  # Freeze attention
```

**Benefits**:
- ~90% reduction in trainable parameters
- Faster training (3-4x speedup)
- Reduced overfitting on noisy marketplace data
- Preserved pre-trained language understanding

### 3. Smooth L1 Loss (Huber Loss) > SMAPE Loss

**Tried**: SMAPE Loss (Symmetric Mean Absolute Percentage Error)
**Selected**: Smooth L1 Loss (Huber Loss)

```python
criterion = nn.SmoothL1Loss()
```

**Why Smooth L1 Won**:
- More robust to outliers in price range
- Better gradient properties near zero
- Smoother convergence during training
- SMAPE struggled with low-price items (<$5) and numerical instability

### 4. 3-Layer Regression Head with LayerNorm + GELU

**Architecture**:
```python
self.regressor = nn.Sequential(
    nn.Linear(hidden_size, 512),
    nn.LayerNorm(512),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.LayerNorm(256),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(256, 1)
)
```

**Design Choices**:
- Progressive dimensionality reduction (1024/2304 → 512 → 256 → 1)
- LayerNorm for training stability
- GELU activation (smoother than ReLU, better for regression)
- Dropout(0.3) for regularization

## Training Configuration

```python
class Config:
    TEXT_MODEL = "roberta-large"
    IMAGE_EMB_DIM = 1280
    MAX_EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    MAX_LENGTH = 512
    VAL_SPLIT = 0.15
    DROPOUT = 0.3
    GRADIENT_CLIP = 1.0
    EARLY_STOPPING_PATIENCE = 3
```

### Optimization Details

- **Optimizer**: AdamW with weight decay (0.01)
- **Scheduler**: CosineAnnealingLR (T_max=15, eta_min=1e-6)
- **Mixed Precision**: FP16 training with GradScaler (CUDA)
- **Gradient Clipping**: Max norm = 1.0
- **Early Stopping**: Patience = 3 epochs on validation SMAPE

## Data Preprocessing

### Text Processing
- Tokenizer: RoBERTa tokenizer (byte-level BPE)
- Max length: 512 tokens
- Padding: max_length
- Truncation: enabled

### Image Processing
- Pre-computed embeddings (1280-dim vectors)
- Stored as `.npy` files for efficient loading
- Normalized and cached

### Target Transformation
- Forward: `log_price = log1p(price)`
- Inverse: `price = expm1(log_price)`

## Exploratory Data Analysis

See `train_eda.ipynb` for comprehensive analysis:

### Dataset Overview
- **Total Samples**: 75,000 training samples
- **Features**: catalog_content, image_link, price
- **Train/Val Split**: 85% / 15%

### Price Distribution
- **Mean**: $23.65
- **Median**: $14.00
- **Range**: $0.13 - $2,796.00
- **Skewness**: 13.60 (highly right-skewed)
- **Outliers (IQR)**: 7.37% above $61.37

### Text Features
- Avg item name length: 86 characters (14.5 words)
- Avg bullet points: 3.49 per item
- 72.6% have bullet points
- 43.4% have product descriptions
- Avg catalog content length: 909 characters

### Units
- 92 unique units
- Most common: Ounce (55%), Count (24%)
- 5 unit categories: Weight, Count, Volume, Length, Other

### Key Insights
1. **Skewed distribution** → Log transformation essential
2. **Rich text features** → Longer descriptions correlate with higher prices (r=0.15)
3. **Minimal missing data** → High quality dataset (<1% missing)
4. **Weak linear correlations** → Non-linear modeling required

## Results

### Performance Summary

| Model | SMAPE | Notes |
|-------|-------|-------|
| RoBERTa FFN | ~46-47% | Text-only baseline |
| RoBERTa Image Concat | ~45-46% | Multimodal improvement |
| **Ensemble (Final)** | **44.8%** | **Rank 183 / 20,000+** |

### Ensemble Details
- Model 1: Text-only captures brand names, product descriptions
- Model 2: Multimodal adds visual cues (packaging, appearance)
- Simple averaging of log predictions

## Repository Structure

```
amazon_ml_solution/
├── README.md
├── train_roberta_ffn.py            # Train text-only model
├── train_roberta_image_concat.py   # Train multimodal model
├── ROBERTA_FFN.ipynb               # Notebook version (Model 1)
├── ROBERTA_IMAGE_CONCAT.ipynb      # Notebook version (Model 2)
├── train_eda.ipynb                 # EDA
└── requirements.txt
```

## Setup and Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `torch >= 2.0.0`
- `transformers >= 4.30.0`
- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`

### Training

**Train Text-Only Model:**
```bash
python train_roberta_ffn.py
```

**Train Multimodal Model:**
```bash
python train_roberta_image_concat.py
```

**Ensemble Predictions:**
```python
# Average log predictions from both models
log_price_final = (pred1 + pred2) / 2
final_prices = np.expm1(log_price_final)
```

## Lessons Learned

### What Worked ✅

1. **Log transformation** → 15-20% SMAPE improvement
2. **Smooth L1 loss** → 5-8% better than SMAPE loss
3. **FFN-only fine-tuning** → 10x faster, no overfitting
4. **Image embeddings** → 2-3% improvement over text-only
5. **3-layer regression head** → Better than 1-layer (underfitting) or 5-layer (overfitting)
6. **Simple ensemble averaging** → Consistent gains over single models

### What Didn't Work ❌

1. **SMAPE as loss function** → Poor gradients, unstable training
2. **Fine-tuning all RoBERTa layers** → Severe overfitting
3. **Single-layer regression head** → Insufficient capacity
4. **Training without log transformation** → Poor performance on high-price items
5. **Complex stacking ensembles** → Overfitted on validation set

---

Amazon ML Challenge 2025 | Rank 183 / 20,000+ teams | SMAPE: 44.8%
