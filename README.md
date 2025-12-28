# Crop Classification with Deep LDA

Tabular crop classification comparing NLLLoss vs DNLLLoss.


## Project Structure

```
EUCROPMAP/
├── config.py              # Hyperparameters and paths
├── dataset.py             # TabularDataset class and data loading
├── models.py              # Encoder and DeepLDA architectures
├── trainer.py             # Training loop and evaluation
├── train.py               # Main training script
├── utils.py               # Visualization and helpers
├── loss_comparison.py     # Loss function comparison utilities
├── EUcropmapLDA.py        # Standalone LDA experiments
├── dataset_check.ipynb    # Dataset exploration notebook
├── src/
│   └── lda.py             # LDA heads and loss functions
├── datasets/              # Place CSV data here
├── checkpoints/           # Saved model weights
└── plots/                 # Generated figures
```

## Getting the Dataset

The dataset consists of tabular features extracted from EU Crop Map 2022 satellite imagery via Google Earth Engine.

### Option 1: Use Pre-extracted CSV (Recommended)

Place your `LU22_tabular.csv` file in the `datasets/` folder. The CSV should have:
- Feature columns (satellite-derived features)
- `Label_clas` column with integer class labels

### Option 2: Extract from Google Earth Engine

1. Set up a [Google Earth Engine](https://earthengine.google.com/) account
2. Open `dataset_check.ipynb` and authenticate with GEE
3. Run the export cells to download the tabular data to Google Drive
4. Move the exported CSV to `datasets/`

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
# Train all models
python train.py

# Train only one model
python train.py --model softmax
python train.py --model dnll
python train.py --model nll
```

### Custom Parameters

```bash
python train.py --epochs 50 --batch_size 128 --lr 0.001

# Use different data file
python train.py --data ./datasets/LU22_minor_tabular.csv

# Custom output directories
python train.py --plot_dir ./plots/experiment1 --checkpoint_dir ./checkpoints/exp1

# Skip plotting
python train.py --no-plot
```

### Full Example
To get the results for Minor dataset, I have used:

```bash
python train.py \
    --data ./datasets/LU22_minor_tabular.csv \
    --epochs 100 \
    --batch_size 64 \
    --plot_dir ./plots/minor \
    --checkpoint_dir ./checkpoints/minor
```

## Outputs

After training:

| File | Description |
|------|-------------|
| `checkpoints/DeepLDA_DNLLLoss_final.pt` | DNLLLoss model weights |
| `checkpoints/DeepLDA_NLLLoss_final.pt` | NLLLoss model weights |
| `checkpoints/*_metrics.json` | Training metrics history |
| `plots/accuracy_comparison.png` | NLLLoss vs DNLLLoss accuracy curves |
| `plots/confusion_matrix_*.png` | Confusion matrices |
| `plots/*_embeddings.png` | PCA visualization of learned embeddings |

## Module Usage

```python
from dataset import load_data, create_dataloaders
from models import create_model
from trainer import train
from src.lda import TrainableLDAHead, DNLLLoss

# Load data
train_df, test_df, feature_cols = load_data("datasets/LU22_tabular.csv", "Label_clas")
train_loader, test_loader, train_dataset = create_dataloaders(
    train_df, test_df, feature_cols, "Label_clas"
)

num_classes = len(train_dataset.classes)
input_dim = len(feature_cols)

# Create model with LDA head
lda_head = TrainableLDAHead(num_classes, num_classes)
model = create_model(
    num_classes=num_classes,
    input_dim=input_dim,
    lda_head=lda_head,
    device='cuda'
)

# Train
optimizer = torch.optim.Adam(model.encoder.parameters(), lr=1e-3)
loss_fn = DNLLLoss(lambda_reg=1.0)
history = train(model, train_loader, test_loader, loss_fn, optimizer, epochs=10)
```

## Configuration

Edit `config.py` to change defaults:

```python
@dataclass
class Config:
    data_path: str = "./datasets/LU22_tabular.csv"
    label_col: str = "Label_clas"
    test_size: float = 0.2
    hidden_dims: list = None  # Defaults to [512, 256, 128]
    dropout: float = 0.3
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    lambda_reg: float = 1.0  # DNLLLoss regularization
```


### Additional Dataset Information

In the EU‑CropMap‑2022 scheme, the “major” (Level‑1) layer contains seven broad land‑cover classes (Artificial land; Arable land; Woodland & Shrubland; Grassland; Bare land & lichens/moss; Water; and Wetlands). These 25 classes come from the Level‑2 layer, which expands the arable‑land class into specific crop‑type categories while retaining the other broad classes. These 25 Level‑2 classes, with their codes, are:

| EU CropMap Code    | Class name                                                         |
| ------- | ------------------------------------------------------------------ |
| **100** | Artificial land                                                    |
| **211** | Common wheat                                                       |
| **212** | Durum wheat                                                        |
| **213** | Barley                                                             |
| **214** | Rye                                                                |
| **215** | Oats                                                               |
| **216** | Maize                                                              |
| **217** | Rice                                                               |
| **218** | Triticale                                                          |
| **219** | Other cereals                                                      |
| **221** | Potatoes                                                           |
| **222** | Sugar beet                                                         |
| **223** | Other root crops                                                   |
| **230** | Other non‑permanent industrial crops (e.g., cotton, tobacco, hemp) |
| **231** | Sunflower                                                          |
| **232** | Rape and turnip rape                                               |
| **233** | Soya                                                               |
| **240** | Dry pulses, vegetables and flowers                                 |
| **250** | Fodder crops (cereals and pulses)                                  |
| **290** | Bare arable land                                                   |
| **300** | Woodland and Shrubland                                             |
| **500** | Grassland (pastures)                                               |
| **600** | Bare land and lichens/moss                                         |
| **700** | Water                                                              |
| **800** | Wetlands                                                           |


Thus, while the Level‑1 (major) layer has seven broad categories, the full Level‑2 classification includes those seven plus 18 crop‑specific classes (19 classes if you count bare arable land separately), giving a total of 25 classes in the detailed (minor) dataset.# EUcropmap_LDA
