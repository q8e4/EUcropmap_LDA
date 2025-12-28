"""Configuration and hyperparameters."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    data_path: str = "./datasets/LU22_tabular.csv"
    checkpoint_dir: Path = Path("checkpoints")
    plot_dir: Path = Path("plots")
    
    # Data
    label_col: str = "Label_clas"
    test_size: float = 0.2
    random_state: int = 42
    
    # Model
    hidden_dims: list = None  # Will default to [512, 256, 128]
    dropout: float = 0.3
    
    # Training
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3
    num_workers: int = 4
    
    # LDA Loss
    lambda_reg: float = 1.0
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)


# Default config instance
cfg = Config()
