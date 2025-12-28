"""Dataset classes for tabular crop data."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular crop classification data."""
    
    def __init__(self, df, feature_cols, label_col, scaler=None, fit_scaler=False):
        """
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            label_col: Name of label column
            scaler: Pre-fitted StandardScaler (for test set)
            fit_scaler: Whether to fit a new scaler (for train set)
        """
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df[label_col].values.astype(np.int64)
        self.classes = np.unique(self.labels)
        
        if fit_scaler:
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)
        elif scaler is not None:
            self.scaler = scaler
            self.features = self.scaler.transform(self.features)
        else:
            self.scaler = None
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def load_data(data_path, label_col, test_size=0.2, random_state=42):
    """Load and split data into train/test DataFrames.
    
    Returns:
        train_df, test_df, feature_cols
    """
    df = pd.read_csv(data_path)
    feature_cols = [col for col in df.columns if col != label_col]
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        stratify=df[label_col],
        random_state=random_state
    )
    
    return train_df, test_df, feature_cols


def create_dataloaders(train_df, test_df, feature_cols, label_col, 
                       batch_size=64, num_workers=4):
    """Create train and test DataLoaders.
    
    Returns:
        train_loader, test_loader, train_dataset (for accessing scaler/classes)
    """
    train_dataset = TabularDataset(
        train_df, feature_cols, label_col, fit_scaler=True
    )
    
    test_dataset = TabularDataset(
        test_df, feature_cols, label_col, scaler=train_dataset.scaler
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, test_loader, train_dataset
