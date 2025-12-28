"""Model architectures for crop classification with LDA."""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP encoder for tabular data."""
    
    def __init__(self, input_dim, output_dim, hidden_dims=None, dropout=0.3):
        """
        Args:
            input_dim: Number of input features
            output_dim: Output embedding dimension (usually num_classes)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class DeepLDA(nn.Module):
    """Deep LDA model: Encoder + LDA head."""
    
    def __init__(self, num_classes, input_dim, latent_dim=None, 
                 hidden_dims=None, dropout=0.3, lda_head=None):
        """
        Args:
            num_classes: Number of output classes
            input_dim: Number of input features
            latent_dim: Dimension of encoder output (defaults to num_classes)
            hidden_dims: Hidden layer dimensions for encoder
            dropout: Dropout probability
            lda_head: LDA head module (TrainableLDAHead)
        """
        super().__init__()
        
        if latent_dim is None:
            latent_dim = num_classes
            
        self.encoder = Encoder(
            input_dim=input_dim, 
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        self.head = lda_head
        
    def forward(self, x):
        z = self.encoder(x)
        if self.head is not None:
            return self.head(z)
        return z
    
    def get_embeddings(self, x):
        """Get encoder embeddings without LDA head."""
        return self.encoder(x)


def create_model(num_classes, input_dim, lda_head=None, hidden_dims=None, 
                 dropout=0.3, device='cpu'):
    """Factory function to create DeepLDA model.
    
    Args:
        num_classes: Number of classes
        input_dim: Number of input features
        lda_head: LDA head module
        hidden_dims: Hidden layer dimensions
        dropout: Dropout probability
        device: Device to place model on
        
    Returns:
        DeepLDA model on specified device
    """
    model = DeepLDA(
        num_classes=num_classes,
        input_dim=input_dim,
        latent_dim=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lda_head=lda_head
    )
    return model.to(device)
