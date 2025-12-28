"""Utility functions for visualization and analysis."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# utils.py - add this function


def plot_confusion_matrix(model, dataloader, classes, device, 
                          save_path=None, normalize=True):
    """Plot confusion matrix for a trained model.
    
    Args:
        model: Trained model
        dataloader: DataLoader to evaluate on
        classes: Array of class names/labels
        device: Device to run inference on
        save_path: Path to save figure
        normalize: Whether to normalize by true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    
    return cm

def plot_embeddings(model, dataloader, num_classes, classes, device, 
                    save_path=None, max_batches=10):
    """Plot 2D PCA projection of encoder embeddings.
    
    Args:
        model: Trained model with encoder
        dataloader: DataLoader to extract embeddings from
        num_classes: Number of classes
        classes: Array of class labels
        device: Device to run inference on
        save_path: Path to save figure
        max_batches: Maximum batches to use
    """
    model.eval()
    embeds, labels = [], []
    
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            z = model.encoder(x).cpu()
            embeds.append(z)
            labels.append(y)
            if i >= max_batches - 1:
                break
    
    z = torch.cat(embeds)
    y = torch.cat(labels)
    
    # 2D projection (PCA)
    z0 = z - z.mean(0, keepdim=True)
    U, S, V = torch.pca_lowrank(z0, q=2)
    z2 = z0 @ V[:, :2]
    
    # Plot settings
    plt.rcParams.update({
        "font.size": 14, 
        "axes.labelsize": 16, 
        "legend.fontsize": 13,
        "xtick.labelsize": 13, 
        "ytick.labelsize": 13
    })
    
    plt.figure(figsize=(8, 6))
    for c in range(num_classes):
        idx = y == c
        plt.scatter(z2[idx, 0], z2[idx, 1], s=8, alpha=0.6, label=classes[c])
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        print(f"Embeddings plot saved to {save_path}")
    
    plt.show()


def plot_training_comparison(history_nll, history_dnll, save_path=None):
    """Plot training curves comparing NLLLoss vs DNLLLoss.
    
    Args:
        history_nll: Dict with 'train_acc', 'val_acc' for NLLLoss training
        history_dnll: Dict with 'train_acc', 'val_acc' for DNLLLoss training
        save_path: Path to save figure
    """
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams['figure.facecolor'] = 'white'
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_nl = np.arange(1, len(history_nll['train_acc']) + 1)
    epochs_lda = np.arange(1, len(history_dnll['train_acc']) + 1)
    
    NLL_color = '#1f77b4'   # Blue
    DNLL_color = '#ff7f0e'  # Orange
    
    # Plot 1: Training Accuracies
    ax1 = axes[0]
    ax1.plot(epochs_nl, history_nll['train_acc'], linewidth=2.5, 
             label='NLLLoss', color=NLL_color, alpha=0.9)
    ax1.plot(epochs_lda, history_dnll['train_acc'], linewidth=2.5, 
             label='DNLLLoss', color=DNLL_color, linestyle='--', alpha=0.9)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('NLLLoss vs DNLLLoss — Train', fontsize=13, fontweight='normal')
    ax1.legend(loc='lower right', frameon=True, fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Plot 2: Validation Accuracies
    ax2 = axes[1]
    ax2.plot(epochs_nl, history_nll['val_acc'], linewidth=2.5, 
             label='NLLLoss', color=NLL_color, alpha=0.9)
    ax2.plot(epochs_lda, history_dnll['val_acc'], linewidth=2.5, 
             label='DNLLLoss', color=DNLL_color, linestyle='--', alpha=0.9)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('NLLLoss vs DNLLLoss — Test', fontsize=13, fontweight='normal')
    ax2.legend(loc='lower right', frameon=True, fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()
    
    # Print results
    print("\n" + "=" * 50)
    print("Final Results:")
    print("=" * 50)
    print(f"NLLLoss  - Train: {history_nll['train_acc'][-1]:.4f}, Val: {history_nll['val_acc'][-1]:.4f}")
    print(f"DNLLLoss - Train: {history_dnll['train_acc'][-1]:.4f}, Val: {history_dnll['val_acc'][-1]:.4f}")


def get_device():
    """Get available device (CUDA if available, else CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    return device
