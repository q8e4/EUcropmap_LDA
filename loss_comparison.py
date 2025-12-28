"""Compare training losses between models."""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_loss_comparison(checkpoint_dir="checkpoints/minor", save_path="plots/minor/loss_comparison.png"):
    """Load metrics and plot loss comparison between NLLLoss and DNLLLoss."""
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # Load metrics
    with open(checkpoint_dir / "DeepLDA_DNLLLoss_metrics.json") as f:
        dnll_metrics = json.load(f)
    
    with open(checkpoint_dir / "DeepLDA_NLLLoss_metrics.json") as f:
        nll_metrics = json.load(f)
    
    # Setup plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    epochs = np.arange(1, len(dnll_metrics['train_losses']) + 1)
    
    # Plot losses
    plt.plot(epochs, nll_metrics['train_losses'], 
             linewidth=2.5, label='NLLLoss', color='#1f77b4', alpha=0.9)
    plt.plot(epochs, dnll_metrics['train_losses'], 
             linewidth=2.5, label='DNLLLoss', color='#ff7f0e', linestyle='--', alpha=0.9)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('NLLLoss vs DNLLLoss — Training Loss', fontsize=13)
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss comparison saved to {save_path}")
    
    plt.show()
    
    # Print final values
    print(f"\nFinal losses:")
    print(f"  NLLLoss:  {nll_metrics['train_losses'][-1]:.4f}")
    print(f"  DNLLLoss: {dnll_metrics['train_losses'][-1]:.4f}")


if __name__ == "__main__":
    plot_loss_comparison()