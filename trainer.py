"""Training and evaluation utilities."""

import json
import torch
from pathlib import Path
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model accuracy on a dataloader.
    
    Args:
        model: PyTorch model
        loader: DataLoader to evaluate on
        device: Device to run evaluation on
        
    Returns:
        Accuracy as float
    """
    model.eval()
    correct = total = 0
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    
    return correct / total


def train_epoch(model, train_loader, loss_fn, optimizer, device):
    """Train for one epoch.
    
    Returns:
        dict with 'loss' and 'accuracy'
    """
    model.train()
    loss_sum = acc_sum = n_sum = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        
        logits = model(x)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred = logits.argmax(1)
            acc_sum += (pred == y).sum().item()
            n_sum += y.size(0)
            loss_sum += loss.item() * y.size(0)
    
    return {
        'loss': loss_sum / n_sum,
        'accuracy': acc_sum / n_sum
    }


def train(model, train_loader, test_loader, loss_fn, optimizer, 
          epochs, device, checkpoint_dir=None, model_name="model"):
    """Full training loop.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        test_loader: Validation DataLoader
        loss_fn: Loss function
        optimizer: Optimizer
        epochs: Number of epochs
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        model_name: Name prefix for saved files
        
    Returns:
        dict with training history
    """
    history = {
        'train_acc': [],
        'val_acc': [],
        'train_loss': []
    }
    best_val_acc = 0.0
    
    pbar = tqdm(range(epochs), desc='Training')
    for epoch in pbar:
        # Train
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device)
        
        # Evaluate
        val_acc = evaluate(model, test_loader, device)
        
        # Store history
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_acc)
        history['train_loss'].append(train_metrics['loss'])
        
        # Update best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        # Update progress bar
        pbar.set_postfix({
            'Epoch': f'{epoch:02d}',
            'Train_Acc': f'{train_metrics["accuracy"]:.4f}',
            'Val_Acc': f'{val_acc:.4f}',
            'Train_Loss': f'{train_metrics["loss"]:.4f}',
        })
    
    # Save checkpoint
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accs': history['train_acc'],
            'val_accs': history['val_acc'],
            'train_losses': history['train_loss'],
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': best_val_acc,
        }, checkpoint_dir / f'{model_name}_final.pt')
        
        # Save metrics as JSON
        metrics = {
            'train_accuracies': history['train_acc'],
            'val_accuracies': history['val_acc'],
            'train_losses': history['train_loss'],
            'final_train_acc': history['train_acc'][-1],
            'final_val_acc': history['val_acc'][-1],
            'best_val_acc': best_val_acc,
        }
        with open(checkpoint_dir / f'{model_name}_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return history
