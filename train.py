"""
Main training script for Crop Classification with Deep LDA.

Usage:
    python train.py                    # Train all models
    python train.py --model dnll       # Train only DNLLLoss model
    python train.py --model nll        # Train only NLLLoss model
    python train.py --model softmax    # Train only Softmax (CrossEntropy) model
    python train.py --epochs 20        # Custom epochs
"""

import argparse
import torch
import torch.nn as nn

from pathlib import Path

from config import cfg
from dataset import load_data, create_dataloaders
from models import create_model
from trainer import train
from utils import get_device, plot_training_comparison, plot_embeddings, plot_confusion_matrix
from src.lda import TrainableLDAHead, DNLLLoss

def parse_args():
    parser = argparse.ArgumentParser(description='Train Deep LDA for crop classification')
    parser.add_argument('--model', type=str, default='all', choices=['all', 'dnll', 'nll', 'softmax'],
                        help='Which model to train: dnll, nll, softmax, or all')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to training CSV')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip plotting')
    parser.add_argument('--plot_dir', type=str, default=None,
                        help='Directory for saving plots')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory for saving checkpoints')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.plot_dir:
        cfg.plot_dir = Path(args.plot_dir)
        cfg.plot_dir.mkdir(exist_ok=True)
    
    if args.checkpoint_dir:
        cfg.checkpoint_dir = Path(args.checkpoint_dir)
        cfg.checkpoint_dir.mkdir(exist_ok=True)
    
    # Override config with command line args
    epochs = args.epochs or cfg.epochs
    batch_size = args.batch_size or cfg.batch_size
    lr = args.lr or cfg.learning_rate
    data_path = args.data or cfg.data_path
    
    # Setup
    device = get_device()
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    train_df, test_df, feature_cols = load_data(
        data_path, 
        cfg.label_col, 
        test_size=cfg.test_size,
        random_state=cfg.random_state
    )
    
    train_loader, test_loader, train_dataset = create_dataloaders(
        train_df, test_df, feature_cols, cfg.label_col,
        batch_size=batch_size, 
        num_workers=cfg.num_workers
    )
    
    num_classes = len(train_dataset.classes)
    input_dim = len(feature_cols)
    
    print(f"Train: {len(train_dataset)}, Val: {len(test_df)}")
    print(f"Classes: {num_classes}, Features: {input_dim}")
    
    history_dnll = None
    history_nll = None
    history_softmax = None
    
    # Train DNLLLoss model
    if args.model in ['all', 'dnll']:
        print("\n" + "=" * 50)
        print("Training with DNLLLoss")
        print("=" * 50)
        
        lda_head = TrainableLDAHead(num_classes, num_classes)
        model_dnll = create_model(
            num_classes=num_classes,
            input_dim=input_dim,
            lda_head=lda_head,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            device=device
        )
        
        optimizer = torch.optim.Adam(model_dnll.encoder.parameters(), lr=lr)
        loss_fn = DNLLLoss(lambda_reg=cfg.lambda_reg)
        
        history_dnll = train(
            model_dnll, train_loader, test_loader, 
            loss_fn, optimizer, epochs, device,
            checkpoint_dir=cfg.checkpoint_dir,
            model_name="DeepLDA_DNLLLoss"
        )
        
        plot_confusion_matrix(
            model_dnll, 
            test_loader, 
            train_dataset.classes, 
            device,
            save_path=cfg.plot_dir / "confusion_matrix_dnll.png"
        )

        # Plot embeddings
        if not args.no_plot:
            plot_embeddings(
                model_dnll, train_loader, num_classes, 
                train_dataset.classes, device,
                save_path=cfg.plot_dir / "DNLLLoss_embeddings.png"
            )
    
    # Train NLLLoss model
    if args.model in ['all', 'nll']:
        print("\n" + "=" * 50)
        print("Training with NLLLoss")
        print("=" * 50)
        
        lda_head = TrainableLDAHead(num_classes, num_classes)
        model_nll = create_model(
            num_classes=num_classes,
            input_dim=input_dim,
            lda_head=lda_head,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            device=device
        )
        
        optimizer = torch.optim.Adam(model_nll.encoder.parameters(), lr=lr)
        loss_fn = nn.NLLLoss()
        
        history_nll = train(
            model_nll, train_loader, test_loader,
            loss_fn, optimizer, epochs, device,
            checkpoint_dir=cfg.checkpoint_dir,
            model_name="DeepLDA_NLLLoss"
        )
        
        plot_confusion_matrix(
            model_nll, 
            test_loader, 
            train_dataset.classes, 
            device,
            save_path=cfg.plot_dir / "confusion_matrix_nll.png"
        )

        # Plot embeddings
        if not args.no_plot:
            plot_embeddings(
                model_nll, train_loader, num_classes,
                train_dataset.classes, device,
                save_path=cfg.plot_dir / "NLLLoss_embeddings.png"
            )
    
    # Train Softmax (CrossEntropy) model
    if args.model in ['all', 'softmax']:
        print("\n" + "=" * 50)
        print("Training with Softmax (CrossEntropyLoss)")
        print("=" * 50)
        
        # No LDA head for softmax - just encoder + linear output
        model_softmax = create_model(
            num_classes=num_classes,
            input_dim=input_dim,
            lda_head=None,  # No LDA head
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            device=device
        )
        
        optimizer = torch.optim.Adam(model_softmax.encoder.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        
        history_softmax = train(
            model_softmax, train_loader, test_loader,
            loss_fn, optimizer, epochs, device,
            checkpoint_dir=cfg.checkpoint_dir,
            model_name="Softmax_CrossEntropy"
        )
        
        plot_confusion_matrix(
            model_softmax, 
            test_loader, 
            train_dataset.classes, 
            device,
            save_path=cfg.plot_dir / "confusion_matrix_softmax.png"
        )

        # Plot embeddings
        if not args.no_plot:
            plot_embeddings(
                model_softmax, train_loader, num_classes,
                train_dataset.classes, device,
                save_path=cfg.plot_dir / "Softmax_embeddings.png"
            )
    
    # Comparison plot (only if we have at least 2 histories)
    if not args.no_plot:
        num_trained = sum(h is not None for h in [history_nll, history_dnll, history_softmax])
        if num_trained >= 2:
            plot_training_comparison(
                history_nll=history_nll,
                history_dnll=history_dnll,
                history_softmax=history_softmax,
                save_path=cfg.plot_dir / "accuracy_comparison.png"
            )

    
# After training

if __name__ == "__main__":
    main()