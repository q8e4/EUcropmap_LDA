import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# 1. Custom Dataset
# ============================================================================
class DFDataset(Dataset):
    def __init__(self, df, feature_cols, label_col, scaler=None, fit_scaler=False):
        self.features = df[feature_cols].values.astype(np.float32)
        self.labels = df[label_col].values.astype(np.int64)
        
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

# ============================================================================
# 2. MLP Model
# ============================================================================
class MLP(nn.Module):
    def __init__(self, input_dim=184, num_classes=25):
        super(MLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)

# ============================================================================
# 3. Training and Evaluation Functions
# ============================================================================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, all_preds, all_labels

# ============================================================================
# 4. Plotting Functions
# ============================================================================
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', marker='o')
    ax1.plot(val_losses, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy', marker='o')
    ax2.plot(val_accs, label='Val Accuracy', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved training curves to 'training_curves.png'")

def plot_confusion_matrix(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes), 
                yticklabels=range(num_classes))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved confusion matrix to 'confusion_matrix.png'")

# ============================================================================
# MAIN SCRIPT
# ============================================================================
if __name__ == "__main__":
    df = pd.read_csv('train_tabular.csv') 
    # Define feature and label columns
    feature_cols = [col for col in df.columns if col != 'Label_clas']
    label_col = 'Label_clas'
    
    # Hyperparameters
    input_dim = len(feature_cols)
    num_classes = df[label_col].nunique()
    batch_size = 128
    num_epochs = 100
    learning_rate = 0.001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Input dim: {input_dim}, Num classes: {num_classes}")
    
    # ========================================================================
    # Data Splitting
    # ========================================================================
    train_val_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df[label_col],
        random_state=42
    )
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.25,
        stratify=train_val_df[label_col],
        random_state=42
    )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # ========================================================================
    # Create Datasets and Dataloaders
    # ========================================================================
    train_dataset = DFDataset(train_df, feature_cols, label_col, fit_scaler=True)
    val_dataset = DFDataset(val_df, feature_cols, label_col, scaler=train_dataset.scaler)
    test_dataset = DFDataset(test_df, feature_cols, label_col, scaler=train_dataset.scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # ========================================================================
    # Initialize Model, Loss, and Optimizer
    # ========================================================================
    model = MLP(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # ========================================================================
    # Save Final Model
    # ========================================================================
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': train_dataset.scaler,
        'input_dim': input_dim,
        'num_classes': num_classes
    }, 'final_mlp_model.pth')
    print("\n✓ Saved final model to 'final_mlp_model.pth'")
    
    # ========================================================================
    # Plot Training Curves
    # ========================================================================
    print("\n" + "="*70)
    print("PLOTTING TRAINING CURVES")
    print("="*70)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    # ========================================================================
    # Test Evaluation
    # ========================================================================
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # ========================================================================
    # Classification Report
    # ========================================================================
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    
    report = classification_report(
        test_labels, 
        test_preds, 
        target_names=[f'Class {i}' for i in range(num_classes)],
        digits=4
    )
    print(report)
    
    # Save classification report
    with open('classification_report.txt', 'w') as f:
        f.write(report)
    print("\n✓ Saved classification report to 'classification_report.txt'")
    
    # ========================================================================
    # Confusion Matrix
    # ========================================================================
    print("\n" + "="*70)
    print("CONFUSION MATRIX")
    print("="*70)
    plot_confusion_matrix(test_labels, test_preds, num_classes)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("\nGenerated files:")
    print("  - final_mlp_model.pth")
    print("  - training_curves.png")
    print("  - confusion_matrix.png")
    print("  - classification_report.txt")