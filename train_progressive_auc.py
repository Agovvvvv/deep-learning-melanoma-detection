"""
Progressive Training Function with AUC-based Model Selection

This version uses validation AUC-ROC instead of validation loss for selecting
the best model. AUC is more appropriate for medical binary classification as it:
- Is threshold-independent
- Handles class imbalance better
- Is the standard metric in medical AI research
"""

import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score


def train_progressive(model, train_loader, test_loader, criterion, device, 
                      train_one_epoch, validate):
    """
    Three-phase progressive unfreezing with AUC-based model selection.
    
    Phase 1 (12 epochs): Train classifier only
    Phase 2 (15 epochs): Unfreeze last 3 blocks  
    Phase 3 (15 epochs): Unfreeze all layers
    
    MODEL SELECTION: Uses validation AUC (higher is better)
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for validation data
        criterion: Loss function
        device: torch.device (cuda or cpu)
        train_one_epoch: Function to train for one epoch
        validate: Function to validate the model
    
    Returns:
        model: Trained model
        history: Dictionary with training metrics including AUC
    """
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auc': []  # Added AUC tracking
    }
    
    best_val_auc = 0.0  # Changed from loss to AUC (maximize)
    best_model_path = 'best_melanoma_improved.pth'
    
    # ========== PHASE 1: CLASSIFIER ONLY ==========
    print("\n" + "🔥 PHASE 1: Training Classifier (Backbone Frozen)")
    print("-" * 60)
    
    # Optimizer for classifier only
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.001,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    for epoch in range(12):
        print(f'\nPhase 1 - Epoch {epoch+1}/12')
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate and calculate AUC
        val_loss, val_acc, _, val_labels, val_probs = validate(
            model, test_loader, criterion, device
        )
        
        # Calculate AUC-ROC
        val_auc = roc_auc_score(val_labels, val_probs)
        
        scheduler.step()
        
        # Store
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f'Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Val AUC: {val_auc:.4f}')
        
        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'epoch': epoch
            }, best_model_path)
            print(f'✓ Saved (AUC improved to {val_auc:.4f})')
    
    # ========== PHASE 2: PARTIAL UNFREEZING ==========
    print("\n" + "🔥 PHASE 2: Unfreezing Last 3 Blocks")
    print("-" * 60)
    
    model.unfreeze_last_n_blocks(3)  # Unfreeze last 3
    
    # Lower learning rate for unfrozen layers
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 0.001},
        {'params': model.features[-3:].parameters(), 'lr': 0.0001}  # 10x lower
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    for epoch in range(15):
        print(f'\nPhase 2 - Epoch {epoch+1}/15')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, val_labels, val_probs = validate(
            model, test_loader, criterion, device
        )
        
        # Calculate AUC-ROC
        val_auc = roc_auc_score(val_labels, val_probs)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f'Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Val AUC: {val_auc:.4f}')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'epoch': epoch + 12
            }, best_model_path)
            print(f'✓ Saved (AUC improved to {val_auc:.4f})')
    
    # ========== PHASE 3: FULL FINE-TUNING ==========
    print("\n" + "🔥 PHASE 3: Full Fine-Tuning (All Layers)")
    print("-" * 60)
    
    model.unfreeze_last_n_blocks(-1)  # Unfreeze all
    
    # Discriminative learning rates
    blocks = list(model.features.children())
    n = len(blocks)
    
    # Helper function to get parameters from a list of blocks
    def get_params(block_list):
        """Collect all parameters from a list of blocks."""
        params = []
        for block in block_list:
            params.extend(block.parameters())
        return params
    
    optimizer = optim.AdamW([
        {'params': get_params(blocks[:n//3]), 'lr': 0.00001},           # Early layers
        {'params': get_params(blocks[n//3:2*n//3]), 'lr': 0.00005},     # Middle layers
        {'params': get_params(blocks[2*n//3:]), 'lr': 0.0001},          # Late layers
        {'params': model.classifier.parameters(), 'lr': 0.001}          # Classifier head
    ], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    
    for epoch in range(15):
        print(f'\nPhase 3 - Epoch {epoch+1}/15')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, val_labels, val_probs = validate(
            model, test_loader, criterion, device
        )
        
        # Calculate AUC-ROC
        val_auc = roc_auc_score(val_labels, val_probs)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f'Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Val AUC: {val_auc:.4f}')
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'epoch': epoch + 27
            }, best_model_path)
            print(f'✓ Saved (AUC improved to {val_auc:.4f})')
    
    print("\n" + "="*80)
    print("✓ Training Complete!")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Final validation AUC: {history['val_auc'][-1]:.4f}")
    print("="*80)
    
    return model, history
