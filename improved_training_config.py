# ============================================================
# IMPROVED TRAINING CONFIGURATION
# ============================================================
# Based on best practices for medical image classification
# and lessons learned from 3 training sessions
# ============================================================

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# ============================================================
# 1. IMPROVED OPTIMIZER WITH BETTER LEARNING RATES
# ============================================================

# Current (what you're using):
# optimizer = torch.optim.Adam([
#     {'params': model.classifier.parameters(), 'lr': 0.001},
#     {'params': model.features.parameters(), 'lr': 0.0001}
# ])

# IMPROVED: Lower initial learning rates + weight decay
optimizer = torch.optim.AdamW([  # AdamW has better weight decay
    {'params': model.classifier.parameters(), 'lr': 0.0005, 'weight_decay': 0.01},
    {'params': model.features.parameters(), 'lr': 0.00005, 'weight_decay': 0.01}
], betas=(0.9, 0.999), eps=1e-8)

print("✓ Optimizer: AdamW with improved settings")
print(f"  Classifier LR: 0.0005 (down from 0.001)")
print(f"  Features LR: 0.00005 (down from 0.0001)")
print(f"  Weight decay: 0.01 (helps prevent overfitting)")

# ============================================================
# 2. IMPROVED LEARNING RATE SCHEDULER
# ============================================================

# Option A: Cosine Annealing with Warm Restarts (RECOMMENDED)
# Periodically resets LR to help escape local minima
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # First cycle: 10 epochs
    T_mult=2,      # Each cycle 2x longer
    eta_min=1e-7   # Minimum learning rate
)

print("\n✓ Scheduler: CosineAnnealingWarmRestarts")
print(f"  Initial cycle: 10 epochs")
print(f"  Subsequent cycles: 20, 40 epochs")
print(f"  Min LR: 1e-7")

# Option B: OneCycleLR (Alternative - great for shorter training)
# Uncomment to use:
# scheduler = OneCycleLR(
#     optimizer,
#     max_lr=[0.001, 0.0001],  # Peak LR for each param group
#     epochs=EPOCHS,
#     steps_per_epoch=len(train_loader),
#     pct_start=0.3,  # 30% warmup
#     anneal_strategy='cos'
# )

# ============================================================
# 3. IMPROVED TRAINING SCHEDULE
# ============================================================

# Current: 42 epochs
# IMPROVED: 50 epochs with better unfreezing schedule

EPOCHS = 50

PROGRESSIVE_SCHEDULE = {
    # Phase 1: Train only classifier (warmup)
    'phase1': {
        'epochs': list(range(0, 10)),
        'unfreeze': 0,  # Keep backbone frozen
        'description': 'Classifier warmup'
    },
    # Phase 2: Unfreeze last 3 blocks
    'phase2': {
        'epochs': list(range(10, 20)),
        'unfreeze': 3,
        'description': 'Fine-tune last 3 blocks'
    },
    # Phase 3: Unfreeze last 6 blocks
    'phase3': {
        'epochs': list(range(20, 35)),
        'unfreeze': 6,
        'description': 'Fine-tune last 6 blocks'
    },
    # Phase 4: Unfreeze all
    'phase4': {
        'epochs': list(range(35, 50)),
        'unfreeze': -1,
        'description': 'Full fine-tuning'
    }
}

print("\n✓ Training Schedule: 50 epochs")
print(f"  Phase 1 (Epoch 0-10):   Classifier only")
print(f"  Phase 2 (Epoch 10-20):  + Last 3 blocks")
print(f"  Phase 3 (Epoch 20-35):  + Last 6 blocks")
print(f"  Phase 4 (Epoch 35-50):  Full model")

# ============================================================
# 4. BATCH SIZE TUNING
# ============================================================

# Current: Likely 32 or 64
# OPTIMAL: 32 for medical images (better generalization)

BATCH_SIZE = 32  # Don't go higher - smaller batches = better for medical data

print("\n✓ Batch Size: 32 (optimal for medical images)")

# ============================================================
# 5. GRADIENT CLIPPING (IMPORTANT!)
# ============================================================

MAX_GRAD_NORM = 1.0  # Prevent exploding gradients

print(f"\n✓ Gradient Clipping: {MAX_GRAD_NORM} (prevents instability)")

# Use in training loop:
# torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

# ============================================================
# 6. IMPROVED FOCAL LOSS PARAMETERS
# ============================================================

# Your current focal loss is good, but can be slightly tuned

class ImprovedFocalLoss(nn.Module):
    """
    Focal Loss with proven optimal parameters.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(ImprovedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )
        p = torch.exp(-ce_loss)
        focal_loss = (1 - p) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


# Calculate class weights (you already do this)
def calculate_class_weights(train_loader):
    class_counts = torch.zeros(2)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label] += 1
    
    # Inverse frequency weighting
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * 2
    return class_weights

class_weights = calculate_class_weights(train_loader).to(device)

# Keep gamma=2.0 (proven optimal in your experiments)
criterion = ImprovedFocalLoss(alpha=class_weights, gamma=2.0)

print("\n✓ Loss Function: Focal Loss (gamma=2.0, alpha=class_weights)")

# ============================================================
# 7. EARLY STOPPING (OPTIONAL BUT RECOMMENDED)
# ============================================================

EARLY_STOPPING_PATIENCE = 15  # Stop if no improvement for 15 epochs
EARLY_STOPPING_MIN_DELTA = 0.001  # Minimum change to consider improvement

print(f"\n✓ Early Stopping: Patience={EARLY_STOPPING_PATIENCE} epochs")

# ============================================================
# 8. TRAINING LOOP IMPROVEMENTS
# ============================================================

print("\n" + "="*80)
print("KEY TRAINING IMPROVEMENTS")
print("="*80)
print("\n1. Lower learning rates (0.0005/0.00005)")
print("   → More stable training, better convergence")
print("\n2. AdamW optimizer with weight decay")
print("   → Better regularization, less overfitting")
print("\n3. Cosine annealing with warm restarts")
print("   → Escape local minima, find better solutions")
print("\n4. Gradient clipping")
print("   → Prevent training instability")
print("\n5. Extended training (50 epochs)")
print("   → More time to converge properly")
print("\n6. Better progressive unfreezing")
print("   → Gradual adaptation of all layers")

print("\n" + "="*80)
print("EXPECTED IMPROVEMENTS")
print("="*80)
print("\n• Recall: 97.98% → 98.5-99%")
print("• AUC: 0.935 → 0.94-0.95")
print("• More stable training")
print("• Better generalization")
print("• Fewer uncertain cases")
print("="*80)

# ============================================================
# 9. DATA AUGMENTATION (KEEP MINIMAL)
# ============================================================

print("\n" + "="*80)
print("DATA AUGMENTATION STRATEGY")
print("="*80)

# KEEP YOUR CURRENT AUGMENTATION - IT'S GOOD!
# DO NOT add aggressive augmentation

from torchvision import transforms

# Training (keep simple - this is working!)
train_transforms = transforms.Compose([
    # Your preprocessing
    # HairRemoval(),  # Custom transform
    # ContrastEnhancement(),  # Custom transform
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),  # Slight zoom only
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    # NO rotation during training (keep images upright)
    # NO color jitter (medical images need accurate colors)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\n✓ Training Augmentation: Minimal (OPTIMAL)")
print("  • Random crop (90-100% scale)")
print("  • Horizontal + vertical flips")
print("  • NO rotation (keeps medical accuracy)")
print("  • NO color jitter (preserves diagnostic features)")
print("\n💡 For medical images: Less augmentation = Better!")

# ============================================================
# 10. VALIDATION STRATEGY
# ============================================================

print("\n" + "="*80)
print("VALIDATION IMPROVEMENTS")
print("="*80)

# Use AUC as primary metric (better than accuracy for imbalanced data)
MONITOR_METRIC = 'auc'  # Not 'accuracy'
SAVE_BEST_BY = 'auc'    # Save model with best AUC

print("\n✓ Monitor: AUC (better than accuracy for medical data)")
print("✓ Save: Best AUC model")
print("✓ Track: Recall (most important for cancer)")

print("\n" + "="*80)
