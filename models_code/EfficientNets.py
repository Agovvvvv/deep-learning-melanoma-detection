import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3, efficientnet_b4, EfficientNet_B3_Weights, EfficientNet_B4_Weights


class SimpleEfficientNet(nn.Module):
    """
    Simple EfficientNet-B3 baseline classifier.
    
    Architecture:
    - EfficientNet-B3 pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleEfficientNet, self).__init__()
        
        # Load pre-trained EfficientNet-B3
        if pretrained:
            weights = EfficientNet_B3_Weights.DEFAULT
            base_model = efficientnet_b3(weights=weights)
        else:
            base_model = efficientnet_b3(weights=None)
        
        # Extract feature extractor (all layers except classifier)
        self.features = base_model.features
        
        # EfficientNet-B3 outputs 1536 features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """Freeze all feature extraction layers."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_blocks(self, n):
        """
        Unfreeze last n blocks for progressive training.
        
        Args:
            n: Number of blocks to unfreeze
               -1 = unfreeze all
               0 = keep all frozen
               1-9 = unfreeze last n blocks
        """
        if n == 0:
            return  # Already frozen
        elif n == -1:
            # Unfreeze everything
            for param in self.features.parameters():
                param.requires_grad = True
            print(f"✓ All feature layers unfrozen")
        else:
            # Unfreeze last n blocks
            all_blocks = list(self.features.children())
            for block in all_blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"✓ Unfroze last {n} blocks")


class SimpleEfficientNetB4(nn.Module):
    """
    Simple EfficientNet-B4 classifier.
    
    Architecture:
    - EfficientNet-B4 pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleEfficientNetB4, self).__init__()
        
        # Load pre-trained EfficientNet-B4
        if pretrained:
            weights = EfficientNet_B4_Weights.DEFAULT
            base_model = efficientnet_b4(weights=weights)
        else:
            base_model = efficientnet_b4(weights=None)
        
        # Extract feature extractor (all layers except classifier)
        self.features = base_model.features
        
        # EfficientNet-B4 outputs 1792 features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1792, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        x = self.features(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """Freeze all feature extraction layers."""
        for param in self.features.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_blocks(self, n):
        """
        Unfreeze last n blocks for progressive training.
        
        Args:
            n: Number of blocks to unfreeze
               -1 = unfreeze all
               0 = keep all frozen
               1-9 = unfreeze last n blocks
        """
        if n == 0:
            return  # Already frozen
        elif n == -1:
            # Unfreeze everything
            for param in self.features.parameters():
                param.requires_grad = True
            print(f"✓ All feature layers unfrozen")
        else:
            # Unfreeze last n blocks
            all_blocks = list(self.features.children())
            for block in all_blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"✓ Unfroze last {n} blocks")