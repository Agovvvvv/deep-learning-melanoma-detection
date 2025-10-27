import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small, convnext_base, convnext_large
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights, ConvNeXt_Large_Weights


class SimpleConvNeXtTiny(nn.Module):
    """
    Simple ConvNeXt-Tiny classifier.
    
    Architecture:
    - ConvNeXt-Tiny pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleConvNeXtTiny, self).__init__()
        
        # Load pre-trained ConvNeXt-Tiny
        if pretrained:
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            base_model = convnext_tiny(weights=weights)
        else:
            base_model = convnext_tiny(weights=None)
        
        # Extract feature extractor (all layers except classifier)
        self.features = base_model.features
        
        # ConvNeXt-Tiny outputs 768 features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
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
               1-7 = unfreeze last n blocks
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


class SimpleConvNeXtSmall(nn.Module):
    """
    Simple ConvNeXt-Small classifier.
    
    Architecture:
    - ConvNeXt-Small pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleConvNeXtSmall, self).__init__()
        
        # Load pre-trained ConvNeXt-Small
        if pretrained:
            weights = ConvNeXt_Small_Weights.DEFAULT
            base_model = convnext_small(weights=weights)
        else:
            base_model = convnext_small(weights=None)
        
        # Extract feature extractor (all layers except classifier)
        self.features = base_model.features
        
        # ConvNeXt-Small outputs 768 features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
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
               1-7 = unfreeze last n blocks
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


class SimpleConvNeXtBase(nn.Module):
    """
    Simple ConvNeXt-Base classifier.
    
    Architecture:
    - ConvNeXt-Base pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleConvNeXtBase, self).__init__()
        
        # Load pre-trained ConvNeXt-Base
        if pretrained:
            weights = ConvNeXt_Base_Weights.DEFAULT
            base_model = convnext_base(weights=weights)
        else:
            base_model = convnext_base(weights=None)
        
        # Extract feature extractor (all layers except classifier)
        self.features = base_model.features
        
        # ConvNeXt-Base outputs 1024 features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(768, 512),
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
               1-7 = unfreeze last n blocks
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


class SimpleConvNeXtLarge(nn.Module):
    """
    Simple ConvNeXt-Large classifier.
    
    Architecture:
    - ConvNeXt-Large pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleConvNeXtLarge, self).__init__()
        
        # Load pre-trained ConvNeXt-Large
        if pretrained:
            weights = ConvNeXt_Large_Weights.DEFAULT
            base_model = convnext_large(weights=weights)
        else:
            base_model = convnext_large(weights=None)
        
        # Extract feature extractor (all layers except classifier)
        self.features = base_model.features
        
        # ConvNeXt-Large outputs 1536 features
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
               1-7 = unfreeze last n blocks
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
