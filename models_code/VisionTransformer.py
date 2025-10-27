import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32
from torchvision.models import ViT_B_16_Weights, ViT_B_32_Weights, ViT_L_16_Weights, ViT_L_32_Weights


class SimpleViTB16(nn.Module):
    """
    Simple Vision Transformer Base/16 classifier.
    
    Architecture:
    - ViT-B/16 pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleViTB16, self).__init__()
        
        # Load pre-trained ViT-B/16
        if pretrained:
            weights = ViT_B_16_Weights.DEFAULT
            base_model = vit_b_16(weights=weights)
        else:
            base_model = vit_b_16(weights=None)
        
        # Extract encoder (all layers except head)
        self.encoder = base_model.encoder
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        
        # ViT-B/16 outputs 768 features
        hidden_dim = base_model.hidden_dim
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 512),
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
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Expand class token to batch
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        # Pass through encoder
        x = self.encoder(x)
        
        # Extract class token (first token)
        x = x[:, 0]
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """Freeze all feature extraction layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.conv_proj.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_blocks(self, n):
        """
        Unfreeze last n transformer blocks for progressive training.
        
        Args:
            n: Number of blocks to unfreeze
               -1 = unfreeze all
               0 = keep all frozen
               1-12 = unfreeze last n blocks
        """
        if n == 0:
            return  # Already frozen
        elif n == -1:
            # Unfreeze everything
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.conv_proj.parameters():
                param.requires_grad = True
            print(f"✓ All encoder layers unfrozen")
        else:
            # Unfreeze last n blocks
            all_blocks = list(self.encoder.layers.children())
            for block in all_blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"✓ Unfroze last {n} transformer blocks")


class SimpleViTB32(nn.Module):
    """
    Simple Vision Transformer Base/32 classifier.
    
    Architecture:
    - ViT-B/32 pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleViTB32, self).__init__()
        
        # Load pre-trained ViT-B/32
        if pretrained:
            weights = ViT_B_32_Weights.DEFAULT
            base_model = vit_b_32(weights=weights)
        else:
            base_model = vit_b_32(weights=None)
        
        # Extract encoder (all layers except head)
        self.encoder = base_model.encoder
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        
        # ViT-B/32 outputs 768 features
        hidden_dim = base_model.hidden_dim
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 512),
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
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Expand class token to batch
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        # Pass through encoder
        x = self.encoder(x)
        
        # Extract class token (first token)
        x = x[:, 0]
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """Freeze all feature extraction layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.conv_proj.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_blocks(self, n):
        """
        Unfreeze last n transformer blocks for progressive training.
        
        Args:
            n: Number of blocks to unfreeze
               -1 = unfreeze all
               0 = keep all frozen
               1-12 = unfreeze last n blocks
        """
        if n == 0:
            return  # Already frozen
        elif n == -1:
            # Unfreeze everything
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.conv_proj.parameters():
                param.requires_grad = True
            print(f"✓ All encoder layers unfrozen")
        else:
            # Unfreeze last n blocks
            all_blocks = list(self.encoder.layers.children())
            for block in all_blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"✓ Unfroze last {n} transformer blocks")


class SimpleViTL16(nn.Module):
    """
    Simple Vision Transformer Large/16 classifier.
    
    Architecture:
    - ViT-L/16 pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleViTL16, self).__init__()
        
        # Load pre-trained ViT-L/16
        if pretrained:
            weights = ViT_L_16_Weights.DEFAULT
            base_model = vit_l_16(weights=weights)
        else:
            base_model = vit_l_16(weights=None)
        
        # Extract encoder (all layers except head)
        self.encoder = base_model.encoder
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        
        # ViT-L/16 outputs 1024 features
        hidden_dim = base_model.hidden_dim
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 768),
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
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Expand class token to batch
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        # Pass through encoder
        x = self.encoder(x)
        
        # Extract class token (first token)
        x = x[:, 0]
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """Freeze all feature extraction layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.conv_proj.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_blocks(self, n):
        """
        Unfreeze last n transformer blocks for progressive training.
        
        Args:
            n: Number of blocks to unfreeze
               -1 = unfreeze all
               0 = keep all frozen
               1-24 = unfreeze last n blocks
        """
        if n == 0:
            return  # Already frozen
        elif n == -1:
            # Unfreeze everything
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.conv_proj.parameters():
                param.requires_grad = True
            print(f"✓ All encoder layers unfrozen")
        else:
            # Unfreeze last n blocks
            all_blocks = list(self.encoder.layers.children())
            for block in all_blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"✓ Unfroze last {n} transformer blocks")


class SimpleViTL32(nn.Module):
    """
    Simple Vision Transformer Large/32 classifier.
    
    Architecture:
    - ViT-L/32 pre-trained backbone
    - 3-layer MLP classifier with BatchNorm
    - Progressive dropout (0.5 → 0.4 → 0.3)
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(SimpleViTL32, self).__init__()
        
        # Load pre-trained ViT-L/32
        if pretrained:
            weights = ViT_L_32_Weights.DEFAULT
            base_model = vit_l_32(weights=weights)
        else:
            base_model = vit_l_32(weights=None)
        
        # Extract encoder (all layers except head)
        self.encoder = base_model.encoder
        self.conv_proj = base_model.conv_proj
        self.class_token = base_model.class_token
        
        # ViT-L/32 outputs 1024 features
        hidden_dim = base_model.hidden_dim
        
        # Simple 3-layer classifier with BatchNorm
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 768),
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
        # Reshape and permute the input tensor
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Expand class token to batch
        batch_size = x.shape[0]
        class_token = self.class_token.expand(batch_size, -1, -1)
        x = torch.cat([class_token, x], dim=1)
        
        # Pass through encoder
        x = self.encoder(x)
        
        # Extract class token (first token)
        x = x[:, 0]
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def freeze_backbone(self):
        """Freeze all feature extraction layers."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.conv_proj.parameters():
            param.requires_grad = False
    
    def unfreeze_last_n_blocks(self, n):
        """
        Unfreeze last n transformer blocks for progressive training.
        
        Args:
            n: Number of blocks to unfreeze
               -1 = unfreeze all
               0 = keep all frozen
               1-24 = unfreeze last n blocks
        """
        if n == 0:
            return  # Already frozen
        elif n == -1:
            # Unfreeze everything
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.conv_proj.parameters():
                param.requires_grad = True
            print(f"✓ All encoder layers unfrozen")
        else:
            # Unfreeze last n blocks
            all_blocks = list(self.encoder.layers.children())
            for block in all_blocks[-n:]:
                for param in block.parameters():
                    param.requires_grad = True
            print(f"✓ Unfroze last {n} transformer blocks")
