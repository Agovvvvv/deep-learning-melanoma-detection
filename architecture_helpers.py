# Low-Cost Deep Learning System for Automated Melanoma Detection
# Copyright (C) 2025 Nicolò Calandra
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
Helper functions for handling different model architectures (CNN vs ViT) in training.
"""

import torch.nn as nn


def get_backbone_blocks(model):
    """
    Get backbone blocks from different model architectures.
    
    Returns:
        list: List of backbone blocks/layers
        str: Architecture type ('cnn' or 'vit')
    """
    if hasattr(model, 'features'):
        # CNN architectures (EfficientNet, ConvNeXt)
        return list(model.features.children()), 'cnn'
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        # Vision Transformer architectures
        return list(model.encoder.layers.children()), 'vit'
    else:
        raise ValueError(f"Unknown architecture type for model: {type(model).__name__}")


def get_backbone_params(model, block_indices=None):
    """
    Get parameters from specific backbone blocks.
    
    Args:
        model: The model
        block_indices: List of block indices to get params from, or None for all
        
    Returns:
        list: List of parameters
    """
    blocks, arch_type = get_backbone_blocks(model)
    
    if block_indices is None:
        # Return all backbone parameters
        params = []
        for block in blocks:
            params.extend(list(block.parameters()))
        
        # For ViT, also include conv_proj parameters
        if arch_type == 'vit' and hasattr(model, 'conv_proj'):
            params.extend(list(model.conv_proj.parameters()))
        
        return params
    else:
        # Return parameters from specific blocks
        params = []
        for idx in block_indices:
            if idx < len(blocks):
                params.extend(list(blocks[idx].parameters()))
        return params


def get_unfrozen_backbone_params(model, n_blocks):
    """
    Get parameters from the last n blocks after unfreezing.
    
    Args:
        model: The model
        n_blocks: Number of last blocks to get params from
        
    Returns:
        list: List of parameters from the last n blocks
    """
    blocks, arch_type = get_backbone_blocks(model)
    
    if n_blocks == -1:
        # All blocks
        return get_backbone_params(model)
    elif n_blocks == 0:
        return []
    else:
        # Last n blocks
        last_blocks = blocks[-n_blocks:]
        params = []
        for block in last_blocks:
            params.extend(list(block.parameters()))
        return params


def get_discriminative_lr_groups(model, lr_early, lr_mid, lr_late, lr_classifier):
    """
    Create parameter groups with discriminative learning rates.
    Works for both CNN and ViT architectures.
    
    Args:
        model: The model
        lr_early: Learning rate for early layers
        lr_mid: Learning rate for middle layers
        lr_late: Learning rate for late layers
        lr_classifier: Learning rate for classifier
        
    Returns:
        list: List of parameter group dictionaries for optimizer
    """
    blocks, arch_type = get_backbone_blocks(model)
    n = len(blocks)
    
    # Split into three groups
    early_blocks = blocks[:n//3]
    mid_blocks = blocks[n//3:2*n//3]
    late_blocks = blocks[2*n//3:]
    
    # Collect parameters
    def get_params(block_list):
        params = []
        for block in block_list:
            params.extend(list(block.parameters()))
        return params
    
    param_groups = [
        {'params': get_params(early_blocks), 'lr': lr_early},
        {'params': get_params(mid_blocks), 'lr': lr_mid},
        {'params': get_params(late_blocks), 'lr': lr_late},
        {'params': model.classifier.parameters(), 'lr': lr_classifier}
    ]
    
    # For ViT, also add conv_proj to early layers
    if arch_type == 'vit' and hasattr(model, 'conv_proj'):
        param_groups[0]['params'].extend(list(model.conv_proj.parameters()))
    
    return param_groups


def print_trainable_parameters(model):
    """Print number of trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
