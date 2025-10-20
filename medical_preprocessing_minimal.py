"""
MINIMAL AUGMENTATION VERSION
=============================
For medical images, less is often more.
This version focuses on medical preprocessing with only essential augmentations.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


# Import the medical preprocessing classes from original file
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from medical_preprocessing import HairRemoval, LesionCropping, ContrastEnhancement
except ImportError:
    print("Warning: Could not import from medical_preprocessing.py")
    print("Make sure medical_preprocessing.py is in the same directory")


def get_medical_train_transforms_minimal(img_size=224, enable_hair_removal=True, 
                                          enable_lesion_crop=True, enable_contrast=True):
    """
    MINIMAL augmentation for medical images.
    
    Only includes:
    - Medical preprocessing (hair, contrast, crop)
    - Flips (medically valid - lesion can appear from any angle)
    - Very mild color jitter (lighting variations only)
    - NO rotation, NO affine, NO erasing, NO blur
    
    Args:
        img_size: Target image size
        enable_hair_removal: Remove hair artifacts
        enable_lesion_crop: Crop to lesion region
        enable_contrast: Enhance contrast
    
    Returns:
        torchvision.transforms.Compose
    """
    
    # Medical preprocessing
    medical_steps = []
    
    if enable_hair_removal:
        medical_steps.append(HairRemoval(kernel_size=17))
    
    if enable_contrast:
        medical_steps.append(ContrastEnhancement(clip_limit=2.0))
    
    if enable_lesion_crop:
        medical_steps.append(LesionCropping(margin=0.15, min_crop_ratio=0.5))
    
    # Resize to target size
    medical_steps.append(transforms.Resize((img_size, img_size)))
    
    # MINIMAL augmentation (medically realistic)
    augmentation_steps = [
        # Flips are medically valid (lesion orientation doesn't matter)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # Very mild color jitter (only lighting variations)
        transforms.ColorJitter(
            brightness=0.1,   # Minimal - just lighting
            contrast=0.1,     # Minimal
            saturation=0.0,   # NO saturation change (keeps realistic colors)
            hue=0.0           # NO hue change (no purple/green lesions)
        ),
        
        transforms.ToTensor(),
        
        # Standard ImageNet normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    return transforms.Compose(medical_steps + augmentation_steps)


def get_medical_train_transforms_flips_only(img_size=224, enable_hair_removal=True, 
                                             enable_lesion_crop=True, enable_contrast=True):
    """
    FLIPS ONLY augmentation - Most conservative, medically safe.
    
    Only includes:
    - Medical preprocessing (hair, contrast, crop)
    - Flips (horizontal + vertical)
    - NO color changes, NO rotation, NO anything else
    
    Args:
        img_size: Target image size
        enable_hair_removal: Remove hair artifacts
        enable_lesion_crop: Crop to lesion region
        enable_contrast: Enhance contrast
    
    Returns:
        torchvision.transforms.Compose
    """
    
    # Medical preprocessing
    medical_steps = []
    
    if enable_hair_removal:
        medical_steps.append(HairRemoval(kernel_size=17))
    
    if enable_contrast:
        medical_steps.append(ContrastEnhancement(clip_limit=2.0))
    
    if enable_lesion_crop:
        medical_steps.append(LesionCropping(margin=0.15, min_crop_ratio=0.5))
    
    # Resize to target size
    medical_steps.append(transforms.Resize((img_size, img_size)))
    
    # FLIPS ONLY - no color changes
    augmentation_steps = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        transforms.ToTensor(),
        
        # Standard ImageNet normalization
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    return transforms.Compose(medical_steps + augmentation_steps)


def get_medical_train_transforms_none(img_size=224, enable_hair_removal=True, 
                                      enable_lesion_crop=True, enable_contrast=True):
    """
    NO augmentation - only medical preprocessing.
    
    Use this to test if augmentation is actually helping or hurting.
    
    Args:
        img_size: Target image size
        enable_hair_removal: Remove hair artifacts
        enable_lesion_crop: Crop to lesion region
        enable_contrast: Enhance contrast
    
    Returns:
        torchvision.transforms.Compose
    """
    
    # Medical preprocessing
    medical_steps = []
    
    if enable_hair_removal:
        medical_steps.append(HairRemoval(kernel_size=17))
    
    if enable_contrast:
        medical_steps.append(ContrastEnhancement(clip_limit=2.0))
    
    if enable_lesion_crop:
        medical_steps.append(LesionCropping(margin=0.15, min_crop_ratio=0.5))
    
    medical_steps.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transforms.Compose(medical_steps)


def get_medical_test_transforms(img_size=224, enable_hair_removal=True,
                                enable_lesion_crop=True, enable_contrast=True):
    """
    Test transforms (same as no-augmentation train).
    
    Args:
        img_size: Target image size
        enable_hair_removal: Remove hair artifacts
        enable_lesion_crop: Crop to lesion region
        enable_contrast: Enhance contrast
    
    Returns:
        torchvision.transforms.Compose
    """
    
    medical_steps = []
    
    if enable_hair_removal:
        medical_steps.append(HairRemoval(kernel_size=17))
    
    if enable_contrast:
        medical_steps.append(ContrastEnhancement(clip_limit=2.0))
    
    if enable_lesion_crop:
        medical_steps.append(LesionCropping(margin=0.15, min_crop_ratio=0.5))
    
    medical_steps.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transforms.Compose(medical_steps)


if __name__ == "__main__":
    print("="*80)
    print("MINIMAL AUGMENTATION FOR MEDICAL IMAGES")
    print("="*80)
    print("\nAvailable transform presets:")
    print("\n1. get_medical_train_transforms_flips_only() ⭐ RECOMMENDED")
    print("   → Medical preprocessing + Flips ONLY")
    print("   → No color changes, no artifacts")
    print("   → Most conservative, medically safe")
    print("\n2. get_medical_train_transforms_minimal()")
    print("   → Medical preprocessing + Flips + Mild color jitter")
    print("   → Slight color variations (may cause purple tints)")
    print("\n3. get_medical_train_transforms_none()")
    print("   → Medical preprocessing ONLY, no augmentation")
    print("   → Use to test if augmentation helps at all")
    print("\n4. get_medical_test_transforms()")
    print("   → Test/validation (no augmentation)")
    print("\nWhy minimal augmentation?")
    print("  • You have 8,868 training images (enough data)")
    print("  • Medical images should look realistic")
    print("  • Heavy augmentation creates artifacts (purple colors, black boxes)")
    print("  • Flips provide 4x data diversity without artifacts")
    print("="*80)
