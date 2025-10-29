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
Medical Image Preprocessing for Melanoma Detection - Final Version
===================================================================
Consolidated preprocessing pipeline with black corner bias removal.

Components:
1. InpaintingMaskFiller - Removes black corner bias
2. HairRemoval - Removes hair artifacts  
3. ContrastEnhancement - CLAHE for better boundaries
4. Transform presets - Ready-to-use pipelines
5. Visualization - Debug preprocessing steps
"""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


# ============================================================
# CORE PREPROCESSING CLASSES
# ============================================================

class InpaintingMaskFiller:
    """
    Removes black corner bias using OpenCV inpainting.
    
    Detects and fills black corners (from circular dermoscopy masks)
    to prevent the model from learning corner-based features.
    """
    
    def __init__(self, threshold=30, inpaint_radius=10, min_black_ratio=0.001, debug=False):
        """
        Args:
            threshold: Intensity threshold for detecting black regions (0-255)
            inpaint_radius: Radius for inpainting algorithm (larger = better color matching)
            min_black_ratio: Minimum black ratio to trigger processing (0.001 = 0.1%)
            debug: Print processing information
        """
        self.threshold = threshold
        self.inpaint_radius = inpaint_radius
        self.min_black_ratio = min_black_ratio
        self.debug = debug
    
    def __call__(self, img):
        """Apply inpainting to remove black corners."""
        try:
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Detect black regions
            _, black_mask = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY_INV)
            
            # Calculate black ratio
            black_ratio = np.sum(black_mask > 0) / (h * w)
            
            # Skip if no significant black regions
            if black_ratio < self.min_black_ratio:
                if self.debug:
                    print(f"InpaintingMaskFiller: No black regions ({black_ratio:.3%}), skipping")
                return img
            
            # Dilate mask slightly for cleaner edges
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            black_mask_dilated = cv2.dilate(black_mask, kernel_dilate, iterations=1)
            
            # Apply inpainting (Navier-Stokes algorithm for better texture)
            result = cv2.inpaint(img_np, black_mask_dilated, self.inpaint_radius, cv2.INPAINT_NS)
            
            if self.debug:
                print(f"InpaintingMaskFiller: ✓ Inpainted {black_ratio:.1%} black area")
            
            return Image.fromarray(result)
            
        except Exception as e:
            if self.debug:
                print(f"InpaintingMaskFiller: Error - {e}, returning original")
            return img


class HairRemoval:
    """
    Removes hair artifacts using morphological operations.
    
    Uses black-hat transform with a line structuring element
    to detect and inpaint hair-like structures.
    """
    
    def __init__(self, kernel_size=17, debug=False):
        """
        Args:
            kernel_size: Size of morphological kernel (larger = thicker hair removed)
            debug: Print processing information
        """
        self.kernel_size = kernel_size
        self.debug = debug
    
    def __call__(self, img):
        """Apply hair removal."""
        try:
            img_np = np.array(img)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Create morphological kernel (line for hair detection)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
            
            # Black-hat transform to detect dark structures (hair)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold to create mask
            _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
            
            # Inpaint hair regions
            result = cv2.inpaint(img_np, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            
            if self.debug:
                hair_pixels = np.sum(hair_mask > 0)
                print(f"HairRemoval: Removed {hair_pixels} hair pixels")
            
            return Image.fromarray(result)
            
        except Exception as e:
            if self.debug:
                print(f"HairRemoval: Error - {e}, returning original")
            return img


class ContrastEnhancement:
    """
    Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Improves lesion boundary definition without over-amplifying noise.
    """
    
    def __init__(self, clip_limit=2.0, tile_size=(8, 8), debug=False):
        """
        Args:
            clip_limit: Contrast limiting threshold (higher = more contrast)
            tile_size: Size of grid for local histogram equalization
            debug: Print processing information
        """
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.debug = debug
    
    def __call__(self, img):
        """Apply CLAHE contrast enhancement."""
        try:
            img_np = np.array(img)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel only
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)
            l_enhanced = clahe.apply(l)
            
            # Merge channels back
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
            
            if self.debug:
                print("ContrastEnhancement: Applied CLAHE")
            
            return Image.fromarray(result)
            
        except Exception as e:
            if self.debug:
                print(f"ContrastEnhancement: Error - {e}, returning original")
            return img


# ============================================================
# TRANSFORM PRESETS
# ============================================================

def get_train_transforms(img_size=224, enable_hair_removal=True, 
                        enable_inpainting=True, enable_contrast=True):
    """
    Training transforms with medical preprocessing and flips only.
    
    Pipeline:
    1. Inpainting (removes black corner bias)
    2. Hair removal (removes artifacts)
    3. CLAHE (enhances contrast)
    4. Resize
    5. Random flips (medically valid augmentation)
    6. Normalize
    
    Args:
        img_size: Target image size
        enable_hair_removal: Apply hair removal
        enable_inpainting: Apply black corner inpainting
        enable_contrast: Apply CLAHE
    
    Returns:
        torchvision.transforms.Compose
    """
    
    medical_steps = []
    
    # Step 1: Remove black corners (MUST be first to remove bias)
    if enable_inpainting:
        medical_steps.append(InpaintingMaskFiller(
            threshold=50,  # Increased: catches darker gray corners (was 30)
            inpaint_radius=15,  # Larger radius: better texture matching (was 10)
            min_black_ratio=0.001
        ))
    
    # Step 2: Remove hair artifacts
    if enable_hair_removal:
        medical_steps.append(HairRemoval(kernel_size=17))
    
    # Step 3: Enhance contrast
    if enable_contrast:
        medical_steps.append(ContrastEnhancement(clip_limit=2.0))
    
    # Standard augmentation
    augmentation_steps = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    return transforms.Compose(medical_steps + augmentation_steps)


def get_test_transforms(img_size=224, enable_hair_removal=True,
                       enable_inpainting=True, enable_contrast=True):
    """
    Test/validation transforms (no augmentation).
    
    Same preprocessing as training but without flips.
    
    Args:
        img_size: Target image size
        enable_hair_removal: Apply hair removal
        enable_inpainting: Apply black corner inpainting
        enable_contrast: Apply CLAHE
    
    Returns:
        torchvision.transforms.Compose
    """
    
    medical_steps = []
    
    if enable_inpainting:
        medical_steps.append(InpaintingMaskFiller(
            threshold=50,  # Increased: catches darker gray corners (was 30)
            inpaint_radius=15,  # Larger radius: better texture matching (was 10)
            min_black_ratio=0.001
        ))
    
    if enable_hair_removal:
        medical_steps.append(HairRemoval(kernel_size=17))
    
    if enable_contrast:
        medical_steps.append(ContrastEnhancement(clip_limit=2.0))
    
    medical_steps.extend([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transforms.Compose(medical_steps)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_preprocessing(img_path, save_path=None):
    """
    Visualize each preprocessing step for debugging.
    
    Shows: Original → Inpainted → Hair Removed → Contrast Enhanced
    
    Args:
        img_path: Path to input image
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load image
    original = Image.open(img_path).convert('RGB')
    
    # Apply each step
    inpainted = InpaintingMaskFiller(threshold=50, inpaint_radius=15, min_black_ratio=0.001)(original)
    hair_removed = HairRemoval(kernel_size=17)(inpainted)
    contrast_enhanced = ContrastEnhancement(clip_limit=2.0)(hair_removed)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Medical Preprocessing Pipeline', fontsize=16, fontweight='bold')
    
    axes[0].imshow(original)
    axes[0].set_title('1. Original', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(inpainted)
    axes[1].set_title('2. Black Corners Filled\n(Inpainting)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(hair_removed)
    axes[2].set_title('3. Hair Removed\n(Morphological)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(contrast_enhanced)
    axes[3].set_title('4. Contrast Enhanced\n(CLAHE)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    
    plt.show()


# ============================================================
# BATCH PREPROCESSING
# ============================================================

def preprocess_and_save_dataset(input_dir, output_dir, enable_hair_removal=True, 
                                enable_inpainting=True, enable_contrast=True, 
                                preserve_structure=True):
    """
    Preprocess all images in a dataset and save to disk.
    
    This allows expensive preprocessing (hair removal, inpainting, CLAHE) 
    to be done ONCE instead of on-the-fly during training.
    
    Args:
        input_dir: Root directory with class subdirectories (e.g., 'train/')
        output_dir: Output directory (will mirror structure)
        enable_hair_removal: Apply hair removal
        enable_inpainting: Apply black corner inpainting  
        enable_contrast: Apply CLAHE
        preserve_structure: Keep original directory structure
    
    Example:
        >>> preprocess_and_save_dataset(
        ...     'HAM10000_binary/train',
        ...     'HAM10000_binary/train_preprocessed'
        ... )
    """
    import os
    from pathlib import Path
    from tqdm import tqdm
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Build preprocessing pipeline
    preprocessing = []
    if enable_inpainting:
        preprocessing.append(InpaintingMaskFiller(threshold=50, inpaint_radius=15, min_black_ratio=0.001))
    if enable_hair_removal:
        preprocessing.append(HairRemoval(kernel_size=17))
    if enable_contrast:
        preprocessing.append(ContrastEnhancement(clip_limit=2.0))
    
    print("="*80)
    print("BATCH PREPROCESSING DATASET")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Pipeline: Inpainting={enable_inpainting}, Hair={enable_hair_removal}, Contrast={enable_contrast}")
    print("="*80)
    
    # Find all image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    if preserve_structure:
        # Process directory structure (e.g., train/benign, train/malignant)
        all_images = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    img_path = Path(root) / file
                    rel_path = img_path.relative_to(input_path)
                    all_images.append((img_path, rel_path))
    else:
        # Flat structure
        all_images = [(p, p.name) for p in input_path.rglob('*') 
                     if p.suffix.lower() in image_extensions]
    
    print(f"\nFound {len(all_images)} images to process")
    
    # Process all images
    processed = 0
    skipped = 0
    
    for img_path, rel_path in tqdm(all_images, desc="Processing"):
        try:
            # Create output directory
            out_file = output_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Skip if already exists
            if out_file.exists():
                skipped += 1
                continue
            
            # Load and preprocess
            img = Image.open(img_path).convert('RGB')
            
            for step in preprocessing:
                img = step(img)
            
            # Save preprocessed image
            img.save(out_file, quality=95)
            processed += 1
            
        except Exception as e:
            print(f"\n⚠ Error processing {img_path}: {e}")
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE")
    print("="*80)
    print(f"✓ Processed: {processed} images")
    if skipped > 0:
        print(f"⊘ Skipped: {skipped} images (already exist)")
    print(f"✓ Saved to: {output_dir}")
    print("="*80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("="*80)
    print("MEDICAL IMAGE PREPROCESSING - FINAL VERSION")
    print("="*80)
    
    print("\n📦 Components:")
    print("  1. InpaintingMaskFiller - Removes black corner bias")
    print("  2. HairRemoval - Removes hair artifacts")
    print("  3. ContrastEnhancement - CLAHE for better boundaries")
    
    print("\n🔧 Transform Functions:")
    print("  • get_train_transforms() - Training with preprocessing + flips")
    print("  • get_test_transforms() - Testing with preprocessing only")
    
    print("\n🎨 Visualization:")
    print("  • visualize_preprocessing(img_path) - Shows all 4 steps")
    
    print("\n📊 Usage in Notebook:")
    print("  from medical_preprocessing_final import get_train_transforms, get_test_transforms")
    print("  ")
    print("  train_transforms = get_train_transforms(img_size=224)")
    print("  test_transforms = get_test_transforms(img_size=224)")
    
    print("\n✓ All preprocessing consolidated into single file!")
    print("="*80)
