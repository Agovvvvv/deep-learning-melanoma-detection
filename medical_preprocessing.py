"""
Medical Image Preprocessing for Melanoma Detection
===================================================
Addresses Grad-CAM focus issues by:
1. Removing hair artifacts
2. Centering and cropping lesions
3. Enhancing contrast
4. Standardizing image quality
"""

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class HairRemoval:
    """
    Removes hair artifacts from dermoscopic images using morphological operations.
    Hair artifacts can distract the model from focusing on the lesion itself.
    """
    
    def __init__(self, kernel_size=17):
        self.kernel_size = kernel_size
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image with hair removed
        """
        try:
            # Convert PIL to numpy
            img_np = np.array(img)
            
            # Convert to grayscale for hair detection
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Black hat transform to detect dark hair
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size, self.kernel_size))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Threshold to get hair mask
            _, hair_mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
            
            # Inpaint to remove hair
            result = cv2.inpaint(img_np, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            
            return Image.fromarray(result)
        except Exception as e:
            # If fails, return original
            return img


class LesionCropping:
    """
    Intelligently crops the image to focus on the lesion using Otsu thresholding.
    This addresses the Grad-CAM issue where the model focuses on background.
    """
    
    def __init__(self, margin=0.1, min_crop_ratio=0.5):
        """
        Args:
            margin: Extra margin around detected lesion (as ratio of size)
            min_crop_ratio: Minimum crop size (to avoid over-cropping)
        """
        self.margin = margin
        self.min_crop_ratio = min_crop_ratio
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image cropped to lesion with margin
        """
        try:
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Otsu's thresholding to separate lesion from background
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find largest contour (assumed to be lesion)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return img
            
            # Get bounding box of largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest_contour)
            
            # Add margin
            margin_w = int(cw * self.margin)
            margin_h = int(ch * self.margin)
            
            x1 = max(0, x - margin_w)
            y1 = max(0, y - margin_h)
            x2 = min(w, x + cw + margin_w)
            y2 = min(h, y + ch + margin_h)
            
            # Ensure minimum crop size
            crop_w = x2 - x1
            crop_h = y2 - y1
            min_w = int(w * self.min_crop_ratio)
            min_h = int(h * self.min_crop_ratio)
            
            if crop_w < min_w or crop_h < min_h:
                # Crop too small, return original
                return img
            
            # Crop
            cropped = img_np[y1:y2, x1:x2]
            return Image.fromarray(cropped)
            
        except Exception as e:
            # If fails, return original
            return img


class ContrastEnhancement:
    """
    Enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Improves lesion boundary definition which helps model focus.
    """
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image with enhanced contrast
        """
        try:
            img_np = np.array(img)
            
            # Convert to LAB color space
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            l_clahe = clahe.apply(l)
            
            # Merge channels
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # Convert back to RGB
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(result)
        except Exception as e:
            # If fails, return original
            return img


class CenterLesion:
    """
    Centers the lesion in the image using center of mass calculation.
    Addresses translation/shift issues in augmentation.
    """
    
    def __init__(self, output_size=256):
        self.output_size = output_size
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            PIL Image with lesion centered
        """
        try:
            img_np = np.array(img)
            h, w = img_np.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            # Threshold
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Calculate center of mass
            M = cv2.moments(thresh)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                # Use image center
                cx, cy = w // 2, h // 2
            
            # Create canvas
            canvas = np.ones((self.output_size, self.output_size, 3), dtype=np.uint8) * 255
            
            # Calculate paste position
            paste_x = self.output_size // 2 - cx
            paste_y = self.output_size // 2 - cy
            
            # Calculate crop/paste region
            src_x1 = max(0, -paste_x)
            src_y1 = max(0, -paste_y)
            src_x2 = min(w, self.output_size - paste_x)
            src_y2 = min(h, self.output_size - paste_y)
            
            dst_x1 = max(0, paste_x)
            dst_y1 = max(0, paste_y)
            dst_x2 = dst_x1 + (src_x2 - src_x1)
            dst_y2 = dst_y1 + (src_y2 - src_y1)
            
            # Paste
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img_np[src_y1:src_y2, src_x1:src_x2]
            
            return Image.fromarray(canvas)
            
        except Exception as e:
            # If fails, return resized original
            return img.resize((self.output_size, self.output_size))


# ============================================================
# PRESET TRANSFORM PIPELINES
# ============================================================

def get_medical_train_transforms(img_size=224, enable_hair_removal=True, 
                                  enable_lesion_crop=True, enable_contrast=True):
    """
    Creates training transforms with medical preprocessing + conservative augmentation.
    
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
    
    # Standard preprocessing - ENSURE proper size without black borders
    medical_steps.extend([
        transforms.Resize((img_size, img_size)),  # Direct resize (no extra padding)
    ])
    
    # CONSERVATIVE augmentations (less aggressive to keep lesion in view)
    augmentation_steps = [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15, fill=255),  # Reduced, use white fill
        
        # Less aggressive affine
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),  # Reduced from 0.15 - keeps lesion centered
            scale=(0.95, 1.05),       # Even more conservative
            shear=3,                   # Further reduced
            fill=255                   # White fill to match medical images
        ),
        
        # Color augmentation (keeps diversity)
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        
        # Mild blur and perspective
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=0.2),
        
        transforms.RandomPerspective(distortion_scale=0.1, p=0.2),  # Reduced from 0.2
        
        transforms.ToTensor(),
        
        # Less aggressive random erasing - using mean value to avoid colorful artifacts
        transforms.RandomErasing(
            p=0.15,              # Further reduced
            scale=(0.02, 0.06),  # Even smaller patches
            ratio=(0.3, 3.3),
            value=0              # Use 0 (black) instead of 'random' to avoid colorful noise
        ),
        
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    return transforms.Compose(medical_steps + augmentation_steps)


def get_medical_test_transforms(img_size=224, enable_hair_removal=True,
                                 enable_lesion_crop=True, enable_contrast=True):
    """
    Creates test transforms with medical preprocessing (no augmentation).
    
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


# ============================================================
# VISUALIZATION HELPER
# ============================================================

def visualize_preprocessing_steps(img_path):
    """
    Visualizes each preprocessing step for debugging.
    
    Args:
        img_path: Path to input image
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    # Load image
    original = Image.open(img_path).convert('RGB')
    
    # Apply each step
    hair_removed = HairRemoval()(original)
    contrast_enhanced = ContrastEnhancement()(hair_removed)
    
    # Plot - 3 images in a row
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(hair_removed)
    axes[1].set_title('After Hair Removal', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(contrast_enhanced)
    axes[2].set_title('After Contrast Enhancement (CLAHE)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("="*80)
    print("MEDICAL IMAGE PREPROCESSING FOR MELANOMA DETECTION")
    print("="*80)
    print("\nAvailable transforms:")
    print("  • HairRemoval - Removes hair artifacts using morphological operations")
    print("  • ContrastEnhancement - CLAHE for better boundary definition")
    print("\nPreset pipelines:")
    print("  • get_medical_train_transforms() - Training with hair removal + contrast")
    print("  • get_medical_test_transforms() - Testing with hair removal + contrast")
    print("\nActive preprocessing steps:")
    print("  ✓ Hair removal (reduces artifacts)")
    print("  ✓ Contrast enhancement (better lesion boundaries)")
    print("  ✓ Conservative augmentation (flips only)")
    print("="*80)
