"""
Improved Lesion Cropping with Better Detection
================================================
Addresses issues with the original LesionCropping that wasn't working.
"""

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class ImprovedLesionCropping:
    """
    More robust lesion detection and cropping.
    
    KEY IMPROVEMENTS:
    1. Better segmentation using multiple methods
    2. Fallback strategies if detection fails
    3. Smaller margin for tighter crops
    4. Validation that cropping actually happened
    """
    
    def __init__(self, margin=0.05, min_crop_ratio=0.3, target_size=None):
        """
        Args:
            margin: Padding around detected lesion (0.05 = 5% padding)
            min_crop_ratio: Minimum crop size relative to image (prevents over-cropping)
            target_size: Optional, resize after cropping (e.g., 224)
        """
        self.margin = margin
        self.min_crop_ratio = min_crop_ratio
        self.target_size = target_size
    
    def __call__(self, img):
        """Apply improved lesion cropping."""
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        # Try multiple detection methods
        bbox = self._detect_lesion_multi_method(img_np)
        
        if bbox is None:
            # Detection failed - return original
            print("⚠️  Lesion detection failed, using original image")
            return Image.fromarray(img_np) if isinstance(img, Image.Image) else img_np
        
        # Crop with margin
        x1, y1, x2, y2 = bbox
        h, w = img_np.shape[:2]
        
        # Add margin
        margin_x = int((x2 - x1) * self.margin)
        margin_y = int((y2 - y1) * self.margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Validate crop size
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        if crop_w < w * self.min_crop_ratio or crop_h < h * self.min_crop_ratio:
            print(f"⚠️  Crop too small ({crop_w}x{crop_h}), using original")
            return Image.fromarray(img_np) if isinstance(img, Image.Image) else img_np
        
        # Perform crop
        cropped = img_np[y1:y2, x1:x2]
        
        # Optional resize
        if self.target_size is not None:
            cropped = cv2.resize(cropped, (self.target_size, self.target_size))
        
        return Image.fromarray(cropped) if isinstance(img, Image.Image) else cropped
    
    def _detect_lesion_multi_method(self, img):
        """
        Try multiple detection methods in order of reliability.
        Returns bounding box (x1, y1, x2, y2) or None if all fail.
        """
        # Method 1: Otsu thresholding on saturation channel
        bbox = self._detect_via_saturation(img)
        if bbox is not None and self._validate_bbox(bbox, img.shape):
            return bbox
        
        # Method 2: Edge detection + contour finding
        bbox = self._detect_via_edges(img)
        if bbox is not None and self._validate_bbox(bbox, img.shape):
            return bbox
        
        # Method 3: Color-based segmentation
        bbox = self._detect_via_color(img)
        if bbox is not None and self._validate_bbox(bbox, img.shape):
            return bbox
        
        return None
    
    def _detect_via_saturation(self, img):
        """Detect lesion using HSV saturation channel (most reliable)."""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            
            # Otsu thresholding
            _, binary = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find largest contour
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            return (x, y, x + w, y + h)
        
        except Exception as e:
            return None
    
    def _detect_via_edges(self, img):
        """Detect lesion using edge detection."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Canny edge detection
            edges = cv2.Canny(blurred, 30, 100)
            
            # Dilate edges to connect nearby contours
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            return (x, y, x + w, y + h)
        
        except Exception as e:
            return None
    
    def _detect_via_color(self, img):
        """Detect lesion based on color differences from background."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            
            # Assume background is lighter (higher L values)
            # Threshold to get darker regions (potential lesion)
            _, binary = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            return (x, y, x + w, y + h)
        
        except Exception as e:
            return None
    
    def _validate_bbox(self, bbox, img_shape):
        """Check if bounding box is reasonable."""
        if bbox is None:
            return False
        
        x1, y1, x2, y2 = bbox
        h, w = img_shape[:2]
        
        # Check if bbox is too small or too large
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h
        img_area = h * w
        
        # Must be between 5% and 95% of image
        area_ratio = bbox_area / img_area
        
        if area_ratio < 0.05 or area_ratio > 0.95:
            return False
        
        # Aspect ratio should be reasonable (not too elongated)
        aspect_ratio = bbox_w / max(bbox_h, 1)
        if aspect_ratio < 0.3 or aspect_ratio > 3.0:
            return False
        
        return True


# ============================================================
# SIMPLE CENTER CROP ALTERNATIVE (More Reliable)
# ============================================================

class CenterCropLesion:
    """
    Simpler alternative: Crop center region where lesions typically are.
    More reliable but less precise than segmentation-based cropping.
    """
    
    def __init__(self, crop_ratio=0.7):
        """
        Args:
            crop_ratio: What fraction of image to keep (0.7 = keep center 70%)
        """
        self.crop_ratio = crop_ratio
    
    def __call__(self, img):
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
        
        h, w = img_np.shape[:2]
        
        # Calculate crop dimensions
        new_h = int(h * self.crop_ratio)
        new_w = int(w * self.crop_ratio)
        
        # Calculate center crop coordinates
        start_y = (h - new_h) // 2
        start_x = (w - new_w) // 2
        
        # Crop
        cropped = img_np[start_y:start_y+new_h, start_x:start_x+new_w]
        
        return Image.fromarray(cropped) if isinstance(img, Image.Image) else cropped


# ============================================================
# USAGE EXAMPLES
# ============================================================

"""
OPTION 1: Improved Lesion Cropping (tries to detect lesion)
------------------------------------------------------------
train_transforms = transforms.Compose([
    HairRemoval(),
    ContrastEnhancement(),
    ImprovedLesionCropping(margin=0.05, min_crop_ratio=0.3),  # ← New
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


OPTION 2: Simple Center Crop (more reliable, less precise)
-----------------------------------------------------------
train_transforms = transforms.Compose([
    HairRemoval(),
    ContrastEnhancement(),
    CenterCropLesion(crop_ratio=0.7),  # ← Keep center 70%
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])


OPTION 3: Skip Cropping (use attention mechanism instead)
----------------------------------------------------------
If cropping continues to be problematic, rely on the attention
mechanism in the model to learn where to focus. This is actually
a valid approach - let the model learn spatial attention rather than
forcing it through preprocessing.
"""
