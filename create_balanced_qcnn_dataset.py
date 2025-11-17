"""
Script to create QCNN dataset with exactly 3000 samples in train set.
- Train set: 3000 images from train_preprocessed (balanced classes)
- Test set: Remaining train_preprocessed + all test_preprocessed images
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Configuration
SOURCE_DIR = Path(r"c:\Users\nicdu\Projects\Lesions detection\HAM10000_binary")
TRAIN_SOURCE = SOURCE_DIR / "train_preprocessed"
TEST_SOURCE = SOURCE_DIR / "test_preprocessed"

OUTPUT_DIR = SOURCE_DIR
TRAIN_OUTPUT = OUTPUT_DIR / "qcnn_train"
TEST_OUTPUT = OUTPUT_DIR / "qcnn_test"

TRAIN_IMAGES = 3000  # Exactly 3000 images for training
CLASSES = ["benign", "malignant"]

# Set seed for reproducibility
random.seed(42)


def collect_images_by_class(source_dir):
    """Collect all image paths organized by class from a single directory."""
    images_by_class = defaultdict(list)
    
    for class_name in CLASSES:
        class_dir = source_dir / class_name
        if class_dir.exists():
            image_files = list(class_dir.glob("*.jpg")) + \
                         list(class_dir.glob("*.jpeg")) + \
                         list(class_dir.glob("*.png"))
            images_by_class[class_name].extend(image_files)
    
    return images_by_class


def create_splits(train_images_by_class, test_images_by_class, train_total):
    """
    Create splits:
    - Train: exactly train_total images from train_preprocessed (balanced)
    - Test: remaining train_preprocessed + all test_preprocessed
    """
    train_per_class = train_total // len(CLASSES)
    
    train_split = defaultdict(list)
    test_split = defaultdict(list)
    
    for class_name in CLASSES:
        train_available = train_images_by_class[class_name]
        test_available = test_images_by_class[class_name]
        
        # Check if we have enough training images
        if len(train_available) < train_per_class:
            print(f"Warning: Only {len(train_available)} {class_name} images in train_preprocessed, "
                  f"but {train_per_class} requested for balanced training set.")
            train_per_class_actual = len(train_available)
        else:
            train_per_class_actual = train_per_class
        
        # Shuffle training images
        random.shuffle(train_available)
        
        # Split training images: first train_per_class_actual go to train, rest to test
        train_split[class_name] = train_available[:train_per_class_actual]
        remaining_train = train_available[train_per_class_actual:]
        
        # All remaining from train_preprocessed + all from test_preprocessed go to test
        test_split[class_name] = remaining_train + test_available
        
        print(f"{class_name.capitalize()}: {len(train_split[class_name])} train, "
              f"{len(test_split[class_name])} test "
              f"({len(remaining_train)} from train_preprocessed + {len(test_available)} from test_preprocessed)")
    
    return train_split, test_split


def copy_images(split_dict, output_dir, split_name):
    """Copy images to output directory maintaining class structure."""
    for class_name, image_paths in split_dict.items():
        class_output_dir = output_dir / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {len(image_paths)} {class_name} images to {split_name}...")
        for img_path in image_paths:
            dest_path = class_output_dir / img_path.name
            shutil.copy2(img_path, dest_path)
    
    print(f"✓ {split_name} dataset created successfully")


def print_summary(train_split, test_split):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("DATASET CREATION SUMMARY")
    print("="*50)
    
    total_train = sum(len(images) for images in train_split.values())
    total_test = sum(len(images) for images in test_split.values())
    
    print(f"\nTrain Set (from train_preprocessed): {total_train} images")
    for class_name in CLASSES:
        print(f"  - {class_name}: {len(train_split[class_name])}")
    
    print(f"\nTest Set (remaining train + all test): {total_test} images")
    for class_name in CLASSES:
        print(f"  - {class_name}: {len(test_split[class_name])}")
    
    print(f"\nTotal: {total_train + total_test} images")
    print("\nOutput directories:")
    print(f"  - {TRAIN_OUTPUT}")
    print(f"  - {TEST_OUTPUT}")


def main():
    print("="*50)
    print("Creating QCNN Dataset")
    print("="*50)
    print(f"\nTarget: {TRAIN_IMAGES} images in train set (balanced)")
    print(f"Classes: {', '.join(CLASSES)}")
    print("Test set: Remaining train_preprocessed + all test_preprocessed")
    
    # Collect images from train and test separately
    print("\n1. Collecting images from source directories...")
    train_images_by_class = collect_images_by_class(TRAIN_SOURCE)
    test_images_by_class = collect_images_by_class(TEST_SOURCE)
    
    print("\nAvailable images in train_preprocessed:")
    for class_name, images in train_images_by_class.items():
        print(f"  - {class_name}: {len(images)} images")
    
    print("\nAvailable images in test_preprocessed:")
    for class_name, images in test_images_by_class.items():
        print(f"  - {class_name}: {len(images)} images")
    
    # Create splits
    print("\n2. Creating splits...")
    train_split, test_split = create_splits(train_images_by_class, test_images_by_class, TRAIN_IMAGES)
    
    # Create output directories and copy images
    print("\n3. Copying images to output directories...")
    
    # Remove existing directories if they exist
    if TRAIN_OUTPUT.exists():
        print(f"Removing existing directory: {TRAIN_OUTPUT}")
        shutil.rmtree(TRAIN_OUTPUT)
    if TEST_OUTPUT.exists():
        print(f"Removing existing directory: {TEST_OUTPUT}")
        shutil.rmtree(TEST_OUTPUT)
    
    copy_images(train_split, TRAIN_OUTPUT, "qcnn_train")
    copy_images(test_split, TEST_OUTPUT, "qcnn_test")
    
    # Print summary
    print_summary(train_split, test_split)
    
    print("\n✓ Dataset creation completed successfully!")


if __name__ == "__main__":
    main()
