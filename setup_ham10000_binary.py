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
Setup HAM10000 dataset for binary classification (malignant vs benign)
Downloads from Kaggle and creates proper binary splits
"""

import os
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def setup_ham10000_binary(data_dir='HAM10000', output_dir='HAM10000_binary'):
    """
    Convert HAM10000 multi-class to binary classification
    
    Malignant: mel, bcc (+ optionally akiec)
    Benign: nv, bkl, df, vasc
    """
    
    print("=" * 80)
    print("SETTING UP HAM10000 FOR BINARY CLASSIFICATION")
    print("=" * 80)
    
    # Check if source data exists
    if not os.path.exists(data_dir):
        print("\n❌ HAM10000 data not found!")
        print("\nTo download HAM10000:")
        print("1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("2. Download the dataset")
        print("3. Extract to 'HAM10000/' directory")
        print("\nExpected structure:")
        print("HAM10000/")
        print("  ├── HAM10000_images_part_1/")
        print("  ├── HAM10000_images_part_2/")
        print("  └── HAM10000_metadata.csv")
        return
    
    # Read metadata
    metadata_path = Path(data_dir) / 'HAM10000_metadata.csv'
    if not metadata_path.exists():
        print(f"❌ Metadata file not found: {metadata_path}")
        return
    
    df = pd.read_csv(metadata_path)
    
    print(f"\nOriginal HAM10000 statistics:")
    print(f"Total images: {len(df)}")
    print(f"\nClass distribution:")
    print(df['dx'].value_counts())
    
    # Define binary mapping
    malignant_classes = ['mel', 'bcc']  # melanoma, basal cell carcinoma
    benign_classes = ['nv', 'bkl', 'df', 'vasc']  # benign classes
    # Note: 'akiec' (Actinic Keratoses) is ambiguous - can be added to either
    
    # Add binary label
    df['binary_label'] = df['dx'].apply(
        lambda x: 'malignant' if x in malignant_classes else 'benign'
    )
    
    # Count binary classes
    binary_counts = df['binary_label'].value_counts()
    print(f"\n{'=' * 80}")
    print("BINARY CLASSIFICATION MAPPING")
    print("=" * 80)
    print(f"\nMalignant ({', '.join(malignant_classes)}): {binary_counts.get('malignant', 0)}")
    print(f"Benign ({', '.join(benign_classes)}): {binary_counts.get('benign', 0)}")
    print(f"Class ratio (benign:malignant): {binary_counts['benign'] / binary_counts['malignant']:.2f}:1")
    
    # Use official train/test split (if available) or create 80/20 split
    if 'split' in df.columns:
        print(f"\n✓ Using official train/test split from metadata")
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']
    else:
        print(f"\n⚠️ No official split found, creating 80/20 random split...")
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, 
            test_size=0.2, 
            stratify=df['binary_label'],
            random_state=42
        )
    
    print(f"\nTrain set: {len(train_df)} images")
    print(f"  Benign: {(train_df['binary_label'] == 'benign').sum()}")
    print(f"  Malignant: {(train_df['binary_label'] == 'malignant').sum()}")
    
    print(f"\nTest set: {len(test_df)} images")
    print(f"  Benign: {(test_df['binary_label'] == 'benign').sum()}")
    print(f"  Malignant: {(test_df['binary_label'] == 'malignant').sum()}")
    
    # Create output directories
    output_path = Path(output_dir)
    for split in ['train', 'test']:
        for label in ['benign', 'malignant']:
            (output_path / split / label).mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_dirs = [
        Path(data_dir) / 'HAM10000_images_part_1',
        Path(data_dir) / 'HAM10000_images_part_2'
    ]
    
    # Build image path lookup
    image_paths = {}
    for img_dir in image_dirs:
        if img_dir.exists():
            for img_path in img_dir.glob('*.jpg'):
                image_id = img_path.stem
                image_paths[image_id] = img_path
    
    print(f"\n{'=' * 80}")
    print("COPYING IMAGES")
    print("=" * 80)
    
    # Copy training images
    print(f"\nCopying training images...")
    for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
        image_id = row['image_id']
        label = row['binary_label']
        
        if image_id in image_paths:
            src = image_paths[image_id]
            dst = output_path / 'train' / label / f"{image_id}.jpg"
            shutil.copy2(src, dst)
    
    # Copy test images
    print(f"\nCopying test images...")
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        image_id = row['image_id']
        label = row['binary_label']
        
        if image_id in image_paths:
            src = image_paths[image_id]
            dst = output_path / 'test' / label / f"{image_id}.jpg"
            shutil.copy2(src, dst)
    
    print(f"\n{'=' * 80}")
    print("✓ SETUP COMPLETE")
    print("=" * 80)
    print(f"\nBinary dataset created in: {output_dir}/")
    print(f"\nDirectory structure:")
    print(f"{output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── benign/")
    print(f"  │   └── malignant/")
    print(f"  └── test/")
    print(f"      ├── benign/")
    print(f"      └── malignant/")
    
    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print("=" * 80)
    print(f"\n1. Update your notebook to use: {output_dir}/")
    print(f"2. Retrain models on clean, official data")
    print(f"3. Evaluate with confidence - no data leakage!")
    
    # Save metadata
    train_df.to_csv(output_path / 'train_metadata.csv', index=False)
    test_df.to_csv(output_path / 'test_metadata.csv', index=False)
    print(f"\n✓ Saved metadata to {output_dir}/")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HAM10000 BINARY CLASSIFICATION SETUP")
    print("=" * 80)
    
    print("\nThis will:")
    print("1. Convert HAM10000 7-class to binary (malignant vs benign)")
    print("2. Use official train/test splits (no contamination)")
    print("3. Create clean directory structure for training")
    
    print("\nBinary classification mapping:")
    print("  Malignant: melanoma (mel), basal cell carcinoma (bcc)")
    print("  Benign: nevi (nv), keratosis (bkl), dermatofibroma (df), vascular (vasc)")
    
    # Check if HAM10000 exists
    if not os.path.exists('HAM10000'):
        print("\n❌ HAM10000 data not found")
        print("\nDownload from: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("Extract to 'HAM10000/' directory")
        exit(1)
    
    choice = input("\nProceed? [y/n]: ").lower()
    
    if choice == 'y':
        setup_ham10000_binary(
            data_dir='HAM10000',
            output_dir='HAM10000_binary'
        )
    else:
        print("\n❌ Cancelled")
