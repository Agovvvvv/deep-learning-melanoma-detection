"""
Batch Preprocess HAM10000 Dataset
==================================
Run this script ONCE to preprocess all images and save them to disk.
This avoids expensive on-the-fly preprocessing during training.

Expected time: ~10-15 minutes for full dataset
Saves: ~3-4 hours per training run!
"""

from medical_preprocessing_final import preprocess_and_save_dataset

if __name__ == "__main__":
    print("\n" + "="*80)
    print("PREPROCESSING HAM10000 DATASET")
    print("="*80)
    print("\nThis will preprocess:")
    print("  • Training images: HAM10000_binary/train")
    print("  • Test images: HAM10000_binary/test")
    print("\nPreprocessing includes:")
    print("  ✓ Black corner removal (inpainting)")
    print("  ✓ Hair removal (morphological)")
    print("  ✓ Contrast enhancement (CLAHE)")
    print("\nExpected time: ~10-15 minutes")
    print("="*80)
    
    response = input("\nProceed with preprocessing? (y/n): ")
    
    if response.lower() != 'y':
        print("Cancelled.")
        exit()
    
    # Preprocess training set
    print("\n" + "="*80)
    print("STEP 1/2: Preprocessing Training Set")
    print("="*80)
    preprocess_and_save_dataset(
        input_dir='HAM10000_binary/train',
        output_dir='HAM10000_binary/train_preprocessed',
        enable_hair_removal=True,
        enable_inpainting=True,
        enable_contrast=True
    )
    
    # Preprocess test set
    print("\n" + "="*80)
    print("STEP 2/2: Preprocessing Test Set")
    print("="*80)
    preprocess_and_save_dataset(
        input_dir='HAM10000_binary/test',
        output_dir='HAM10000_binary/test_preprocessed',
        enable_hair_removal=True,
        enable_inpainting=True,
        enable_contrast=True
    )
    
    print("\n" + "="*80)
    print("✓ ALL PREPROCESSING COMPLETE!")
    print("="*80)
    print("\nPreprocessed datasets saved to:")
    print("  • HAM10000_binary/train_preprocessed/")
    print("  • HAM10000_binary/test_preprocessed/")
    print("\nNext steps:")
    print("  1. Update your notebook to use preprocessed directories")
    print("  2. Use simple transforms (no medical preprocessing)")
    print("  3. Training will be 5-10x faster!")
    print("="*80)
