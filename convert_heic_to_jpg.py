"""
HEIC to JPG Converter & Renamer
Converts all HEIC images from iPhone to JPG format for use with OpenCV/TensorFlow
Renames files to good1.jpg, good2.jpg, bad1.jpg, etc.
Deletes original HEIC files after conversion
"""

import os
from pathlib import Path
import re

try:
    from pillow_heif import register_heif_opener
    from PIL import Image
    register_heif_opener()
except ImportError:
    print("Required packages not installed. Installing now...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'pillow-heif', 'Pillow'])
    from pillow_heif import register_heif_opener
    from PIL import Image
    register_heif_opener()


def rename_jpg_files(source_dir):
    """
    Rename all JPG files in class folders to sequential names (good1.jpg, bad1.jpg, etc.)
    """
    source_path = Path(source_dir)
    class_folders = ["Good", "Bad", "Ugly"]
    
    for class_name in class_folders:
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"Folder not found: {class_dir}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {class_name} folder...")
        print(f"{'='*50}")
        
        # Get all jpg files - use set to remove duplicates from OneDrive sync
        all_jpgs = set(class_dir.glob("*.jpg")) | set(class_dir.glob("*.JPG"))
        
        # Separate already-renamed files from IMG_* files
        already_renamed = []
        to_rename = []
        
        pattern = re.compile(rf"^{class_name.lower()}\d+\.jpg$", re.IGNORECASE)
        
        for jpg in all_jpgs:
            if pattern.match(jpg.name):
                # Extract number from filename
                num = int(re.search(r'\d+', jpg.name).group())
                already_renamed.append((num, jpg))
            else:
                to_rename.append(jpg)
        
        print(f"  Already renamed: {len(already_renamed)}")
        print(f"  Need to rename: {len(to_rename)}")
        
        if not to_rename:
            continue
        
        # Find the max number used
        if already_renamed:
            max_num = max(num for num, _ in already_renamed)
        else:
            max_num = 0
        
        # Rename remaining files
        for jpg in sorted(to_rename):
            # Check if file still exists (OneDrive sync can cause issues)
            if not jpg.exists():
                print(f"  Skipping (already moved): {jpg.name}")
                continue
                
            max_num += 1
            new_name = f"{class_name.lower()}{max_num}.jpg"
            new_path = class_dir / new_name
            
            # Skip if target already exists
            if new_path.exists():
                print(f"  Skipping (target exists): {new_name}")
                continue
            
            try:
                print(f"  Renaming: {jpg.name} -> {new_name}")
                jpg.rename(new_path)
            except Exception as e:
                print(f"  Error renaming {jpg.name}: {e}")
        
        # Count final files
        final_count = len(list(class_dir.glob(f"{class_name.lower()}*.jpg")))
        print(f"  Final count: {final_count} files")


def convert_and_rename_heic_to_jpg(source_dir, delete_originals=True):
    """
    Convert all HEIC files in source_dir to JPG format with sequential naming.
    Files are renamed based on their parent folder (good1.jpg, bad1.jpg, ugly1.jpg, etc.)
    """
    source_path = Path(source_dir)
    class_folders = ["Good", "Bad", "Ugly"]
    
    total_converted = 0
    total_deleted = 0
    total_errors = 0
    
    for class_name in class_folders:
        class_dir = source_path / class_name
        if not class_dir.exists():
            print(f"Folder not found: {class_dir}")
            continue
        
        print(f"\n{'='*50}")
        print(f"Processing {class_name} folder...")
        print(f"{'='*50}")
        
        # Find all HEIC files in this class folder
        heic_files = sorted(list(class_dir.glob("*.HEIC")) + list(class_dir.glob("*.heic")))
        
        if not heic_files:
            print(f"No HEIC files found in {class_name}")
            continue
        
        print(f"Found {len(heic_files)} HEIC files")
        
        # Find existing renamed files to get starting number
        existing = list(class_dir.glob(f"{class_name.lower()}*.jpg"))
        if existing:
            nums = [int(re.search(r'\d+', f.name).group()) for f in existing if re.search(r'\d+', f.name)]
            next_num = max(nums) + 1 if nums else 1
        else:
            next_num = 1
        
        # Convert and rename
        for heic_file in heic_files:
            new_name = f"{class_name.lower()}{next_num}.jpg"
            jpg_path = class_dir / new_name
            
            try:
                print(f"Converting: {heic_file.name} -> {new_name}")
                img = Image.open(heic_file)
                img = img.convert('RGB')
                img.save(jpg_path, 'JPEG', quality=95)
                total_converted += 1
                next_num += 1
                
                # Delete original HEIC file
                if delete_originals:
                    heic_file.unlink()
                    total_deleted += 1
                    
            except Exception as e:
                print(f"Error converting {heic_file.name}: {e}")
                total_errors += 1
    
    print(f"\n{'='*50}")
    print("CONVERSION SUMMARY")
    print(f"{'='*50}")
    print(f"  Converted: {total_converted}")
    print(f"  HEIC files deleted: {total_deleted}")
    print(f"  Errors: {total_errors}")
    
    return total_converted, total_deleted, total_errors


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    training_data_dir = script_dir / "Training_Data"
    
    print(f"Processing files in: {training_data_dir}")
    print("-" * 50)
    
    if not training_data_dir.exists():
        print(f"Error: Training_Data directory not found at {training_data_dir}")
    else:
        # First convert any remaining HEIC files
        convert_and_rename_heic_to_jpg(training_data_dir, delete_originals=True)
        
        # Then rename any IMG_*.jpg files that weren't renamed
        print("\n\n" + "="*50)
        print("RENAMING REMAINING JPG FILES")
        print("="*50)
        rename_jpg_files(training_data_dir)
