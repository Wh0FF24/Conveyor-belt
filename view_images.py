"""
Quick image viewer to check training images
"""
import cv2 as cv
from pathlib import Path

def view_images(folder_path, max_images=5):
    """View images from a folder, properly resized to fit screen"""
    folder = Path(folder_path)
    jpg_files = sorted(list(folder.glob("*.jpg")))[:max_images]
    
    print(f"Viewing {len(jpg_files)} images from {folder.name}")
    print("Press any key to continue, 'q' to quit")
    
    for img_path in jpg_files:
        img = cv.imread(str(img_path))
        if img is None:
            print(f"Could not load: {img_path.name}")
            continue
        
        # Get original dimensions
        h, w = img.shape[:2]
        print(f"\n{img_path.name}: Original size {w}x{h}")
        
        # Resize to fit screen (max 800 height)
        max_height = 800
        if h > max_height:
            scale = max_height / h
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_resized = cv.resize(img, (new_w, new_h))
        else:
            img_resized = img
        
        cv.imshow(f"{folder.name} - {img_path.name}", img_resized)
        key = cv.waitKey(0)
        cv.destroyAllWindows()
        
        if key == ord('q'):
            break


if __name__ == "__main__":
    base_dir = Path(__file__).parent / "Training_Data"
    
    print("="*50)
    print("VIEWING GOOD IMAGES")
    print("="*50)
    view_images(base_dir / "Good", max_images=3)
    
    print("\n" + "="*50)
    print("VIEWING BAD IMAGES")
    print("="*50)
    view_images(base_dir / "Bad", max_images=3)
    
    print("\n" + "="*50)
    print("VIEWING UGLY IMAGES")
    print("="*50)
    view_images(base_dir / "Ugly", max_images=3)
