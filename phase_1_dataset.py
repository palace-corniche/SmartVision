# SmartVision Phase 1: Object Detection Dataset Preparation
# VS Code Local Version
# Prepares dataset for detecting: Traffic Lights, Vehicles, Bicycles

import cv2
import os
import numpy as np
from pathlib import Path
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

OBJECT_CLASSES = {
    'traffic_light_red': 0,
    'traffic_light_green': 1,
    'traffic_light_yellow': 2,
    'vehicle': 3,
    'bicycle': 4
}

BASE_PROJECT_PATH = "./SmartVision"
PHASE_1_PATH = os.path.join(BASE_PROJECT_PATH, "phase_1_object_detection")

# ============================================================
# OBJECT DATASET PROCESSOR CLASS
# ============================================================

class ObjectDatasetProcessor:
    """Handles object detection dataset preparation"""
    
    def __init__(self, object_classes, base_path):
        self.object_classes = object_classes
        self.base_path = base_path
        self.class_count = {cls: 0 for cls in object_classes.keys()}
        print("✓ ObjectDatasetProcessor initialized")
    
    def create_directory_structure(self):
        """Create required folder structure"""
        print("\nCreating directory structure...")
        
        # Create folders for raw images
        for class_name in self.object_classes.keys():
            train_class_path = os.path.join(self.base_path, "train", "images", class_name)
            test_class_path = os.path.join(self.base_path, "test", "images", class_name)
            
            os.makedirs(train_class_path, exist_ok=True)
            os.makedirs(test_class_path, exist_ok=True)
        
        # Create annotation folders
        train_annot_path = os.path.join(self.base_path, "train", "annotations")
        test_annot_path = os.path.join(self.base_path, "test", "annotations")
        
        os.makedirs(train_annot_path, exist_ok=True)
        os.makedirs(test_annot_path, exist_ok=True)
        
        print(f"✓ Directory structure created at: {self.base_path}\n")
        self.print_structure()
    
    def print_structure(self):
        """Display the folder structure"""
        print("Project Structure:")
        print(f"{self.base_path}/")
        print("├── train/")
        print("│   ├── images/")
        for cls in self.object_classes.keys():
            print(f"│   │   ├── {cls}/")
        print("│   └── annotations/")
        print("├── test/")
        print("│   ├── images/")
        for cls in self.object_classes.keys():
            print(f"│   │   ├── {cls}/")
        print("│   └── annotations/")
        print("├── class_mapping.txt")
        print("└── DATASET_INFO.txt\n")
    
    def create_class_mapping_file(self):
        """Create a mapping file for class names and IDs"""
        mapping_file = os.path.join(self.base_path, "class_mapping.txt")
        with open(mapping_file, 'w') as f:
            f.write("SmartVision Object Classes\n")
            f.write("=" * 40 + "\n\n")
            for class_name, class_id in self.object_classes.items():
                f.write(f"{class_id}: {class_name}\n")
        print(f"✓ Class mapping saved to: {mapping_file}")
    
    def validate_image(self, image_path):
        """Check if image is valid"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return False, "Cannot read image"
            return True, "Valid"
        except Exception as e:
            return False, str(e)
    
    def get_dataset_statistics(self):
        """Count images per class"""
        stats = {}
        
        for split in ['train', 'test']:
            stats[split] = {}
            images_path = os.path.join(self.base_path, split, "images")
            
            if not os.path.exists(images_path):
                continue
            
            for class_name in self.object_classes.keys():
                class_path = os.path.join(images_path, class_name)
                if os.path.exists(class_path):
                    image_files = [f for f in os.listdir(class_path) 
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    stats[split][class_name] = len(image_files)
                else:
                    stats[split][class_name] = 0
        
        return stats
    
    def validate_dataset(self):
        """Validate all images and annotations"""
        print("\n" + "="*60)
        print("VALIDATING DATASET")
        print("="*60 + "\n")
        
        stats = self.get_dataset_statistics()
        
        print("TRAINING SET:")
        train_total = 0
        for class_name, count in stats.get('train', {}).items():
            print(f"  {class_name}: {count} images")
            train_total += count
        print(f"  TOTAL: {train_total} images\n")
        
        print("TEST SET:")
        test_total = 0
        for class_name, count in stats.get('test', {}).items():
            print(f"  {class_name}: {count} images")
            test_total += count
        print(f"  TOTAL: {test_total} images\n")
        
        print(f"✓ Total dataset size: {train_total + test_total} images\n")
        
        return stats
    
    def load_images_for_training(self, image_size=(416, 416)):
        """Load and prepare images for training"""
        print("\n" + "="*60)
        print("LOADING IMAGES")
        print("="*60 + "\n")
        
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        
        supported_formats = ('.jpg', '.jpeg', '.png', '.bmp')
        
        # Load training data
        print("Loading training images...")
        train_images_path = os.path.join(self.base_path, "train", "images")
        if os.path.exists(train_images_path):
            for class_name in os.listdir(train_images_path):
                class_path = os.path.join(train_images_path, class_name)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(supported_formats):
                            img_path = os.path.join(class_path, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_resized = cv2.resize(img, image_size)
                                train_images.append(img_resized)
                                train_labels.append(self.object_classes[class_name])
        
        # Load test data
        print("Loading test images...")
        test_images_path = os.path.join(self.base_path, "test", "images")
        if os.path.exists(test_images_path):
            for class_name in os.listdir(test_images_path):
                class_path = os.path.join(test_images_path, class_name)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(supported_formats):
                            img_path = os.path.join(class_path, img_file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_resized = cv2.resize(img, image_size)
                                test_images.append(img_resized)
                                test_labels.append(self.object_classes[class_name])
        
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        
        print(f"✓ Training set: {train_images.shape}")
        print(f"✓ Test set: {test_images.shape}\n")
        
        return train_images, train_labels, test_images, test_labels
    
    def save_dataset(self, train_x, train_y, test_x, test_y, filename="smartvision_dataset.npz"):
        """Save dataset to NPZ file"""
        output_path = os.path.join(self.base_path, filename)
        np.savez_compressed(
            output_path,
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y
        )
        print(f"✓ Dataset saved to: {output_path}\n")
        return output_path
    
    def create_dataset_info(self, stats):
        """Create an info file about the dataset"""
        info_file = os.path.join(self.base_path, "DATASET_INFO.txt")
        with open(info_file, 'w') as f:
            f.write("SmartVision Phase 1 - Object Detection Dataset\n")
            f.write("=" * 50 + "\n\n")
            f.write("TRAINING SET:\n")
            for class_name, count in stats.get('train', {}).items():
                f.write(f"  {class_name}: {count} images\n")
            
            f.write(f"\nTotal training images: {sum(stats.get('train', {}).values())}\n\n")
            
            f.write("TEST SET:\n")
            for class_name, count in stats.get('test', {}).items():
                f.write(f"  {class_name}: {count} images\n")
            
            f.write(f"\nTotal test images: {sum(stats.get('test', {}).values())}\n\n")
            
            f.write("ANNOTATION FORMAT (YOLO):\n")
            f.write("  <class_id> <x_center> <y_center> <width> <height>\n")
            f.write("  (normalized values between 0 and 1)\n\n")
            
            f.write("CLASS MAPPING:\n")
            for class_name, class_id in self.object_classes.items():
                f.write(f"  {class_id}: {class_name}\n")
        
        print(f"✓ Dataset info saved to: {info_file}")


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*60)
    print("SmartVision Phase 1 - Object Detection Dataset")
    print("="*60 + "\n")
    
    # Initialize processor
    processor = ObjectDatasetProcessor(OBJECT_CLASSES, PHASE_1_PATH)
    
    # Create directory structure
    processor.create_directory_structure()
    
    # Create mapping file
    processor.create_class_mapping_file()
    
    # Display instructions
    print("="*60)
    print("INSTRUCTIONS")
    print("="*60)
    print("\n1. UPLOAD YOUR IMAGES:")
    print(f"   Navigate to: {PHASE_1_PATH}")
    print("   Upload ~30 images per class to: train/images/<class_name>/")
    print("   Upload ~10 images per class to: test/images/<class_name>/")
    
    print("\n2. ANNOTATE YOUR IMAGES:")
    print("   Use tools like: LabelImg, CVAT, or Roboflow")
    print("   Export in YOLO format (.txt files)")
    print("   Save to: train/annotations/ and test/annotations/")
    
    # Validate and display statistics
    stats = processor.validate_dataset()
    
    # Load images
    train_x, train_y, test_x, test_y = processor.load_images_for_training()
    
    # Save dataset
    processor.save_dataset(train_x, train_y, test_x, test_y)
    
    # Create info file
    processor.create_dataset_info(stats)
    
    print("\n" + "="*60)
    print("✓ PHASE 1 COMPLETE!")
    print("="*60)
    print(f"\nNext: Proceed to Phase 2 for YOLO training")
    print(f"Dataset ready for: Traffic Light, Vehicle, Bicycle detection\n")


if __name__ == "__main__":
    main()