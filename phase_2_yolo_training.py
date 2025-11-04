# SmartVision Phase 2: YOLO v8 Object Detection Training
# Trains YOLO v8 to detect: Traffic Lights, Vehicles, Bicycles

import os
import cv2
import numpy as np
import yaml
from pathlib import Path
import shutil

# ============================================================
# INSTALLATION REMINDER
# ============================================================
# Run these commands first:
# pip install ultralytics opencv-python pillow pyyaml

# ============================================================
# CONFIGURATION
# ============================================================

BASE_PROJECT_PATH = "./SmartVision"
PHASE_1_PATH = os.path.join(BASE_PROJECT_PATH, "phase_1_object_detection")
PHASE_2_PATH = os.path.join(BASE_PROJECT_PATH, "phase_2_yolo_training")

OBJECT_CLASSES = {
    'traffic_light_red': 0,
    'traffic_light_green': 1,
    'traffic_light_yellow': 2,
    'vehicle': 3,
    'bicycle': 4
}

# YOLO training parameters
YOLO_PARAMS = {
    'model': 'yolov8n.pt',  # nano model (fastest, smallest)
    'epochs': 50,  # Full training
    'imgsz': 416,  # Full resolution
    'batch': 16,  # Balanced for GPU
    'device': 0,  # GPU device (0 = first GPU)
    'patience': 20,
    'save': True,
    'conf': 0.5,
    'workers': 4,  # Parallel data loading
    'cache': True,  # Cache images in memory (GPU available)
    'augment': True,  # Enable augmentation
    'plots': True  # Generate plots
}

# ============================================================
# YOLO DATASET MANAGER CLASS
# ============================================================

class YOLODatasetManager:
    """Manages YOLO dataset preparation and conversion"""
    
    def __init__(self, phase1_path, phase2_path, object_classes):
        self.phase1_path = phase1_path
        self.phase2_path = phase2_path
        self.object_classes = object_classes
        
        os.makedirs(phase2_path, exist_ok=True)
        print("✓ YOLO Dataset Manager initialized")
    
    def create_yolo_structure(self):
        """Create YOLO-compatible directory structure"""
        print("\nCreating YOLO directory structure...")
        
        # Create YOLO dataset folder
        yolo_dataset_path = os.path.join(self.phase2_path, "yolo_dataset")
        
        for split in ['train', 'val']:
            images_path = os.path.join(yolo_dataset_path, split, "images")
            labels_path = os.path.join(yolo_dataset_path, split, "labels")
            
            os.makedirs(images_path, exist_ok=True)
            os.makedirs(labels_path, exist_ok=True)
        
        print(f"✓ YOLO structure created at: {yolo_dataset_path}")
        return yolo_dataset_path
    
    def copy_and_organize_images(self, yolo_dataset_path):
        """Copy images from Phase 1 to YOLO format"""
        print("\nOrganizing images for YOLO training...")
        
        # Copy train images
        train_src = os.path.join(self.phase1_path, "train", "images")
        train_dst = os.path.join(yolo_dataset_path, "train", "images")
        
        if os.path.exists(train_src):
            for class_name in os.listdir(train_src):
                class_path = os.path.join(train_src, class_name)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            src_file = os.path.join(class_path, img_file)
                            dst_file = os.path.join(train_dst, img_file)
                            shutil.copy2(src_file, dst_file)
            print(f"✓ Training images copied: {len(os.listdir(train_dst))} images")
        
        # Copy test images (used as validation in YOLO)
        test_src = os.path.join(self.phase1_path, "test", "images")
        val_dst = os.path.join(yolo_dataset_path, "val", "images")
        
        if os.path.exists(test_src):
            for class_name in os.listdir(test_src):
                class_path = os.path.join(test_src, class_name)
                if os.path.isdir(class_path):
                    for img_file in os.listdir(class_path):
                        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            src_file = os.path.join(class_path, img_file)
                            dst_file = os.path.join(val_dst, img_file)
                            shutil.copy2(src_file, dst_file)
            print(f"✓ Validation images copied: {len(os.listdir(val_dst))} images")
    
    def create_dummy_annotations(self, yolo_dataset_path):
        """
        Create dummy YOLO annotations if not available from Phase 1
        In production, these should come from actual labeling
        """
        print("\nCreating placeholder annotations...")
        
        for split in ['train', 'val']:
            images_path = os.path.join(yolo_dataset_path, split, "images")
            labels_path = os.path.join(yolo_dataset_path, split, "labels")
            
            if os.path.exists(images_path):
                for img_file in os.listdir(images_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Create corresponding label file
                        label_file = os.path.splitext(img_file)[0] + '.txt'
                        label_path = os.path.join(labels_path, label_file)
                        
                        # Create dummy annotation (class_id, normalized bbox)
                        # Format: <class_id> <x_center> <y_center> <width> <height>
                        with open(label_path, 'w') as f:
                            # Add a dummy annotation
                            class_id = np.random.randint(0, len(self.object_classes))
                            x_center = np.random.uniform(0.2, 0.8)
                            y_center = np.random.uniform(0.2, 0.8)
                            width = np.random.uniform(0.1, 0.5)
                            height = np.random.uniform(0.1, 0.5)
                            
                            f.write(f"{class_id} {x_center:.3f} {y_center:.3f} {width:.3f} {height:.3f}\n")
        
        print("⚠ Placeholder annotations created")
        print("⚠ IMPORTANT: Use LabelImg to create REAL annotations!")
    
    def create_data_yaml(self, yolo_dataset_path):
        """Create YOLO data.yaml configuration file"""
        print("\nCreating data.yaml configuration...")
        
        data_yaml_path = os.path.join(yolo_dataset_path, "data.yaml")
        
        # Create YAML content
        data_yaml = {
            'path': os.path.abspath(yolo_dataset_path),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.object_classes),
            'names': list(self.object_classes.keys())
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        print(f"✓ data.yaml created at: {data_yaml_path}")
        return data_yaml_path
    
    def get_dataset_statistics(self, yolo_dataset_path):
        """Display dataset statistics"""
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60 + "\n")
        
        for split in ['train', 'val']:
            images_path = os.path.join(yolo_dataset_path, split, "images")
            labels_path = os.path.join(yolo_dataset_path, split, "labels")
            
            if os.path.exists(images_path):
                num_images = len([f for f in os.listdir(images_path) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                num_labels = len([f for f in os.listdir(labels_path) 
                                if f.lower().endswith('.txt')])
                
                print(f"{split.upper()} SET:")
                print(f"  Images: {num_images}")
                print(f"  Annotations: {num_labels}")
                print()


# ============================================================
# YOLO TRAINER CLASS
# ============================================================

class YOLOTrainer:
    """Handles YOLO model training"""
    
    def __init__(self, yolo_params):
        self.yolo_params = yolo_params
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
            print("✓ YOLO module loaded successfully")
        except ImportError:
            print("✗ ultralytics not installed. Run: pip install ultralytics")
            self.YOLO = None
    
    def train_model(self, data_yaml_path, output_dir):
        """Train YOLO model"""
        if self.YOLO is None:
            print("✗ Cannot train: YOLO module not available")
            return None
        
        print("\n" + "="*60)
        print("TRAINING YOLO v8")
        print("="*60 + "\n")
        
        try:
            # Load pretrained YOLO model
            model = self.YOLO(self.yolo_params['model'])
            
            # Train model
            results = model.train(
                data=data_yaml_path,
                epochs=self.yolo_params['epochs'],
                imgsz=self.yolo_params['imgsz'],
                batch=self.yolo_params['batch'],
                device=self.yolo_params['device'],
                patience=self.yolo_params['patience'],
                project=output_dir,
                name='smartvision_detector',
                save=self.yolo_params['save'],
                verbose=True
            )
            
            print("\n✓ Training completed successfully!")
            return model
        
        except Exception as e:
            print(f"✗ Training error: {str(e)}")
            return None
    
    def evaluate_model(self, model, data_yaml_path):
        """Evaluate trained model"""
        if model is None:
            print("✗ No model to evaluate")
            return
        
        print("\n" + "="*60)
        print("EVALUATING MODEL")
        print("="*60 + "\n")
        
        try:
            metrics = model.val()
            print("\n✓ Model evaluation completed!")
            return metrics
        except Exception as e:
            print(f"✗ Evaluation error: {str(e)}")
    
    def predict_on_image(self, model, image_path, conf_threshold=0.5):
        """Run inference on a single image"""
        if model is None:
            print("✗ No model for inference")
            return None
        
        try:
            results = model.predict(
                source=image_path,
                conf=conf_threshold,
                save=False
            )
            return results[0]
        except Exception as e:
            print(f"✗ Prediction error: {str(e)}")
            return None


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("\n" + "="*70)
    print("SmartVision Phase 2 - YOLO v8 Object Detection Training")
    print("="*70 + "\n")
    
    # Step 1: Initialize dataset manager
    print("STEP 1: Initializing Dataset Manager")
    print("-" * 70)
    dataset_manager = YOLODatasetManager(PHASE_1_PATH, PHASE_2_PATH, OBJECT_CLASSES)
    
    # Step 2: Create YOLO directory structure
    print("\nSTEP 2: Creating YOLO Directory Structure")
    print("-" * 70)
    yolo_dataset_path = dataset_manager.create_yolo_structure()
    
    # Step 3: Organize images
    print("\nSTEP 3: Organizing Images")
    print("-" * 70)
    dataset_manager.copy_and_organize_images(yolo_dataset_path)
    
    # Step 4: Create annotations
    print("\nSTEP 4: Handling Annotations")
    print("-" * 70)
    print("Checking for annotations from Phase 1...")
    
    annotations_exist = False
    train_annot_path = os.path.join(PHASE_1_PATH, "train", "annotations")
    if os.path.exists(train_annot_path) and len(os.listdir(train_annot_path)) > 0:
        annotations_exist = True
        print("✓ Annotations found from Phase 1")
    else:
        print("⚠ No annotations found. Creating placeholders...")
        dataset_manager.create_dummy_annotations(yolo_dataset_path)
    
    # Step 5: Create data.yaml
    print("\nSTEP 5: Creating YOLO Configuration")
    print("-" * 70)
    data_yaml_path = dataset_manager.create_data_yaml(yolo_dataset_path)
    
    # Step 6: Display statistics
    print("\nSTEP 6: Dataset Statistics")
    print("-" * 70)
    dataset_manager.get_dataset_statistics(yolo_dataset_path)
    
    # Step 7: Train model
    print("\nSTEP 7: Training YOLO Model")
    print("-" * 70)
    
    if not annotations_exist:
        print("\n⚠ WARNING: Training with placeholder annotations!")
        print("For accurate results, please:")
        print("  1. Use LabelImg to annotate images in YOLO format")
        print("  2. Save .txt files in: train/annotations/ and test/annotations/")
        print("  3. Run this script again\n")
    
    print("YOLO Training Parameters:")
    print(f"  Model: {YOLO_PARAMS['model']}")
    print(f"  Epochs: {YOLO_PARAMS['epochs']}")
    print(f"  Batch Size: {YOLO_PARAMS['batch']}")
    print(f"  Image Size: {YOLO_PARAMS['imgsz']}")
    print(f"  Device: {YOLO_PARAMS['device'].upper() if isinstance(YOLO_PARAMS['device'], str) else 'GPU'}\n")
    
    trainer = YOLOTrainer(YOLO_PARAMS)
    model = trainer.train_model(data_yaml_path, PHASE_2_PATH)
    
    # Step 8: Evaluate model
    if model is not None:
        print("\nSTEP 8: Evaluating Model")
        print("-" * 70)
        trainer.evaluate_model(model, data_yaml_path)
    
    # Summary
    print("\n" + "="*70)
    print("✓ PHASE 2 COMPLETE!")
    print("="*70)
    print(f"\nTraining outputs saved to: {PHASE_2_PATH}")
    print(f"\nNext Steps:")
    print(f"  1. Review training results in: {os.path.join(PHASE_2_PATH, 'smartvision_detector')}")
    print(f"  2. Proceed to Phase 3: Real-time detection with Streamlit")
    print(f"\nDetection Classes:")
    for class_name, class_id in OBJECT_CLASSES.items():
        print(f"  {class_id}: {class_name}")
    print()


if __name__ == "__main__":
    main()