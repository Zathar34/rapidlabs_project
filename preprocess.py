import os
import cv2
import numpy as np
from pathlib import Path

class DatasetPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.target_width = 1920
        self.target_height = 1080

    def analyze_box_sizes(self, subset='train'):
        """Analyze current bounding box sizes in the dataset"""
        labels_dir = self.dataset_path / subset / 'labels'
        widths = []
        heights = []
        
        for label_file in os.listdir(labels_dir):
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:  # class_id, x, y, w, h
                        widths.append(float(parts[3]))
                        heights.append(float(parts[4]))
        
        mean_width = np.mean(widths)
        mean_height = np.mean(heights)
        std_width = np.std(widths)
        std_height = np.std(heights)
        
        print(f"\nCurrent box size statistics:")
        print(f"Width stats: mean={mean_width:.4f}, std={std_width:.4f}")
        print(f"Height stats: mean={mean_height:.4f}, std={std_height:.4f}")
        
        return mean_width, mean_height

    def standardize_boxes(self, subset, standard_width, standard_height):
        """Standardize bounding boxes to given dimensions"""
        labels_dir = self.dataset_path / subset / 'labels'
        if not labels_dir.exists():
            return

        # Process each label file
        for label_file in labels_dir.glob('*.txt'):
            standardized_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = parts[0]
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        standardized_line = f"{class_id} {x_center:.6f} {y_center:.6f} {standard_width:.6f} {standard_height:.6f}\n"
                        standardized_lines.append(standardized_line)
            
            # Overwrite original file with standardized annotations
            with open(label_file, 'w') as f:
                f.writelines(standardized_lines)

    def transform_images(self, subset):
        """Transform images to target resolution"""
        images_path = self.dataset_path / subset / 'images'
        if not images_path.exists():
            return

        # Process images
        for img_path in images_path.glob('*.*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Resize image to target dimensions
                    resized_img = cv2.resize(img, (self.target_width, self.target_height), 
                                           interpolation=cv2.INTER_LINEAR)
                    # Overwrite original image
                    cv2.imwrite(str(img_path), resized_img)
                else:
                    print(f"Failed to load image: {img_path}")

    def process_dataset(self):
        """Process entire dataset"""
        print("Starting dataset preprocessing...")
        
        # First analyze the box sizes
        mean_width, mean_height = self.analyze_box_sizes('train')
        
        # Process each subset
        for subset in ['train', 'test', 'valid']:
            subset_path = self.dataset_path / subset
            if not subset_path.exists():
                continue
                
            print(f"\nProcessing {subset} set:")
            # First standardize boxes
            print(f"- Standardizing bounding boxes...")
            self.standardize_boxes(subset, mean_width, mean_height)
            
            # Then transform images
            print(f"- Transforming images to {self.target_width}x{self.target_height}...")
            self.transform_images(subset)
        
        # Final analysis to confirm changes
        print("\nFinal box size verification:")
        self.analyze_box_sizes('train')
        
        print("\nDataset preprocessing completed!")
        print(f"- All images transformed to {self.target_width}x{self.target_height}")
        print(f"- All bounding boxes standardized to {mean_width:.4f}x{mean_height:.4f}")

def main():
    dataset_path = "AllHockey.v1i.yolov11"
    preprocessor = DatasetPreprocessor(dataset_path)
    preprocessor.process_dataset()

if __name__ == "__main__":
    main()