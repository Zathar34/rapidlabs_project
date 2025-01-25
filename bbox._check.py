import os
import numpy as np

def analyze_box_sizes(labels_dir):
    widths = []
    heights = []
    
    for label_file in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:  # class_id, x, y, w, h
                    widths.append(float(parts[3]))
                    heights.append(float(parts[4]))
    
    print(f"Width stats: mean={np.mean(widths):.4f}, std={np.std(widths):.4f}")
    print(f"Height stats: mean={np.mean(heights):.4f}, std={np.std(heights):.4f}")

analyze_box_sizes('AllHockey.v1i.yolov11/train/labels')