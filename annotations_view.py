import os
import cv2
import random
import time

def visualize_annotations(image_path, label_path, class_names=None):
    """
    Visualize YOLO format annotations on an image
    Args:
        image_path (str): Path to the image file
        label_path (str): Path to the annotation file (.txt)
        class_names (list): List of class names (optional)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return

    img_height, img_width = image.shape[:2]

    # Read the annotations
    with open(label_path, 'r') as f:
        annotations = f.readlines()

    # Define colors for different classes
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]  # Green, Red, Blue

    for ann in annotations:
        # Parse YOLO format annotation
        parts = ann.strip().split()
        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height

        # Convert to rectangle coordinates
        x_min = int(x_center - width/2)
        y_min = int(y_center - height/2)
        x_max = int(x_center + width/2)
        y_max = int(y_center + height/2)

        # Get color for class
        color = colors[class_id % len(colors)]

        # Draw bounding box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

        # Add class label
        label = class_names[class_id] if class_names else f"Class {class_id}"
        cv2.putText(image, label, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the image
    cv2.imshow("Annotation Visualization", image)
    return image

# Example usage
if __name__ == "__main__":
    # Path configuration (modify these according to your dataset structure)
    dataset_base = "AllHockey.v1i.yolov11"
    image_dir = os.path.join(dataset_base, "train", "images")
    label_dir = os.path.join(dataset_base, "train", "labels")

    # Get list of images
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in the directory")
    else:
        # Define class names (modify according to your classes)
        class_names = ["puck"]  # Assuming class 0 is puck

        # Loop through all images
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + ".txt")

            # Visualize annotations
            visualize_annotations(image_path, label_path, class_names)

            # Wait for 2 seconds (2000 ms) or until 'q' is pressed
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                break

        # Close all OpenCV windows
        cv2.destroyAllWindows()