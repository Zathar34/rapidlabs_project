# Hockey Puck Detection and Tracking

This repository contains code for training and running a hockey puck detection and tracking system using YOLOv8.

## Setup and Training

### 1. Dataset Preparation
- Extract the `AllHockey.v1i.yolov11` dataset into your project directory
- Dataset structure should look like:
  ```
  AllHockey.v1i.yolov11/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  ├── test/
  │   ├── images/
  │   └── labels/
  └── data.yaml
  ```

### 2. Data Preprocessing
Run the preprocessing script to standardize bounding boxes and resize images:
```bash
python preprocess.py
```
This script will:
- Standardize all bounding box sizes based on mean dimensions
- Resize all images to 1920x1080 resolution
- Modify the original dataset in place

### 3. Training
Before training:
1. Update data.yaml with absolute paths:
   ```yaml
   train: /absolute/path/to/AllHockey.v1i.yolov11/train/images
   val: /absolute/path/to/AllHockey.v1i.yolov11/valid/images
   test: /absolute/path/to/AllHockey.v1i.yolov11/test/images
   nc: 1
   names: ['puck']
   ```

2. Run training:
   ```bash
   python train.py
   ```
   - Uses YOLOv8x architecture
   - Trained weights will be saved in `runs/detect/` directory

## Inference

### Using Pre-trained Model
1. Place your test video in the project directory
2. Run inference:
   ```bash
   python test.py
   ```
   use the following link to download the pretrained model and place it in the project directory.
   https://drive.google.com/file/d/1mw0ETcOdpC4AL9tiB_liG_0uTXu74A9h/view?usp=drive_link

3. Update the video source path:
   ```python
   results = model.track(
       source='path_to_your_test_video.mp4',
       ...
   )
   ```

### Using Your Trained Model
1. Modify test.py to use your trained model:
   ```python
   model = YOLO('path/best.pt')
   ```



