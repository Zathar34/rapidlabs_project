from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/detect/yolov11_icehockey_puck5/weights/best.pt')  # Path to your trained model

# Run inference on the test video
results = model.track(
    source='Puck-Track-Test-Video.mp4',  # Path to your test video
    conf=0.2,  # Confidence threshold
    iou=0.7,  # IoU threshold
    show=True,  # Show results in real-time
    save=True,  # Save the results
    save_txt=True,  # Save results as .txt file
    save_conf=True,  # Save confidence scores
    save_crop=False,  # Save cropped images
    tracker='bytetrack.yaml'  # Use ByteTrack tracker
)

# Optionally, you can save the tracked video with bounding boxes
results.save('tracked_video.mp4')