from ultralytics import YOLO

# Load the trained model
model = YOLO('best.pt')

# Run inference on the test video
results = model.track(
    source='Puck-Track-Test-Video.mp4',
    conf=0.2,                    # Confidence threshold
    iou=0.7,                     # IoU threshold
    show=True,                   # Show results in real-time
    save=True,                   # Save the results
    save_txt=True,               # Save results as .txt file
    save_conf=True,              # Save confidence scores
    save_crop=False,             # Save cropped images
    tracker='botsort.yaml'       # Use BOT-SORT tracker
)

# Save the tracked video
results.save('tracked_video_botsort.mp4')