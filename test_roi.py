import cv2
from ultralytics import YOLO
import numpy as np

def process_video_with_zoom():
    # Load the model
    model = YOLO('runs/detect/yolov11_icehockey_puck5/weights/best.pt')
    
    # Open the video
    video_path = 'Puck-Track-Test-Video.mp4'
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Calculate zoom (15% from each side means taking 70% of original size)
    zoom_ratio = 0.15
    zoom_scale = 1 - (zoom_ratio * 2)  # if zoom_ratio is 0.15, zoom_scale will be 0.7
    
    # Calculate dimensions for zoomed frame
    zoomed_width = int(width * zoom_scale)
    zoomed_height = int(height * zoom_scale)
    
    # Calculate crop start points (to center the crop)
    start_x = (width - zoomed_width) // 2
    start_y = (height - zoomed_height) // 2
    
    # Setup video writer
    output_path = 'tracked_video_zoomed.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Store original frame
        original_frame = frame.copy()
        
        # Crop center portion of frame (zoom effect)
        zoomed_frame = frame[start_y:start_y+zoomed_height, start_x:start_x+zoomed_width]
        
        # Run detection on zoomed frame
        results = model(zoomed_frame, conf=0.3)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
                
                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Adjust coordinates back to original frame
                    x1 += start_x
                    x2 += start_x
                    y1 += start_y
                    y2 += start_y
                    
                    # Draw detection box
                    cv2.rectangle(original_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if track_ids is not None and i < len(track_ids):
                        track_id = int(track_ids[i])
                        cv2.putText(original_frame, f"ID: {track_id}", 
                                  (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                  0.5, (0, 255, 0), 2)
        
        # Draw zoom boundaries
        cv2.rectangle(original_frame, 
                     (start_x, start_y), 
                     (start_x + zoomed_width, start_y + zoomed_height), 
                     (255, 0, 0), 2)
        
        # Draw zoom info
        cv2.putText(original_frame, f"Zoom: {int(zoom_scale * 100)}%", 
                    (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 0, 0), 2)
        
        # Write and display frame
        out.write(original_frame)
        cv2.imshow('Zoomed Tracking', original_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video_with_zoom()