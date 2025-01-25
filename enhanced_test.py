from ultralytics import YOLO
import cv2
import numpy as np
from collections import deque
from filterpy.kalman import KalmanFilter
import numpy as np

class PuckTracker:
    def __init__(self, model_path, history_length=30):
        self.model = YOLO(model_path)
        self.history = deque(maxlen=history_length)
        self.kalman = self._init_kalman()
        self.last_measurement = None
        self.frames_since_detection = 0
        self.max_prediction_frames = 15  # Maximum frames to predict without detection
        
    def _init_kalman(self):
        # Initialize Kalman filter for position and velocity tracking
        kalman = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, dx, dy], Measurement: [x, y]
        
        # State transition matrix
        kalman.F = np.array([
            [1, 0, 1, 0],  # x = x + dx
            [0, 1, 0, 1],  # y = y + dy
            [0, 0, 1, 0],  # dx = dx
            [0, 0, 0, 1]   # dy = dy
        ])
        
        # Measurement matrix
        kalman.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Covariance matrices
        kalman.R = np.array([[10, 0],      # Measurement noise
                            [0, 10]])
        kalman.Q = np.array([[1, 0, 0, 0],    # Process noise
                            [0, 1, 0, 0],
                            [0, 0, 10, 0],
                            [0, 0, 0, 10]]) * 0.1
        
        return kalman
    
    def _get_box_center(self, box):
        return ((box[0] + box[2])/2, (box[1] + box[3])/2)
    
    def _predict_next_position(self):
        # Use Kalman filter to predict next position
        self.kalman.predict()
        predicted_state = self.kalman.x
        return predicted_state[:2]  # Return predicted x, y
    
    def _update_kalman(self, measurement):
        # Update Kalman filter with new measurement
        self.kalman.update(measurement)
        self.last_measurement = measurement
    
    def preprocess_frame(self, frame):
        # Enhance frame for varying lighting conditions
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    def filter_detections(self, results, frame_size):
        filtered_boxes = []
        confidences = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                
                for i, box in enumerate(xyxy):
                    # Size-based filtering
                    box_w = box[2] - box[0]
                    box_h = box[3] - box[1]
                    w_ratio = box_w / frame_size[0]
                    h_ratio = box_h / frame_size[1]
                    
                    # Check if box size is within expected puck size range
                    if (0.01 < w_ratio < 0.05 and 
                        0.01 < h_ratio < 0.05 and 
                        confs[i] > 0.25):
                        filtered_boxes.append(box)
                        confidences.append(confs[i])
        
        return filtered_boxes, confidences
    
    def track_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = 'enhanced_tracking.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Run detection with tracking
            results = self.model.track(
                source=processed_frame,
                conf=0.25,
                iou=0.6,
                tracker='botsort.yaml',
                persist=True
            )
            
            # Filter detections
            filtered_boxes, confidences = self.filter_detections(results, (width, height))
            
            if filtered_boxes:
                self.frames_since_detection = 0
                best_box_idx = np.argmax(confidences)
                best_box = filtered_boxes[best_box_idx]
                confidence = confidences[best_box_idx]
                
                # Get center of detected box
                center = self._get_box_center(best_box)
                self._update_kalman(np.array([center[0], center[1]]))
                self.history.append(center)
                
                # Draw detection with confidence
                x1, y1, x2, y2 = map(int, best_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            else:
                self.frames_since_detection += 1
                if self.frames_since_detection < self.max_prediction_frames:
                    # Predict position using Kalman filter
                    predicted_pos = self._predict_next_position()
                    x, y = map(int, predicted_pos)
                    
                    # Draw predicted position
                    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "Predicted", (x-30, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Draw trajectory
            if len(self.history) > 1:
                points = np.array(list(self.history), dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 0, 0), 2)
            
            # Add tracking status
            status = "Tracking" if self.frames_since_detection == 0 else "Predicting"
            cv2.putText(frame, f"Status: {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            out.write(frame)
            cv2.imshow('Puck Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PuckTracker('runs/detect/yolov11_icehockey_puck5/weights/best.pt')
    tracker.track_video('Puck-Track-Test-Video.mp4')