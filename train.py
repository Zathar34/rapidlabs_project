from ultralytics import YOLO
if __name__ == '__main__':
    # Load a pre-trained YOLOv11 model
    model = YOLO('./yolo11x.pt')  # Load a pre-trained model (recommended for training)

    # Train the model
    results = model.train(
        data='AllHockey.v1i.yolov11\data.yaml',  # Path to your data.yaml file
        epochs=10,  # Number of training epochs
        imgsz=640,  # Image size
        batch=4,  # Batch size
        name='yolov11_icehockey_puck',  # Name of the training run
        device='0'  # GPU device id
    )

