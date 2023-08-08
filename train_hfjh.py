from ultralytics import YOLO

# Load a model
model = YOLO('./runs/detect/train/weights/best.pt')  # load a pretrained model (recommended for training)

# Train the model with multi GPUs
model.train(data='hfjh.yaml', epochs=100, imgsz=704, device=[0])