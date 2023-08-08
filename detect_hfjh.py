from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('./runs/best.pt')

# Run inference on 'bus.jpg' with arguments
# model.predict(r'D:\Solution\datas\HFJH\AAI0005JM1\#19\Layer_0\TopFlat\split_img_19', save=True, imgsz=704, conf=0.5)
model.predict(r'C:\Users\Pu\Desktop\1\2', save_crop=True, save_txt=True, save=True, imgsz=704, conf=0.5)