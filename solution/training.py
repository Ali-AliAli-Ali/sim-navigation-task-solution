from ultralytics import YOLO

model = YOLO("yolov8n-seg.pt") 

train_metrics = model.train(
    data='lane_segmentn.yaml',  
    epochs=100,                  
    imgsz=640,                 
    batch=16,                  
    name='yolov8_simnav',
    device='cpu'
)
