import wandb
from ultralytics import YOLO
import os
import sys


n_epochs = 150
image_size = 640
batch_size = 16
device_name = 'cpu'

wandb.init(
    project='yolov8_simnav', 
    name='yolov8n-seg-simnav',      
    config={
        'model': 'yolov8n-seg',
        'dataset': 'lane_segmentn.yaml',
        'epochs': n_epochs,
        'imgsz': image_size,
        'batch': batch_size,
        'device': device_name
    }
)


model = YOLO('yolov8n-seg.pt')

train_metrics = model.train(
    data=f'{ os.path.dirname(os.path.abspath(sys.argv[0])) }\lane_segmentn.yaml',
    epochs=n_epochs,
    imgsz=image_size,
    batch=batch_size,
    name='yolov8_simnav',
    device=device_name
)


wandb.finish()