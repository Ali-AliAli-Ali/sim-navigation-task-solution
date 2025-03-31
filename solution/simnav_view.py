import os
import cv2
import numpy as np
from ultralytics import YOLO

global_part_path = os.path.dirname(os.path.abspath(__file__))
yolo_best_path = f"{global_part_path}/../runs/segment/yolov8_simnav6/weights/best.pt"

model = YOLO(yolo_best_path) 

# train_metrics = model.train(
#     data='lane_segmentn.yaml',  
#     epochs=100,                  
#     imgsz=640,                 
#     batch=16,                  
#     name='yolov8_simnav',
#     device='cpu'
# )

# metrics = model.val()


### FOR TESTS ###

# semantic segmentation of frame

frame = cv2.resize(cv2.imread(f"{global_part_path}/../../lane_segmentn_dataset/images/val/23.png"), 
                   (640, 640), 
                   interpolation=cv2.INTER_LINEAR)
result = model.predict(frame, show=True)[0]

# get binary mask for lane and bridges

masks = result.masks.data
class_ids = result.boxes.cls

lane_bridge_masks = [
    mask.numpy().astype(np.uint8)
    for mask, class_id in zip(masks, class_ids)
    if class_id in [2, 3]
]

lane_bridge_masks_merged = np.zeros(lane_bridge_masks[0].shape).astype(np.uint8)
for mask in lane_bridge_masks:
    lane_bridge_masks_merged = cv2.bitwise_or(lane_bridge_masks_merged, mask)

# get the middle of lane and bridges

from skimage.morphology import skeletonize

skeleton = skeletonize(lane_bridge_masks_merged).astype(np.uint8)

# removing "tree-like" parts belonging to obstacles, mistook for lane, from the path skeleton

kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=np.uint8)
neighbors = cv2.filter2D(skeleton, -1, kernel, borderType=cv2.BORDER_CONSTANT)  # search for points with 3+ white neighbours
tree_joints = ((neighbors >= 3) & (skeleton > 0)).astype(np.uint8)
tree_joints_areas = cv2.dilate(tree_joints, np.ones((10, 10), np.uint8))

skeleton_clean = cv2.subtract(skeleton, tree_joints_areas)

contours, _ = cv2.findContours(skeleton_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
large_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > 128] 

lane_bridge_skeleton = np.zeros_like(mask)
cv2.drawContours(lane_bridge_skeleton, large_contours, -1, 1, thickness=1)

w, h = lane_bridge_skeleton.shape[0] // 2, lane_bridge_skeleton.shape[1] // 2
local_path = lane_bridge_skeleton[w//4*3 : w//4*5, h//4*3 : h//4*5]
local_path.argmax(axis=0)

cv2.rectangle(lane_bridge_skeleton, (w//4*3 - 1, h//4*3 - 1), (w//4*5 + 1, h//4*5 + 1), color=(1,0,0), thickness=1)    # area of view
cv2.rectangle(lane_bridge_skeleton,                                                                   
              (local_path.argmax(axis=0).argmax() + w//4*3, local_path.argmax(axis=0).max() + h//4*3),                 # intersection with area of view 
              (local_path.argmax(axis=0).argmax() + w//4*3 + 4, local_path.argmax(axis=0).max() + h//4*3 + 4), 
              color=(1,0,0), 
              thickness=4)
cv2.imshow('Path bin mask', lane_bridge_skeleton * 255)
# print(lane_bridge_skeleton[local_path.argmax(axis=0).argmax()])
# plt.imshow('Skeleton', lane_bridge_skeleton * 255)

cv2.waitKey(0)
cv2.destroyAllWindows()

