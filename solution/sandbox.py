from pioneer_sdk import Camera
from pioneer_sdk import Pioneer
# from edubot_sdk import EdubotGCS

import time
import threading
import queue
import sys
import os

import numpy as np
import cv2
from ultralytics import YOLO
from skimage.morphology import skeletonize


global_part_path = os.path.dirname(os.path.abspath(__file__))

height_obs = 5
cam_resolutn = 640

see_bridge, on_bridge, see_finish, on_finish = False, False, False, False

speed_queue = queue.Queue()


def camera_stream_segment(camera: Camera):
    yolo_best_path = f"{global_part_path}/../runs/segment/yolov8_simnav6/weights/best.pt"
    model = YOLO(yolo_best_path) 

    while True:
        orig_frame = camera.get_cv_frame()
        frame = cv2.resize(orig_frame, (cam_resolutn, cam_resolutn), interpolation=cv2.INTER_LINEAR)
        
        if frame is not None:
            result = model.predict(frame, verbose=False)[0] #, show=True)[0]

            while not ((2 in result.boxes.cls) and (3 in result.boxes.cls)):
                orig_frame = camera.get_cv_frame()
                frame = cv2.resize(orig_frame, (cam_resolutn, cam_resolutn), interpolation=cv2.INTER_LINEAR)
                if frame is not None:
                    result = model.predict(frame, verbose=False)[0] #, show=True)[0]

            get_speeds(result)

            # if check_finish(results[0]):
            #     see_finish = True
            #     go_to_finish(results[0])
            # if check_bridge(results[0]) and not on_bridge:
            #     see_bridge = True
            #     go_to_bridge(results[0])

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows() 


def build_path(result):          
    """
    get exact path of lane and bridges (TODO: join them into a graph to use bridges)

    :return: binary mask of camera frame, where 1 is path for the drone to follow
    """                       
    # get binary mask for lane and bridges
    masks = result.masks.data
    boxes = result.boxes
    class_ids = boxes.cls.numpy()

    lane_bridge_masks = [
        mask.numpy().astype(np.uint8)
        for mask, class_id in zip(masks, class_ids)
        if class_id in [2, 3]
    ]

    lane_bridge_masks_merged = np.zeros(lane_bridge_masks[0].shape).astype(np.uint8)
    for mask in lane_bridge_masks:
        lane_bridge_masks_merged = cv2.bitwise_or(lane_bridge_masks_merged, mask)

    # get the middle of lane and bridges
    skeleton = skeletonize(lane_bridge_masks_merged).astype(np.uint8)

    # removing "tree-like" parts belonging to obstacles, mistook for lane, from the path skeleton
    kernel = np.array([[1, 1, 1],                                                                # search for points with 3+ white neighbours
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbors = cv2.filter2D(skeleton, -1, kernel, borderType=cv2.BORDER_CONSTANT)  
    tree_joints = ((neighbors >= 3) & (skeleton > 0)).astype(np.uint8)
    tree_joints_areas = cv2.dilate(tree_joints, np.ones((10, 10), np.uint8))

    skeleton_clean = cv2.subtract(skeleton, tree_joints_areas)

    contours, _ = cv2.findContours(skeleton_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # removing short "branches"
    large_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > 128] 

    lane_bridge_skeleton = np.zeros_like(mask)
    cv2.drawContours(lane_bridge_skeleton, large_contours, -1, 1, thickness=1)

    return lane_bridge_skeleton     # this is called "path" further

def get_next_point(path):
    """
    get the next point to fly

    :return: pair of int coordinates in camera frame matrix
    """
    center = cam_resolutn // 2
    local_path = path[ center//4*3 : center//4*5, 
                       center//4*3 : center//4*5 ]
    
    cv2.rectangle(path,                                                                   
              (local_path.argmax(axis=0).argmax() + center//4*3, local_path.argmax(axis=0).max() + center//4*3),                 # intersection with area of view 
              (local_path.argmax(axis=0).argmax() + center//4*3 + 4, local_path.argmax(axis=0).max() + center//4*3 + 4), 
              color=(1,0,0), 
              thickness=3)
    cv2.imshow('Path binary mask', path * 255)

    return (local_path.argmax(axis=0).argmax() + center//4*3, 
            local_path.argmax(axis=0).max() + center//4*3)

def get_speeds(result):
    """
    speed for x, y. Assume the speed for x is 1, if we move or 0 

    :return: float
    """
    next_point = get_next_point( build_path(result) )
    curr_point = (cam_resolutn // 2, cam_resolutn // 2)
    distances = (next_point[0] - curr_point[0], curr_point[1] - next_point[1])     # in drone's coord system y is flipped
    
    speed_queue.put( ( distances[0] / (abs(distances[1]) + abs(distances[0])), 
                       distances[1] / (abs(distances[1]) + abs(distances[0])) ) ) 


def check_finish(result):
    return 4 in result.boxes.cls
def check_bridge(result):
    return 3 in result.boxes.cls

def go_to_finish():
    pass # check if there are obstacles on the way. If not, go straight
def go_to_bridge(result):
    result.boxes.xyxy[ result.boxes.cls == 3 ]  # bounding boxes coords
    # understand in where the bridge begins 
    pass


def pioneer_control(pioneer: Pioneer):
    pioneer.arm()
    time.sleep(1)
    pioneer.takeoff()
    time.sleep(3)

    while pioneer.get_local_position_lps(True)[2] < height_obs:
        pioneer.set_manual_speed(*[0, 0, 3], 0)

    while pioneer.get_local_position_lps(True)[0] > 29: 
        speeds = speed_queue.get()
        pioneer.set_manual_speed(*speeds, 0, 0)
        with speed_queue.mutex:
            speed_queue.queue.clear()

        # while not pioneer.point_reached():
        #     pass
        # break
    print(on_finish)

    

# def geobot_control(geobot: EdubotGCS):
#     pass
#     '''Пример управления Геоботом'''
#     geobot_way = [[28.20, -10.45], [20.3, -8.5], [10.73, 2.44], [3.73, 3.14], [-8.53, -9.02], [-16.88, 2.2]]

#     while True:
#         for point in geobot_way:
#             geobot.go_to_local_point(*point)
#             while not geobot.point_reached():
#                 pass
#             time.sleep(0.2)



def main():
    args = sys.argv[1:]
    if len(args) < 3:
        print("Не переданы необходимые аргументы: порт Пионера, порт камеры Пионера, порт Геобота. Пример: python main.py 8001 18001 8002")
        exit(1)

    pioneer = Pioneer(ip="127.0.0.1", mavlink_port=int(args[0]))
    pioneer_camera = Camera(ip="127.0.0.1", port=int(args[1]))
    # geobot = EdubotGCS(ip="127.0.0.1", mavlink_port=int(args[2]))

    camera_thread = threading.Thread(target=camera_stream_segment, args=(pioneer_camera, ))
    pioneer_thread = threading.Thread(target=pioneer_control, args=(pioneer, ))
    # geobot_thread = threading.Thread(target=geobot_control, args=(geobot, ))

    try:
        camera_thread.start()
        print("Camera started")
        pioneer_thread.start()
        print("Camera started")
        # geobot_thread.start()

        camera_thread.join()
        pioneer_thread.join()
        # geobot_thread.join()
    except KeyboardInterrupt:
        pioneer.land()
        pioneer.disarm()


if __name__ == "__main__":
    sys.argv += ['8000', '18000', '8001']
    main()
