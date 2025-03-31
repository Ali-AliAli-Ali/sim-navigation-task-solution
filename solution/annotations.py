import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.data.converter import convert_segment_masks_to_yolo_seg
from ultralytics.data.utils import visualize_image_annotations


def mask_to_bin(pic):
    pic = cv2.resize(pic, (640, 640), interpolation=cv2.INTER_NEAREST)
    # in RGB: all zeros = black = background
    #         255, 0, 0 = red = obstacle
    #         0, 255, 0 = green = lane
    #         0, 0, 255 = blue = bridge
    #         255, 255, 255 = white = finish
    colors = [ (0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 255) ] # in BGR for OpenCV
    mask = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.uint8)

    # class_ids +1 as YOLO downgrades them thinking people use 1 for background
    for i in range(pic.shape[0]):      
        for j in range(pic.shape[1]): 
            pixel = pic[i, j]          
            if np.array_equal(pixel, colors[0]):
                mask[i, j] = 1
            elif np.array_equal(pixel, colors[1]): 
                mask[i, j] = 2
            elif np.array_equal(pixel, colors[2]):  
                mask[i, j] = 3
            elif np.array_equal(pixel, colors[3]): 
                mask[i, j] = 4
            elif np.array_equal(pixel, colors[4]): 
                mask[i, j] = 5
            else:
                mask[i, j] = 1  
    return mask

def check_mask(mask_num):
    mask = cv2.imread(f'{dataset_path}bin_masks/train/{mask_num}.png', cv2.IMREAD_GRAYSCALE)
    plt.imshow(mask, cmap="jet")
    plt.colorbar()
    plt.show()

# doesn't work:  
# def visualize_annotations(image_num):
#     label_map = {  
#         0: "background",
#         1: "obstacle",
#         2: "lane",
#         3: "bridge",
#         4: "finish"
#     }
#     visualize_image_annotations(
#         f"{dataset_path}images/train/{image_num}.png",
#         f"{dataset_path}labels/train/{image_num}.txt", 
#         label_map,
#     )


dataset_path = "C:/Users/e.vladimirova/Desktop/geobot_projects/sim_navigation_competition/lane_segmentn_dataset/"

for i in range(1, 17):
    mask = cv2.imread(f'masks/train/{i}.png')
    mask_bin = mask_to_bin(mask)
    cv2.imwrite(f'{dataset_path}bin_masks/train/{i}.png', mask_bin)
    print(f"mask {i} processed")

for i in range(17, 24):
    mask = cv2.imread(f'{dataset_path}masks/val/{i}.png')
    mask_bin = mask_to_bin(mask)
    cv2.imwrite(f'{dataset_path}bin_masks/val/{i}.png', mask_bin)
    print(f"mask {i} processed")


print("Converting train to YOLO format...")
convert_segment_masks_to_yolo_seg(f"{dataset_path}bin_masks/train",
                                  f"{dataset_path}labels/train", 
                                  5)
print("Converting train to YOLO format done!")
print("Converting val to YOLO format...")
convert_segment_masks_to_yolo_seg(f"{dataset_path}bin_masks/val",
                                  f"{dataset_path}labels/val", 
                                  5)
print("Converting val to YOLO format done!")
