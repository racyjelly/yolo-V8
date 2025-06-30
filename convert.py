import json
import os
import cv2
import math
import numpy as np
import shutil
import natsort
import argparse
from typing import Optional, Union, Tuple, List, Callable, Dict
from utils.vutils import find_json_image_pairs


img_dir = "/mnt/sda/kjs_folder/Military_datasets/data/Validation/images"
label_dir = "/mnt/sda/kjs_folder/Military_datasets/data/Validation/ori_label"
save_label_dir = "/mnt/sda/kjs_folder/Military_datasets/data/Validation/labels"
root_dir = "/mnt/sda/kjs_folder/Military_datasets/data/"

TRAINING = None
VALIDATION = None

for root, dirs, files in os.walk(root_dir):
    for dir_name in dirs:
        if dir_name.startswith('Training'):
            TRAINING = os.path.join(root, dir_name)
        elif dir_name.startswith('Validation'):
            VALIDATION = os.path.join(root, dir_name)

#training_pair_list = find_json_image_pairs(TRAINING)
validation_pair_list = find_json_image_pairs(VALIDATION)

#print(len(training_pair_list))
print(len(validation_pair_list))
        
class_names_dic = {
    "fishing ship": 0,
    "merchant ship": 1, 
    "warship": 2,
    "person": 3,
    "tanker": 4,
    "ship": 5}


def convert_to_yolo_format2(width, height, box):
    # 정규화 계수 계산
    dw = 1.0 / width
    dh = 1.0 / height
    
    # 중심점 좌표 계산
    x = (float(box[2]) + float(box[0])) / 2.0
    y = (float(box[3]) + float(box[1])) / 2.0
    
    # 너비와 높이 계산
    w = float(box[2]) - float(box[0])
    h = float(box[3]) - float(box[1])
    
    # 정규화 (0~1 범위로 변환)
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)

def convert_to_yolo_format(width, height, box):
    x_center = (box[0] + box[2] / 2.0) / width
    y_center = (box[1] + box[3] / 2.0) / height
    
    # 너비와 높이 정규화
    w = box[2] / width
    h = box[3] / height
    
    return (x_center, y_center, w, h)


converted_label = []

for (label_path, img_path) in validation_pair_list:
    # stream = open(image_path + '/' + image_path_list[i], 'rb')
    stream = open(img_path, 'rb')
    bytes = bytearray(stream.read())
    numpyarray = np.asarray(bytes, dtype=np.uint8)
    img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
    img2 = img.copy()
    height, width, _ = img.shape
    
    temp_box_list =[]
    with open(label_path, encoding='UTF-8') as json_file:
        json_data = json.load(json_file)
        
        for word in list(json_data.keys()):
            if 'annotations' in word:
                annot = json_data[word]
                # bbox 복수객체
                if len(annot) > 1:
                    for j in range(len(annot)):
                        bbox = annot[j]['bbox']
                        class_id = annot[j]['class']
                        x, y, w, h = convert_to_yolo_format(width, height, bbox)

                        temp = [class_id, x, y, w, h]
                        temp_box_list.append(temp)
                # bbox 단일객체
                else:
                    bbox = annot[0]['bbox']
                    class_id = annot[0]['class']
                    x, y, w, h = convert_to_yolo_format(width, height, bbox)
        
                    temp = [class_id, x, y, w, h]
                    temp_box_list.append(temp)
        
    converted_label.append(temp_box_list)

print(len(converted_label))

for i in range(len(converted_label)):
    label_name = validation_pair_list[i][0].split('/')[-1]
    f = open(save_label_dir + "/" + label_name.replace('.json', '.txt'), 'w')
    for j in range(len(converted_label[i])):
        f.write(str(converted_label[i][j][0])+' ')
        f.write(str(converted_label[i][j][1])+' ')
        f.write(str(converted_label[i][j][2])+' ')
        f.write(str(converted_label[i][j][3])+' ')
        f.write(str(converted_label[i][j][4])+'\n')
    f.close()

print("Convert to Yolo label format done!")
