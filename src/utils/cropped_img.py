from tqdm import tqdm
import os
from os import makedirs, listdir
import cv2
from .letterbox import letterbox_image
import numpy as np
from retinaface_detector import FaceDetector
import math
from PIL import Image
def detect_face(image_path):
  image = cv2.imread(image_path)
  detector = FaceDetector()
  boxes, scores, landmarks = detector(image)
  faceSizeList = []
  for box in boxes:
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    dis = math.sqrt((x1-x0)**2 + (y1-y0)**2)
    faceSizeList.append(dis)
  idx = np.argmax(faceSizeList)
  box_main = boxes[idx]
  # x0,y0,x1,y1 = box_main[0], box_main[1], box_main[2], box_main[3]
  W, H = x1-x0, y1-y0
  croped = image[box_main[1]:box_main[3], box_main[0]:box_main[2]][:,:,::-1]
  img = Image.fromarray(croped)
  return img



def cropped_img(root, output_root):
    """
    Hàm này dùng để detect cắt ra ảnh chứa mặt và resize ảnh về 112*112

    args: 
        root: thư mục chứa ảnh sau khi rotate
        output_root: thư mục sau khi crop mặt
    return:
        None
    """
    for person in tqdm(listdir(root)):
        person_dir = os.path.join(root, person)
        output_person_dir = os.path.join(output_root, person)
        if not os.path.exists(output_person_dir):
            makedirs(output_person_dir, exist_ok=True)
            for datafile in listdir(person_dir):
                datafile_path = os.path.join(person_dir,datafile)
                output_datafile_path = os.path.join(output_person_dir,datafile)
                # try:

                cropped_image = detect_face(image_path = datafile_path)
                if cropped_image is None:
                    continue
                cropped_image = letterbox_image(cropped_image, (112,112))
                cv2.imwrite(output_datafile_path,np.array(cropped_image)[:,:,::-1])
                # except Exception as e:
                #   # os.remove(datafile_path)
                #   print(e)
                    # pass
                
            