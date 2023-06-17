from retinaface_detector import FaceDetector
import cv2
import numpy as np
import os
from os import makedirs, listdir
from PIL import Image, ImageOps


from tqdm import tqdm
def img_exif_transpose(root, output_root):
    """
    Hàm này dùng để quay ảnh về ảnh gốc sau khi bị quay bới meta data
    args:
        root: folder chứa dữ liệu sau khi đã ở dạng ảnh
        output_root: folder chứa dữ liệu sau khi quay
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
          image =  Image.open(datafile_path)
          image = ImageOps.exif_transpose(image)
          image.save(output_datafile_path)


def rotate_img(image_path):
  image = cv2.imread(image_path)
  list_image_rotate = []
  list_image_rotate.append(image)
  list_image_rotate.append(cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE))
  list_image_rotate.append(cv2.rotate(image,cv2.ROTATE_180))
  list_image_rotate.append(cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE))
  detector = FaceDetector()
  list_score = []
  for img in list_image_rotate:
    resp = detector(img)
    # print(resp)
    try:
      score = resp[1][0]
    except:
      score = 0
    list_score.append(score)
  print(list_score)
  idx_max = np.argmax(list_score)
  cv2.imwrite(image_path,list_image_rotate[idx_max])


def rotate_all_img(root):
    """
    This function is used to rotate all face image in root folder
    args: 
        root: folder contains all image
    return:
        None  


    """
    for person in tqdm(os.listdir(root)):
        person_dir = os.path.join(root, person)
        for datafile in tqdm(os.listdir(person_dir),leave = False):
            datafile_path = os.path.join(person_dir,datafile)
            if 'frame' not in datafile_path:
                print(datafile_path)
                rotate_img(datafile_path)