import cv2
import numpy as np
import pillow_heif
from os import makedirs,path,listdir
import os
from tqdm import tqdm


def extract_image_one_fps_laplacian(video_source_path, frame_count = 15):

    vidcap = cv2.VideoCapture(video_source_path)
    frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = int(frames/frame_count)
    count = 0
    success = True
    accumulate = []
    extracted = []
    while success:
      success, image = vidcap.read()
      # Ghi tạm frame vào accumulate
      # Tăng biến đếm lên 1
      accumulate.append(image)
      count = count + 1
      # print(count)
      # Nếu biến đếm bằng FPS, reset biến đếm
      # Lấy ra frame nét nhất trong số những accumulate
      # Thêm frame nét nhất vào extracted
      # Gán accumulated bằng rỗng
      if count == fps:
        count = 0
        val = 0
        extract = None
        for image in accumulate:
          if image is not None:
            resLap = cv2.Laplacian(image, cv2.CV_64F)
            score = resLap.var()
            if score >= val: 
              val = score
              extract = image
        extracted.append(extract)
        accumulate = []
        pass
    return extracted


def extract_image_one_fps(video_source_path, frame_count = 12):
  vidcap = cv2.VideoCapture(video_source_path)
  frames = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
  fps = int(frames/frame_count)
  success = True
  extracted = []
  i = 0
  while success:
    success, image = vidcap.read()
    i += 1
    if i%fps == 0:
      i = 0
      extracted.append(image)

  return extracted

def convert_vid2img(root: str,output_root: str):
    """Ham nay lam abc xyz

    Args:
        root:
            thu muc input
        output_root:
            ....

    Returns:
        None
    """
    for person in tqdm(listdir(root)):
        person_dir = os.path.join(root, person)
        output_person_dir = path.join(output_root, person)
        video_idx = 0
        if not os.path.exists(output_person_dir):
            makedirs(output_person_dir, exist_ok=True)
        
        for datafile in listdir(person_dir):
            lower = datafile.lower()
            output_file_path = path.join(output_person_dir,datafile)
            datafile_path = path.join(person_dir,datafile)

            if lower.endswith(".mov") or lower.endswith(".mp4"):
                # read video and save frames
                video_idx += 1
                ans = extract_image_one_fps_laplacian(datafile_path)
                for i, frame in enumerate(ans):
                    j = i + 1
                    output_file_name = str(video_idx) + "_frame" + str(j) + ".jpg"
                    output_file_path = path.join(output_person_dir, output_file_name)
                    cv2.imwrite(output_file_path, frame)
            elif lower.endswith(".jpg"):
                image = cv2.imread(datafile_path)
                cv2.imwrite(output_file_path, image)
            elif lower.endswith(".heic"):
                output_datafile = os.path.splitext(output_file_path)
                output_datafile_path = output_datafile[0] + '.png'
                # print(output_datafile)
                heif_file = pillow_heif.open_heif(datafile_path,convert_hdr_to_8bit=False, bgr_mode = True)
                image = np.asarray(heif_file)
                cv2.imwrite(output_datafile_path, image)
