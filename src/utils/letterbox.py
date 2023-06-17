import cv2
import numpy as np
# def letterbox_image(image, size):
#     '''resize image with unchanged aspect ratio using padding'''
#     iw, ih = image.size
#     w, h = size
#     scale = min(w/iw, h/ih)
#     nw = int(iw*scale)
#     nh = int(ih*scale)

#     image = image.resize((nw,nh), Image.BICUBIC)
#     new_image = Image.new('RGB', size, (128,128,128))
#     new_image.paste(image, ((w-nw)//2, (h-nh)//2))
#     return new_image

def letterbox_image(image, expected_size):
    """
    This function is used to resize image with unchanged aspect ratio using padding
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    ih, iw, _ = image.shape
    eh, ew = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)

    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    new_img = np.full((eh, ew, 3), 128, dtype='uint8')
    # fill new image with the resized image and centered it
    new_img[(eh - nh) // 2:(eh - nh) // 2 + nh,
            (ew - nw) // 2:(ew - nw) // 2 + nw,
            :] = image.copy()
    return new_img