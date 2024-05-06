# https://new.express.adobe.com/tools/remove-background

from rembg import remove
from PIL import Image

import cv2
# import numpy as np

import os


def remove_bg(img_path, output_path):
    input_image = Image.open(img_path)
    output_image = remove(input_image)
    # output_image = output_image.convert("RGBA")
    output_image.save(output_path)
    print(output_path + ' with background removed saved')

def inc_contrast(img_path, output_path):
    img = cv2.imread(img_path, 1)
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    try:
        cv2.imwrite(output_path, enhanced_img)
    except FileNotFoundError:
        os.mkdir(output_path)
        cv2.imwrite(output_path, enhanced_img)
    # cv2.show('Result', enhanced_img)
    print(output_path + ' increased contrast saved')

def remove_bg_img_folder(img_folder_path, out_folder_path):
    files = [f 
            for f in os.listdir(img_folder_path) 
             if os.path.isfile(os.path.join(img_folder_path, f)) and
            (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]

    for img in files:
        if img.endswith('.jpg'):
            # img = img.replace('.jpg', '.png')
            continue
        input_path = img_folder_path + '\\' + img
        output_path = out_folder_path + '\\' + img
        remove_bg(input_path, output_path)

def inc_contrast_img_folder(img_folder_path, out_folder_path):
    files = [f 
            for f in os.listdir(img_folder_path) 
             if os.path.isfile(os.path.join(img_folder_path, f)) and
            (f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jpeg'))]

    for img in files:
        if img.endswith('.jpg'):
            # img = img.replace('.jpg', '.png')
            continue
        input_path = img_folder_path + '\\' + img
        output_path = out_folder_path + '\\' + img
        inc_contrast(input_path, output_path)


if __name__ == "__main__":
    img_folder_path = fr'.\img'
    out_folder_path = fr'.\img_rm_bg'
    # remove_bg_img_folder(img_folder_path, out_folder_path)


    img_path = fr".\side_kick.png"
    output_path = fr".\side_kick_out.png"
    output_contrast_path = fr".\side_kick_contrast.png"
    # remove_bg(img_path, output_path)
    inc_contrast(output_path, output_contrast_path)


