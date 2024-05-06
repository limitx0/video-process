# https://github.com/de-code/python-tf-bodypix/

import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import cv2
import numpy as np
import os


def black_bg_to_transparent(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    lower_black = np.array([0, 0, 0, 255])
    upper_black = np.array([0, 0, 0, 255])
    mask = cv2.inRange(img, lower_black, upper_black)
    img[mask == 255] = [255, 255, 255, 0]
    
    # https://stackoverflow.com/questions/40527769/removing-black-background-and-make-transparent-from-grabcut-output-in-python-ope
    b, g, r, _ = cv2.split(img)
    _, alpha = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    bgra = cv2.merge((b, g, r, alpha))
    dst = cv2.addWeighted(bgra, 1, bgra, 0, 0)
    
    return dst

def overlay_transparent(background_path, overlay_path, x=0, y=0):
    background = cv2.imread(background_path, -1)
    overlay = cv2.imread(overlay_path, -1)
    
    background_width = background.shape[1]
    background_height = background.shape[0]
    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype) * 255
            ],
            axis=2,
        )
    
    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y:y + h, x:x + w] = (1.0 - mask) * background[y:y + h, x:x + w] + mask * overlay_image

    return background

def overlay_img(background_path, overlay_path):
    # https://stackoverflow.com/questions/69620706/overlay-image-on-another-image-with-opencv-and-numpy
    background = cv2.imread(background_path)
    overlay = cv2.imread(overlay_path)
    
    # extract alpha channel from foreground image as mask and make 3 channels
    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    alpha = overlay[:,:,3]
    alpha = cv2.merge([alpha,alpha,alpha])

    # extract bgr channels from foreground image
    front = overlay[:,:,0:3]

    # blend the two images using the alpha channel as controlling mask
    result = np.where(alpha==(0,0,0), background, front)

    return result


def test_bodypix(path, output_dir='./output/', cv2_show=False):
    print('Start testing BodyPix...')
    
    # Create output folder
    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    out_folder_name = path.split('/')[-1].split('.')[0]
    out_folder = output_dir + out_folder_name + '/'
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
        os.mkdir(out_folder + 'frames')
        os.mkdir(out_folder + 'colored_mask')
        os.mkdir(out_folder + 'masked_image')
        os.mkdir(out_folder + 'mask')
    
    # Load BodyPix model
    print(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16)
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
    
    
    # Load video and create video writer
    cap = cv2.VideoCapture(path)
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer_seg = cv2.VideoWriter(
        out_folder + out_folder_name + '_seg'  + '.avi',
        cv2.VideoWriter_fourcc(*'XVID'),
        30,
        (video_w, video_h)
    )
    # writer_masked = cv2.VideoWriter(
    #     out_folder + out_folder_name + '_masked'  + '.avi',
    #     cv2.VideoWriter_fourcc(*'XVID'),
    #     30,
    #     (video_w, video_h)
    # )
    
    cnt = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue
        if cnt == 1e3:
            break

        # BodyPix Segmentation
        result = bodypix_model.predict_single(frame)
        mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)
        # masked_image = cv2.bitwise_and(frame, frame, mask=mask) # background removed
        seg = result.get_colored_part_mask(mask)


        # seg = black_bg_to_transparent(seg)
        tf.keras.preprocessing.image.save_img(
            out_folder + f"colored_mask/output-colored-mask_{cnt}.png",
            seg
        )
        tf.keras.preprocessing.image.save_img(
            out_folder + f"mask/output-mask_{cnt}.jpg",
            mask
        )
        # tf.keras.preprocessing.image.save_img(
        #     out_folder + f"masked_image/output-masked-image_{cnt}.jpg",
        #     masked_image
        # )

        seg = seg.astype(np.uint8)
        writer_seg.write(seg)
        # writer_mask.write(mask)
        # writer_masked.write(masked_image)
        cv2.imwrite(out_folder + f"frames/frame_{cnt}.jpg", frame)
        if cv2_show:
            cv2.imshow('BodyPix', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        cnt += 1
        if cnt % 10 == 0:
            print(f'{cnt=}')
    cap.release()
    cv2.destroyAllWindows()
    print('Done testing BodyPix...')


    
def test_bodypix_image(img_path, output_dir='./output/', cv2_show=False):
    print('Start testing BodyPix image...')
    
    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16)
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16))
    img=cv2.imread(img_path, 1)

    # BodyPix Segmentation
    result = bodypix_model.predict_single(img)
    mask = result.get_mask(threshold=0.5).numpy().astype(np.uint8)
    # masked_image = cv2.bitwise_and(frame, frame, mask=mask) # background removed
    seg = result.get_colored_part_mask(mask)
    scaled_part_seg = result.get_scaled_part_segmentation(mask)
    
    with open(output_dir+f"output_args.txt", 'w') as f:
        s = 'mask=\n' + str(mask.tolist()) + '\n'
        s += 'seg=\n' + str(seg.tolist()) + '\n'
        s = 'scaled_part_seg=\n' + str(scaled_part_seg.tolist()) + '\n'
        s += '\n\n\n'
        s += str(result.__dict__)
        f.write(s)
        
    tf.keras.preprocessing.image.save_img(
        output_dir+f"output-colored-mask.jpg",
        seg
    )
    
    if cv2_show:
        cv2.imshow('BodyPix', img)
        cv2.waitKey(0)
    
    print('Done testing BodyPix image...')



if __name__ == '__main__':
    
    video_path = '/home/training_data/FHD-240FPS/cut/BackHookKickRight/_BackHookKickRight_1.mp4'
    
    img_path = '/home/script/tmp/000034.png'
    
    
    cv2_show = False
    
    test_bodypix_image(img_path=img_path, output_dir='./output/bodypix_image/', cv2_show=cv2_show)
    
    # test_bodypix(https_url, output_dir='./output/', cv2_show=cv2_show)
    # test_bodypix(video_path, output_dir='./output/', cv2_show=cv2_show)





