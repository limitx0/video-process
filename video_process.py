# 【python】OpenCV—Video to Imag / Image to Video https://blog.csdn.net/bryant_meng/article/details/110079285
# Python基于OpenCV将视频逐帧保存为图片 https://blog.csdn.net/sinat_42378539/article/details/88837252
# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder

# from email.mime import image
import cv2
import os
import shutil
from img_process import *
# from rm_bg_api.ClipdropAPI import *
# from rm_bg_api.PhotoRoomAPI import *
from cut_video import *

clicked = False
black_img_path = fr".\1920x1080-black-solid-color-background.png"

def onMouse(event, x, y, flags, param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True

def playVideoOnWindow(video_path):
    video = cv2.VideoCapture(video_path)
    cv2.namedWindow("MyWindow")
    cv2.setMouseCallback('MyWindow', onMouse)
    print("Show video, click window or press any key to stop.")

    success, frame = video.read()
    while success and cv2.waitKey(1) == -1 and not clicked:
        cv2.imshow('MyWindow', frame)
        sucees, frame = video.read()

    cv2.destroyWindow('MyWindow')
    video.release()

def countFrame(video_path):
    video_cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while (True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        frame_count = frame_count + 1
    video_cap.release()

    print(f'{frame_count=}')
    return frame_count

def countFPS(video_path):
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    size = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    print(f'{size=}')
    video.release()
    return fps

def video2image(video_path, img_out_path):    
    interval = 1 # 保存时的帧数间隔
    frame_count = 0 # 保存帧的索引
    frame_index = 0 # 原视频的帧索引，与 interval*frame_count = frame_index 
    frame_num = countFrame(video_path)

    black_frame = cv2.imread(black_img_path)
    cap = cv2.VideoCapture(video_path)

    if cap.isOpened():
        success = True
    else:
        success = False
        print("读取失败!")

    while success and frame_index < frame_num:
        success, frame = cap.read()
        if success is False: # insert black frame if read failed
            print("---> 第%d帧读取失败: insert black frame" % frame_index)
            # break
            if frame_index % interval == 0:
                cv2.imwrite(fr"{img_out_path}/{frame_count:06d}.png", black_frame)
                frame_count += 1
            frame_index += 1
            success = True
            continue
            
        print("---> 正在读取第%d帧:" % frame_index, success)
        
        if frame_index % interval == 0:
            try:
                cv2.imwrite(fr"{img_out_path}/{frame_count:06d}.png", frame)
            except FileNotFoundError:
                os.mkdir(img_out_path)
                cv2.imwrite(fr"{img_out_path}/{frame_count:06d}.png", frame)
            frame_count += 1
        frame_index += 1
    cap.release()
    print()

def image2video(img_in_path, video_path, fps) -> None:
    if img_in_path[-1] != '\\':
        img_in_path += '\\'
    img = cv2.imread(img_in_path + f'{0:06d}' + '.png')  # 读取保存的任意一张图片
    fps = fps # 根据第二节介绍的方法获取视频的 
    size = (img.shape[1],img.shape[0])  # 获取视频中图片宽高度信息
    print(f'{size=}')
    """
    “I”，“4”，“2”，“0”，未压缩的 YUV 颜色编码，扩展名为 .avi
    “P”，“I”，“M”，“1”，MPEG-1 编码类型，扩展名为 .avi
    “X”，“V”，“I”，“D”，MPEG-4 编码类型，扩展名为 .avi
    “T”，“H”，“E”，“O”，Ogg Vorbis，扩展名为 .ogv
    “F”，“L”，“V”，“1”，Flash 视频，扩展名为 .flv
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # 视频编码格式：XVID -> .avi, MP4V -> .mp4, MP42 -> .avi, MJPG -> .mp4
    # fourcc = cv2.VideoWriter_fourcc(*"xvid")
    videoWrite = cv2.VideoWriter(video_path, fourcc, fps, size)# 根据图片的大小，创建写入对象 （文件名，支持的编码器，帧率，视频大小（图片大小））

    files = os.listdir(img_in_path)
    out_num = len(files)
    for i in range(0, out_num):
        fileName = img_in_path + f'{i:06d}' + '.png'  # 循环读取所有的图片,假设以数字顺序命名
        img = cv2.imread(fileName)
        videoWrite.write(img) # 将图片写入所创建的视频对象
    videoWrite.release() # 释放了才能完成写入，连续写多个视频的时候这句话非常关键
    print(f'{video_path} saved\n')

def inc_contrast_video(video_path, out_path):
    fps = countFPS(video_path)
    tmp_img_path = fr".\tmp"
    
    if not os.path.exists(tmp_img_path):
        os.mkdir(tmp_img_path)
        print(f'{tmp_img_path} created')
    video2image(video_path, tmp_img_path)
    inc_contrast_img_folder(tmp_img_path, tmp_img_path)
    image2video(tmp_img_path, out_path, fps)
    shutil.rmtree(tmp_img_path)
    print(f"\"{video_path}\"\n with contrast increased is saved in \"{out_path}\"\n")

def remove_bg_video(video_path, out_path):
    fps = countFPS(video_path)
    tmp_img_path = fr".\tmp"
    
    if not os.path.exists(tmp_img_path):
        os.mkdir(tmp_img_path)
        print(f'{tmp_img_path} created')
    video2image(video_path, tmp_img_path)
    remove_bg_img_folder(tmp_img_path, tmp_img_path)
    image2video(tmp_img_path, out_path, fps)
    shutil.rmtree(tmp_img_path)
    print(f"\"{video_path}\"\n with background removed is saved in \"{out_path}\"\n")

def rembg_inc_contrast_video(video_path, out_path):
    fps = countFPS(video_path)
    tmp_img_path = fr".\tmp\\"

    if not os.path.exists(tmp_img_path):
        os.mkdir(tmp_img_path)
        print(f'{tmp_img_path} created')
    video2image(video_path, tmp_img_path)
    remove_bg_img_folder(tmp_img_path, tmp_img_path)
    inc_contrast_img_folder(tmp_img_path, tmp_img_path)
    image2video(tmp_img_path, out_path, fps)
    shutil.rmtree(tmp_img_path)
    print(f"\"{video_path}\"\n with background removed and contrast increased is saved in \"{out_path}\"\n")

def remove_audio_from_videos(input_folder_path, output_folder_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f'{output_folder_path} created.\n')

    vids = [f
            for f in os.listdir(input_folder_path)
            if os.path.isfile(os.path.join(input_folder_path, f)) and
            (f.endswith('.mp4') or f.endswith('.mov') or f.endswith('.avi'))]
    for v in vids:
        in_file = os.path.join(input_folder_path, v)
        out_file = os.path.join(output_folder_path, v)
        cmd = f'ffmpeg -i \"{in_file}\" -c copy -an \"{out_file}\"'
        # print(f'{cmd=}')
        os.system(cmd)
        print(f'{v} with audio removed saved')
    print(f'Done remove_audio_from_videos\n')



if __name__ == "__main__":


    remove_bg_video(fr"D:\NCU\Special_Project\Taekwondo_media\video\video-11m14s-31-x3htXTI7nDI.mp4", fr"D:\NCU\Special_Project\Taekwondo_media\video\rm_bg_video-11m14s-31-x3htXTI7nDI.mp4")
