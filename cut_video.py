import os
# import sys


def get_video_duration(video_file_path):
    cmd = f'ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 \"{video_file_path}\"'
    duration = os.popen(cmd).read()
    return duration

def cut_video(video_file_path, start, vid_len, output_file_path):
    # -y: overwrite output file without asking, -c copy: copy the video stream without re-encoding, -ss: start time, -t: duration, -i: input file
    # sys.stdout.write(f'Cutting video from {start} to {vid_len}...\n')
    # sys.path.append(fr"D:\ffmpeg\bin")
    cmd = f'ffmpeg -i \"{video_file_path}\" -ss {start} -t {vid_len} -y -c copy \"{output_file_path}\"'
    os.system(cmd)

def get_timestamp(timestamp_file_path):
    timestamp = []
    with open(timestamp_file_path, 'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            timestamp.append(line.strip().split(' ')[ : 2])
    return timestamp

def cut_videos(input_file_path, output_folder_path, timestamp_file_path):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
        print(f'{output_folder_path} created.\n')

    timestamp = get_timestamp(timestamp_file_path)
    print(f'{timestamp=}')

    for i in range(len(timestamp)):
        start = timestamp[i][0]
        vid_len = timestamp[i][1]
        output_file_path = os.path.join(output_folder_path, f'{video_name[ : -4]}_{i + 1}.mp4')
        cut_video(input_file_path, start, vid_len, output_file_path)
        print(f'video {i + 1} cut done.\n\n')

    print('Done\n')



if __name__ == '__main__':

    video_folder_path = fr'D:\NCU\Special_Project\Taekwondo_media\self_rec\FHD-240FPS\\'
    video_name = fr'360旋踢左.mp4'
    output_folder_path = fr'D:\NCU\Special_Project\Taekwondo_media\self_rec\FHD-240FPS\cut\\' + video_name[ : -4] + '\\'
    timestamp_file_path = video_folder_path + video_name[ : -4] + fr'.txt'

    # cut_videos(os.path.join(video_folder_path, video_name), output_folder_path, timestamp_file_path)
    # ss = '00:01:49.000' # start time
    # t = '00:00:03.000' # time
    # num = 1
    # ts = get_timestamp(timestamp_file_path) # timestamp
    # ss = ts[num - 1][0]
    # t = ts[num - 1][1]
    # cut_video(os.path.join(video_folder_path, video_name), ss, t, os.path.join(output_folder_path, f'{video_name[ : -4]}_{num}.mp4'))
    cut_video(fr"D:\NCU\Special_Project\Taekwondo_media\self_rec\FHD-240FPS\RoundhouseKickRight.mp4",
              '0:03', '0:26', fr"D:\NCU\Special_Project\Taekwondo_media\self_rec\FHD-240FPS\cut\\" + fr'RoundhouseKickRight1-10.mp4')


