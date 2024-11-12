import time
import os
import numpy as np
import uuid
import cv2
import tqdm
import shutil
import sys
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel
from scipy.io import wavfile
import math
import torch
import habana_frameworks.torch.core as htcore

def main():
    # 检查命令行参数的数量
    if len(sys.argv) != 4:
        print("Usage: python demo.py <video_path> <output_video_name>")
        sys.exit(1)  # 参数数量不正确时退出程序

    # 获取video_name参数
    video_path = sys.argv[1]
    print(f"Video path is set to: {video_path}")
    audio_path = sys.argv[2]
    print(f"Audio path is set to: {audio_path}")
    output_video_name = sys.argv[3]
    print(f"output video name is set to: {output_video_name}")

    audioModel = AudioModel()
    audioModel.loadModel("checkpoint/audio.pkl")

    renderModel = RenderModel()
    renderModel.loadModel("checkpoint/render.pth")
    pkl_path = "{}/keypoint_rotate.pkl".format(video_path)
    video_path = "{}/circle.mp4".format(video_path)
    renderModel.reset_charactor(video_path, pkl_path)

    # wavpath = "video_data/audio0.wav"
    for wavpath in [audio_path, "video_data/audio0.wav", "video_data/audio1.wav"]:
        #wavpath = audio_path
        mouth_frame = audioModel.interface_wav(wavpath)

        from scipy.io import wavfile
        rate, wav = wavfile.read(wavpath, mmap=False)
        wav_len = len(wav)

        cap_input = cv2.VideoCapture(video_path)
        vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)  # 宽度
        vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 高度
        cap_input.release()

        S = time.time()

        total_frame_num = len(mouth_frame)
        MAX_BATCH_SIZE = 32    # suggest to set <=64
        total_run = (total_frame_num + MAX_BATCH_SIZE -1) // MAX_BATCH_SIZE

        task_id = str(uuid.uuid1())

        sample_rate, audio_data = wavfile.read(wavpath)
        frame_rate = 25

        # Duration of each video chunk in seconds
        chunk_duration = MAX_BATCH_SIZE / frame_rate
        # Convert chunk duration to samples
        sample_wav_for_each_run = int(chunk_duration * sample_rate)
        # sample_wav_for_each_run = len(audio_data) * MAX_BATCH_SIZE / total_frame_num
        

        output_paths = []

        for i in range(total_run):
            cur_S = time.time()
            
            os.makedirs("output/{}".format(task_id), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            save_path = "output/{}/silence_{}.mp4".format(task_id, i)
            videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width) * 1, int(vid_height)))

            # read_partial_audio(audio_path, start_sample, num_samples)
            
            # breakpoint()
            print(i*sample_wav_for_each_run, (i+1)*sample_wav_for_each_run)
            partial_audio = audio_data[i*sample_wav_for_each_run: (i+1)*sample_wav_for_each_run]

            save_wav_path = f"output/{task_id}/wav_{i}.wav"
            wavfile.write(save_wav_path, sample_rate, partial_audio)
            


            mouth_frame_run = mouth_frame[i*MAX_BATCH_SIZE: (i+1)*MAX_BATCH_SIZE]
            # breakpoint()
            #with torch.cpu.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
            out_frames = renderModel.interface_batch(mouth_frame_run)
            for frame in out_frames:
                videoWriter.write(frame)


            videoWriter.release()
            os.system(
            "ffmpeg -v quiet -y -i {} -i {} -shortest -strict -2 -q:v 1 -async 1 {}" #"ffmpeg -y -i {} -i {}  -c:v libx264 -pix_fmt yuv420p -loglevel quiet {}"
            .format(save_wav_path, save_path, f"output/{task_id}/out_{i}.mp4"))

            # output_paths.append(f"output/{task_id}/{output_video_name.split('.')[0]}_{i}.mp4")
            output_paths.append(f"out_{i}.mp4")

            
            print(f"Round {i} finished with time {time.time()-cur_S}.")


    # merge
    with open(f'output/{task_id}/file_list.txt', 'w') as f:
        for p in output_paths:
            f.write(f"file '{p}'\n")
    os.system(f"ffmpeg -y -f concat -safe 0 -i output/{task_id}/file_list.txt -c copy {output_video_name}")

    E = time.time()
    print(f"inference time: {E-S}")
    # shutil.rmtree("output/{}".format(task_id))



if __name__ == "__main__":
    main()
