import time
import os
import uuid
import cv2
import tqdm
import shutil
from talkingface.audio_model import AudioModel
from talkingface.render_model import RenderModel
from starlette.middleware.cors import CORSMiddleware
import argparse
import base64

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from fastapi import File, UploadFile, HTTPException

from pydub import AudioSegment

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

def dh_convert(wavpath, use_batching=False):
    """Input: 16000Hz mono audio file, Output: generated mp4 file path"""

    mouth_frame = audioModel.interface_wav(wavpath)
    cap_input = cv2.VideoCapture(video_path)
    vid_width = cap_input.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = cap_input.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_input.release()
    task_id = str(uuid.uuid1())
    os.makedirs("output/{}".format(task_id), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = "output/{}/silence.mp4".format(task_id)
    videoWriter = cv2.VideoWriter(save_path, fourcc, 25, (int(vid_width) * 1, int(vid_height)))
    S = time.time()

    if not use_batching:
        for frame in tqdm.tqdm(mouth_frame):
            out_frame = renderModel.interface(frame)
            # cv2.imshow("s", frame)
            # cv2.waitKey(40)

            videoWriter.write(out_frame)
    else:
        total_frame_num = len(mouth_frame)
        MAX_BATCH_SIZE = 32    # suggest to set <=64
        total_run = (total_frame_num + MAX_BATCH_SIZE -1) // MAX_BATCH_SIZE
        for i in range(total_run):
            mouth_frame_run = mouth_frame[i*MAX_BATCH_SIZE: (i+1)*MAX_BATCH_SIZE]
            out_frames = renderModel.interface_batch(mouth_frame_run)
            for frame in out_frames:
                videoWriter.write(frame)


    videoWriter.release()
    E = time.time()
    print(f"inference time: {E-S}")
    val_video = "../output/{}.mp4".format(task_id)

    output_video_path = f"{uuid.uuid4()}.mp4"
    os.system(
        "ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p -loglevel quiet {}".format(save_path, wavpath, output_video_path))
    shutil.rmtree("output/{}".format(task_id))
    return output_video_path

def generate_video(video_path):
    with open(video_path, mode="rb") as file_like:  # 
        yield from file_like  # 
    
    os.remove(video_path)


@app.get("/v1/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)

@app.post("/v1/digital_human")
async def digital_human(file: UploadFile = File(...)):
    """Input: audio file, Output: Streaming video response."""
    print("Digital human inference begin.")
    print(file.content_type)
    if file.content_type not in ["audio/wav", "application/octet-stream", "audio/wave", ]:
        raise HTTPException(status_code=400, detail="File must be a WAV format")

    uid = str(uuid.uuid4())
    file_name = uid + ".wav"
    # Save the uploaded file
    with open(file_name, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    audio = AudioSegment.from_file(file_name)
    audio = audio.set_frame_rate(16000)
    audio.export(f"{file_name}", format="wav")
    
    output_video_path = dh_convert(file_name, use_batching=use_batching)
    os.remove(file_name)

    return StreamingResponse(generate_video(output_video_path), media_type="video/mp4")

# @app.post("/v1/digital_human/change_refer")
# async def change_refer(file: UploadFile = File(...)):

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--video_path", type=str, default="video_data/test")
    parser.add_argument("--zh", help="data_zh", action="store_true")
    parser.add_argument("--use_batching", help="use batching optimization will speedup inference on specific hardwares", action="store_true")
    args = parser.parse_args()

    video_path = args.video_path
    print(f"Base video path is set to: {video_path}")
    use_batching = args.use_batching
    print(f"Use batching optimization: {use_batching}")

    audioModel = AudioModel()
    audioModel.loadModel("checkpoint/audio.pkl")

    renderModel = RenderModel()
    renderModel.loadModel("checkpoint/render.pth")
    pkl_path = "{}/keypoint_rotate.pkl".format(video_path)
    video_path = "{}/circle.mp4".format(video_path)
    renderModel.reset_charactor(video_path, pkl_path)
    

    uvicorn.run(app, host=args.host, port=args.port)