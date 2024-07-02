import cv2
import os
import shutil
from tqdm import tqdm

def save_even_video_frames(dataset_name):
    video_path = f"demo_data/{dataset_name}/rgb.mp4"
    output_dir = f"demo_data/{dataset_name}/rgb"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    saved_frame_number = 0

    with tqdm(total=total_frames, desc="Processing even frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"{frame_number:06d}.png"
            output_path = os.path.join(output_dir, frame_filename)
            cv2.imwrite(output_path, frame)
            pbar.update(1)
            frame_number += 1
    cap.release()

dataset_name = "bone_4k"
save_even_video_frames(dataset_name)