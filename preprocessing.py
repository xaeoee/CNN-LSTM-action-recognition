import cv2
import os
import numpy as np
import torch
from tqdm import tqdm
from module.landmark_preprocessing import preprocess_landmark
from module.video_preprocessing import preprocess_video

video_dir = "/home/jaeyoung/CNN-LSTM-action-recognition"
feature_output_root = "video_data"
landmark_output_root = "landmark_data"
previous_landmark = None

os.makedirs(feature_output_root, exist_ok=True)
os.makedirs(landmark_output_root, exist_ok=True)

class_folders = sorted([f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f)) and f.startswith("0")])

for class_name in class_folders:
    class_path = os.path.join(video_dir, class_name)
    video_files = sorted([f for f in os.listdir(class_path) if f.endswith(".avi")])

    landmark_output_dirs = os.path.join(landmark_output_root, class_name)
    video_output_dirs = os.path.join(feature_output_root, class_name)

    os.makedirs(video_output_dirs, exist_ok=True)
    os.makedirs(landmark_output_dirs, exist_ok=True)

    print(f"📂 Processing class: {class_name} ({len(video_files)} videos)")
    
    for video_name in tqdm(video_files, desc=f"Processing {class_name}"):

        video_path = os.path.join(class_path, video_name)
        
        # 전처리 수행
        video = preprocess_video(video_path)
        landmark = preprocess_landmark(video_path)

        landmark_save_path = os.path.join(landmark_output_dirs, video_name.replace('.avi', '.npy'))

        video_save_path = os.path.join(video_output_dirs, video_name.replace('.avi', '.pt'))

        np.save(landmark_save_path, landmark)
        
        torch.save(video, video_save_path)

        previous_landmark = landmark
