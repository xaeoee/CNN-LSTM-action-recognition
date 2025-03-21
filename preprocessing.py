import cv2
import os
import numpy as np
import torch
from tqdm import tqdm

video_dir = "/home/jaeyoung/creamo/pose_detection"
output_dir = "processed_videos"
os.makedirs(output_dir, exist_ok=True)

class_folders = sorted([f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))])

def preprocess_video(video_path, frame_size=(224, 224), max_frames=150):

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, frame_size)  # 크기 조정
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR → RGB 변환
        frame = frame / 255.0  # 정규화 (0~1 사이)
        frames.append(frame)

    cap.release()
    
    # 부족한 프레임 0 패딩 프레임 바꾸기
    while len(frames) < max_frames:
        frames.append(np.zeros((frame_size[0], frame_size[1], 3), dtype=np.float32))

    frames = np.array(frames)  # (T, H, W, C)
    frames = np.transpose(frames, (0, 3, 1, 2))  # (T, H, W, C) → (T, C, H, W)

    return torch.tensor(frames, dtype=torch.float32)  # PyTorch Tensor 변환

for class_name in class_folders:
    class_path = os.path.join(video_dir, class_name)
    video_files = sorted([f for f in os.listdir(class_path) if f.endswith(".avi")])

    class_output_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    print(f"📂 Processing class: {class_name} ({len(video_files)} videos)")
    
    for video_name in tqdm(video_files, desc=f"Processing {class_name}"):
        video_path = os.path.join(class_path, video_name)
        
        # 전처리 수행
        features = preprocess_video(video_path)
        
        save_path = os.path.join(class_output_dir, video_name.replace('.avi', '.pt'))
        
        torch.save(features, save_path)
