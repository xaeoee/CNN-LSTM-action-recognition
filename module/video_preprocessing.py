import cv2
import numpy as np
import torch

def preprocess_video(video_path, frame_size=(224, 224), max_frames=50):

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