import cv2
import torch
import torchvision.transforms as transforms
from resnet import ResNetFeatureExtractor
from lstm import ActionRecognitionLSTM
import mediapipe as mp
import numpy as np
import torch.nn.functional as F
import os

# DEVICE 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 하이퍼파라미터
SEQ_LEN = 50
INPUT_SIZE = 536  # 512 (CNN feature) + 24 (landmark)

# 클래스 라벨 매핑
label_map = {
    0: "throw",
    1: "sit down",
    2: "stand up",
    3: "clapping",
    4: "kick",
    5: "headbanging"
}

# 모델 로딩
cnn = ResNetFeatureExtractor(feature_dim=512).to(DEVICE)
lstm = ActionRecognitionLSTM(input_size=INPUT_SIZE, hidden_size=256, num_layers=1, num_classes=6).to(DEVICE)

checkpoint = torch.load("action_recognition.pth", map_location=DEVICE)
cnn.load_state_dict(checkpoint['cnn_state_dict'])
lstm.load_state_dict(checkpoint['lstm_state_dict'])

cnn.eval()
lstm.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 랜드마크 추출 함수: 인덱스 11~22 사용 (12개 포인트 × 2 = 24차원)
def extract_landmark_tensor_from_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        coords = [(landmarks[i].x, landmarks[i].y) for i in range(11, 23)]
        coords_flat = [c for pair in coords for c in pair]  # flatten (24,)
        return torch.tensor(coords_flat, dtype=torch.float32)
    else:
        return torch.zeros(24, dtype=torch.float32)

def predict_action(video_frames, landmark_sequence):
    video_tensor = torch.stack(video_frames).unsqueeze(0).to(DEVICE)         # (1, 50, 3, 224, 224)
    landmark_tensor = torch.stack(landmark_sequence).unsqueeze(0).to(DEVICE) # (1, 50, 24)

    B, T, C, H, W = video_tensor.shape                                        # B=1, T=50
    video_reshaped = video_tensor.view(B * T, C, H, W)                        # (50, 3, 224, 224) cnn 들어갈때 B C H W

    with torch.no_grad():
        cnn_features = cnn(video_reshaped)              # (50, 512)
        cnn_features = cnn_features.view(B, T, -1)      # (1, 50, 512)
        combined = torch.cat([cnn_features, landmark_tensor], dim=-1)  # (1, 50, 536)
        output = lstm(combined)                         # (1, 6)
        probs = F.softmax(output, dim=1)                # (1, 6)
        predicted_class = torch.argmax(probs, dim=1).item()  # 예측 인덱스

    return predicted_class, probs.squeeze().cpu().numpy()

# 실시간 웹캠 루프
def real_time_action_recognition():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    frame_buffer = []
    landmark_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 수신 실패")
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = transform(rgb_frame)  # (3, 224, 224)

        # print("type :", type(tensor_frame))             # <class 'torch.Tensor'>
        # print("shape:", tensor_frame.shape)             # torch.Size([3, 224, 224])
        # print("min  :", tensor_frame.min().item())      # 최소값 0
        # print("max  :", tensor_frame.max().item())      # 최대값 1


        # 랜드마크 추출 (24,)
        landmark_tensor = extract_landmark_tensor_from_frame(frame)

        # 버퍼에 저장
        frame_buffer.append(tensor_frame)
        landmark_buffer.append(landmark_tensor)

        # 예측 실행
        if len(frame_buffer) == SEQ_LEN:
            predicted_class, class_probs = predict_action(frame_buffer, landmark_buffer)
            action_name = label_map[predicted_class]

            # 화면에 표시
            cv2.putText(frame, f"Action: {action_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            # 터미널에 클래스별 확률 출력
            print("\n📊 클래스별 확률:")
            for idx, prob in enumerate(class_probs):
                print(f"  {label_map[idx]:<12}: {prob*100:.2f}%")
            print(f"🎯 최종 예측: {action_name} ({class_probs[predicted_class]*100:.2f}%)")
            frame_buffer.pop(0)
            landmark_buffer.pop(0)

        # 결과 창 띄우기
        cv2.imshow("Real-Time Action Recognition (6 classes)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    real_time_action_recognition()
