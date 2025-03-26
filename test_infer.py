import cv2
import torch
import torchvision.transforms as transforms
from resnet import ResNetFeatureExtractor
from lstm import ActionRecognitionLSTM
import mediapipe as mp
import numpy as np
import torch.nn.functional as F

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

# 랜드마크 추출 함수 (11~22번 인덱스 사용 → 총 24차원)
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

# 예측 함수
def predict_action(video_frames, landmark_sequence):
    video_tensor = torch.stack(video_frames).unsqueeze(0).to(DEVICE)         # (1, 50, 3, 224, 224)
    landmark_tensor = torch.stack(landmark_sequence).unsqueeze(0).to(DEVICE) # (1, 50, 24)

    B, T, C, H, W = video_tensor.shape
    video_reshaped = video_tensor.view(B * T, C, H, W)                        # (50, 3, 224, 224)

    with torch.no_grad():
        cnn_features = cnn(video_reshaped)              # (50, 512)
        cnn_features = cnn_features.view(B, T, -1)      # (1, 50, 512)
        combined = torch.cat([cnn_features, landmark_tensor], dim=-1)  # (1, 50, 536)
        output = lstm(combined)                         # (1, 6)
        probs = F.softmax(output, dim=1)                # (1, 6)
        predicted_class = torch.argmax(probs, dim=1).item()

    return predicted_class, probs.squeeze().cpu().numpy()

# 영상에서 50프레임 뽑고 추론
def infer_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 영상 열기 실패: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, SEQ_LEN, dtype=int)

    frame_buffer = []
    landmark_buffer = []

    frame_id = 0
    target_id = 0

    while cap.isOpened() and len(frame_buffer) < SEQ_LEN:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id == indices[target_id]:
            frame = cv2.flip(frame, 1)  # 좌우 반전
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor_frame = transform(rgb_frame)
            landmark_tensor = extract_landmark_tensor_from_frame(frame)

            frame_buffer.append(tensor_frame)
            landmark_buffer.append(landmark_tensor)

            target_id += 1

        frame_id += 1

    cap.release()

    # ⚠️ 프레임 수 부족하면 패딩으로 채우기
    if len(frame_buffer) < SEQ_LEN:
        print(f"⚠️ 프레임 수 부족: {len(frame_buffer)}개 → {SEQ_LEN}개로 패딩합니다.")
        last_frame = frame_buffer[-1]
        last_landmark = landmark_buffer[-1]
        while len(frame_buffer) < SEQ_LEN:
            frame_buffer.append(last_frame.clone())
            landmark_buffer.append(last_landmark.clone())

    predicted_class, class_probs = predict_action(frame_buffer, landmark_buffer)

    print("\n📊 클래스별 확률:")
    for idx, prob in enumerate(class_probs):
        print(f"  {label_map[idx]:<12}: {prob*100:.2f}%")
    print(f"\n🎯 최종 예측: {label_map[predicted_class]} ({class_probs[predicted_class]*100:.2f}%)")


# ▶ 실행
if __name__ == "__main__":
    video_path = "/home/jaeyoung/CNN-LSTM-action-recognition/clap/Veoh_Alpha_Dog_1_clap_u_nm_np1_fr_goo_62.avi"  # ⬅️ 분석할 영상 경로 지정
    infer_from_video(video_path)
