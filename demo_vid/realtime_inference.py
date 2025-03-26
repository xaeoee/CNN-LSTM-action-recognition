import cv2
import torch
import torchvision.transforms as transforms
from resnet import ResNetFeatureExtractor
from lstm import ActionRecognitionLSTM
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
cnn = ResNetFeatureExtractor(feature_dim=512).to(DEVICE)
lstm = ActionRecognitionLSTM(input_size=512, hidden_size=256, num_layers=1, num_classes=6).to(DEVICE)

checkpoint = torch.load("action_recognition.pth", map_location=DEVICE)
cnn.load_state_dict(checkpoint['cnn_state_dict'])
lstm.load_state_dict(checkpoint['lstm_state_dict'])

cnn.eval()
lstm.eval()

label_map = {
    0: "throw",
    1: "sit down",
    2: "stand up",
    3: "clapping",
    4: "kick",
    5: "headbanging"
}

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_from_frames(frames):
    max_frames = len(frames)
    video_tensor = torch.stack(frames).unsqueeze(0).to(DEVICE)  # (1, T, 3, 224, 224)

    features = torch.zeros(1, max_frames, 512).to(DEVICE)
    for t in range(max_frames):
        features[:, t, :] = cnn(video_tensor[:, t, :, :, :])

    with torch.no_grad():
        output = lstm(features)
        _, predicted = torch.max(output, 1)

    return predicted.item()

def real_time_action_recognition():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ 웹캠을 열 수 없습니다. 다른 장치를 시도하거나 권한을 확인하세요.")
        exit()

    frame_buffer = []
    sequence_length = 30

    print("🎥 실시간 동작 인식 시작 (종료: Q 키)")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ 프레임을 가져오지 못했습니다. 카메라를 확인하세요.")
            break

        frame = cv2.flip(frame, 1)  # 좌우 반전
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor_frame = transform(rgb_frame)

        # 프레임 버퍼 유지
        frame_buffer.append(tensor_frame)
        if len(frame_buffer) > sequence_length:
            frame_buffer.pop(0)  # 오래된 프레임 삭제

        if len(frame_buffer) == sequence_length:
            predicted_class = predict_from_frames(frame_buffer)
            action_name = label_map[predicted_class]
            print(f"📌 예측된 행동: {action_name}")

            # 화면에 출력
            cv2.putText(frame, f"Action: {action_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        # 화면에 출력
        cv2.imshow("Real-Time Action Recognition", frame)

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    real_time_action_recognition()
