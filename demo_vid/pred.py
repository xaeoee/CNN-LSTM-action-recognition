import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from resnet import ResNetFeatureExtractor
from lstm import ActionRecognitionLSTM
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
cnn = ResNetFeatureExtractor(feature_dim=512).to(DEVICE)
lstm = ActionRecognitionLSTM(input_size=512, hidden_size=256, num_layers=1, num_classes=6).to(DEVICE)

# 저장된 체크포인트 로드
checkpoint = torch.load("action_recognition.pth")
cnn.load_state_dict(checkpoint['cnn_state_dict'])
lstm.load_state_dict(checkpoint['lstm_state_dict'])
cnn.eval()
lstm.eval()

# 라벨 인덱스 → 행동 이름 매핑
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

def realtime_action_recognition():
    # 웹캠 열기
    cap = cv2.VideoCapture(0)  # 0은 기본 웹캠
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    frame_buffer = []
    max_frames = 50
    prediction = None
    last_prediction_time = time.time()
    prediction_interval = 1.0  # 1초마다 예측
    
    # OpenCV 창 생성
    cv2.namedWindow("Real-time Action Recognition", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("웹캠에서 프레임을 읽을 수 없습니다.")
            break
        
        # 화면에 보여주기 위한 원본 프레임 복사
        display_frame = frame.copy()
        
        # 프레임 전처리
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = transform(rgb_frame)
        
        # 프레임 버퍼 관리
        frame_buffer.append(processed_frame)
        if len(frame_buffer) > max_frames:
            frame_buffer.pop(0)
        
        current_time = time.time()
        
        # 충분한 프레임이 쌓이고 일정 시간이 경과하면 예측 수행
        if len(frame_buffer) >= max_frames and (current_time - last_prediction_time) >= prediction_interval:
            # 부족한 프레임은 마지막 프레임으로 채우기 (이 경우는 필요 없지만 안전을 위해)
            if len(frame_buffer) < max_frames:
                last = frame_buffer[-1]
                frame_buffer.extend([last] * (max_frames - len(frame_buffer)))
            
            # (1, max_frames, 3, 224, 224) 형태로 배치 구성
            video_tensor = torch.stack(frame_buffer[-max_frames:]).unsqueeze(0).to(DEVICE)
            
            # CNN -> Feature 추출
            features = torch.zeros(1, max_frames, 512).to(DEVICE)
            for t in range(max_frames):
                with torch.no_grad():
                    features[:, t, :] = cnn(video_tensor[:, t, :, :, :])
            
            # LSTM -> 예측
            with torch.no_grad():
                output = lstm(features)
                _, predicted = torch.max(output, 1)
            
            prediction = predicted.item()
            print(f"🔍 예측된 클래스 인덱스: {prediction}")
            print(f"📌 예측된 행동: {label_map[prediction]}")
            
            last_prediction_time = current_time
        
        # 예측 결과를 화면에 표시
        if prediction is not None:
            action_text = f"Action: {label_map[prediction]}"
            cv2.putText(display_frame, action_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # FPS 계산 및 표시
        fps_text = f"FPS: {int(1 / (time.time() - current_time + 0.001))}"
        cv2.putText(display_frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 입력 버퍼 프레임 수 표시
        buffer_text = f"Buffer: {len(frame_buffer)}/{max_frames}"
        cv2.putText(display_frame, buffer_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 프레임 화면에 표시
        cv2.imshow("Real-time Action Recognition", display_frame)
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_action_recognition()