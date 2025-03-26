import torch
import torch.nn as nn
from resnet import ResNetFeatureExtractor
from lstm import ActionRecognitionLSTM
from tqdm import tqdm
from multimodal import get_multimodal_dataloaders  # 또는 dataset_loader 사용 가능

# 설정값
BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 6

# 데이터 경로
video_root = "video_data"
landmark_root = "landmark_data"

# 평가용 데이터 로더만 불러오기
_, _, test_loader = get_multimodal_dataloaders(
    video_root, landmark_root, batch_size=BATCH_SIZE,
    use_video=True, use_landmark=True
)

# 모델 불러오기
cnn = ResNetFeatureExtractor(feature_dim=512).to(DEVICE)
lstm = ActionRecognitionLSTM(input_size=536, hidden_size=256, num_layers=1, num_classes=NUM_CLASSES).to(DEVICE)

# 저장된 가중치 로드
checkpoint = torch.load("action_recognition.pth", map_location=DEVICE)
cnn.load_state_dict(checkpoint["cnn_state_dict"])
lstm.load_state_dict(checkpoint["lstm_state_dict"])

cnn.eval()
lstm.eval()

# 손실 함수
criterion = nn.CrossEntropyLoss()

def evaluate():
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")

        for videos, landmarks, labels in progress_bar:
            videos = videos.to(DEVICE)               # (B, T, C, H, W)
            landmarks = landmarks.to(DEVICE)         # (B, T, 24)
            labels = labels.to(DEVICE)               # (B,)

            B, T, C, H, W = videos.shape
            videos_reshaped = videos.view(B * T, C, H, W)
            features = cnn(videos_reshaped).view(B, T, -1)  # (B, T, 512)

            combined = torch.cat([features, landmarks], dim=-1)  # (B, T, 536)
            outputs = lstm(combined)  # (B, num_classes)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), acc=correct / total)

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    print(f"\n✅ 평가 완료 - 평균 Loss: {avg_loss:.4f}, 정확도: {accuracy*100:.2f}%")

if __name__ == "__main__":
    evaluate()
