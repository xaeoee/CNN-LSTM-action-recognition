import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import get_dataloaders
from resnet import ResNetFeatureExtractor
from lstm import ActionRecognitionLSTM
from tqdm import tqdm 

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "processed_videos"
train_loader, val_loader, test_loader = get_dataloaders(root_dir, batch_size=BATCH_SIZE)

cnn = ResNetFeatureExtractor(feature_dim=512).to(DEVICE)
lstm = ActionRecognitionLSTM(input_size=512, hidden_size=256, num_layers=1, num_classes=6).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(list(cnn.parameters()) + list(lstm.parameters()), lr=LEARNING_RATE)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train():
    """CNN-LSTM 모델 학습 및 평가 함수"""

    for epoch in range(EPOCHS):
        # 학습 모드 설정 (Dropout/BatchNorm 활성화)
        cnn.train()
        lstm.train()

        total_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for videos, labels in progress_bar:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            batch_size, seq_len, C, H, W = videos.shape  # (batch, 30, 3, 224, 224)

            # CNN 연산 최적화 (reshape 후 병렬 처리)
            videos_reshaped = videos.view(batch_size * seq_len, C, H, W)  # (batch*seq, 3, 224, 224)
            features_reshaped = cnn(videos_reshaped)  # (batch*seq, 512)
            features = features_reshaped.view(batch_size, seq_len, -1)  # (batch, seq, 512)

            # LSTM 통과
            outputs = lstm(features)

            # Loss 계산 및 역전파
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accuracy 계산
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(loss=loss.item(), acc=correct / total)

        # 평가 모드 전환 (Dropout/BatchNorm 비활성화)
        cnn.eval()
        lstm.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc="Evaluating")

            for videos, labels in progress_bar:
                videos, labels = videos.to(DEVICE), labels.to(DEVICE)
                batch_size, seq_len, C, H, W = videos.shape  # (batch, 30, 3, 224, 224)

                # CNN 연산 최적화 (reshape 후 병렬 처리)
                videos_reshaped = videos.view(batch_size * seq_len, C, H, W)
                features_reshaped = cnn(videos_reshaped)
                features = features_reshaped.view(batch_size, seq_len, -1)

                # LSTM 통과
                outputs = lstm(features)

                # Loss 계산
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                # Accuracy 계산
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix(loss=loss.item(), acc=correct / total)

        acc = correct / total
        avg_loss = total_loss / len(test_loader)
        print(f"Evaluation Completed - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

        # 평가 후 다시 학습 모드 복귀 (중요)
        cnn.train()
        lstm.train()


if __name__ == "__main__":
    train()
    torch.save({
        'cnn_state_dict': cnn.state_dict(),
        'lstm_state_dict': lstm.state_dict()
    }, "action_recognition.pth")
    print("모델 학습 완료 & 저장됨!")
