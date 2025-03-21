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

def train():
    # train과 evaluate 모드가 있는데, 지금은 train모드를 설정하는것이다.
    # 이유는 dropout이나 batch normalization과 같은 layer들이 train과 evaluate 모드에 따라 동작이 달라지기 때문이다.
    cnn.train()
    lstm.train()
    
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0, 0, 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for videos, labels in progress_bar:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)

            batch_size, seq_len, C, H, W = videos.shape  # (batch, 30, 3, 224, 224)
            
            # CNN을 통과한 Feature 저장할 공간
            features = torch.zeros(batch_size, seq_len, 512).to(DEVICE)

            for t in range(seq_len):
                features[:, t, :] = cnn(videos[:, t, :, :, :])  # CNN 통과

            # LSTM 통과
            # 8 x 6
            outputs = lstm(features)
            
            # Loss 계산
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

        acc = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

if __name__ == "__main__":
    train()
    torch.save({
        'cnn_state_dict': cnn.state_dict(),
        'lstm_state_dict': lstm.state_dict()
    }, "action_recognition.pth")
    print("모델 학습 완료 & 저장됨!")
