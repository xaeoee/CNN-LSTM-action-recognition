import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import get_dataloaders
from resnet import ResNetFeatureExtractor
from lstm import ActionRecognitionLSTM
from tqdm import tqdm

BATCH_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

root_dir = "processed_videos"
_, _, test_loader = get_dataloaders(root_dir, batch_size=BATCH_SIZE)

cnn = ResNetFeatureExtractor(feature_dim=512).to(DEVICE)
lstm = ActionRecognitionLSTM(input_size=512, hidden_size=256, num_layers=2, num_classes=6).to(DEVICE)

checkpoint = torch.load("action_recognition.pth")
cnn.load_state_dict(checkpoint['cnn_state_dict'])
lstm.load_state_dict(checkpoint['lstm_state_dict'])

cnn.eval()
lstm.eval()

def evaluate():
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")

        for videos, labels in progress_bar:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)

            batch_size, seq_len, C, H, W = videos.shape  # (batch, 30, 3, 224, 224)
            
            # CNN을 통과한 Feature 저장할 공간
            features = torch.zeros(batch_size, seq_len, 512).to(DEVICE)

            for t in range(seq_len):
                features[:, t, :] = cnn(videos[:, t, :, :, :])  # CNN 통과

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
    print(f"✅ Evaluation Completed - Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

if __name__ == "__main__":
    evaluate()
