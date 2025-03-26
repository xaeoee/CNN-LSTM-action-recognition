import torch
import torch.nn as nn

class ActionRecognitionLSTM(nn.Module):
    def __init__(self, input_size=536, hidden_size=256, num_layers=1, num_classes=6):
        super(ActionRecognitionLSTM, self).__init__()

        # batch_first=True: 입력 텐서의 차원 순서가 (배치, 시퀀스 길이, 특징 차원)임을 나타낸다.
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        
        last_hidden = lstm_out[:, -1, :]  # 마지막 타임스텝(hidden state) 선택 → (batch, hidden_size)

        output = self.fc(last_hidden)  # 최종 클래스 예측 → (batch, num_classes)
        
        return output

if __name__ == "__main__":
    model = ActionRecognitionLSTM()
    sample_input = torch.randn(4, 30, 536)
    output = model(sample_input)
    # output.shape = torch.Size([4, 6])
    print(f"{output.shape = }")
