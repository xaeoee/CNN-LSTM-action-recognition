import torch
import torch.nn as nn
import torchvision.models as models

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim=512, fine_tune=True):
        super(ResNetFeatureExtractor, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        
        # Fully Connected Layer 변경 → feature_dim 크기로 변환
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, feature_dim)

        if not fine_tune:
            for param in self.resnet.parameters():
                param.requires_grad = False  # 모든 가중치 고정 (Feature Extractor로만 사용)

    def forward(self, x):
        x = self.resnet(x)  # ResNet을 통과한 feature vector 반환
        return x

if __name__ == "__main__":
    model = ResNetFeatureExtractor()
    # 30프레임, 3채널, 224 by 224
    sample_input = torch.randn(30, 3, 224, 224)
    output = model(sample_input)
    # output.shape = torch.Size([30, 512])
    print(f"{output.shape = }")
