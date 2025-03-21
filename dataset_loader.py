import torch
import os
from torch.utils.data import Dataset, DataLoader
import random

class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # (파일 경로, 클래스 라벨) 저장할 리스트
        self.class_labels = sorted(os.listdir(root_dir))  # 폴더명 기준 정렬

        for label_idx, class_name in enumerate(self.class_labels):
            class_folder = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_folder):
                continue  # 폴더가 아니면 스킵
            
            video_files = sorted([f for f in os.listdir(class_folder) if f.endswith('.pt')])
            for video_file in video_files:
                video_path = os.path.join(class_folder, video_file)
                self.data.append((video_path, label_idx))  # (파일 경로, 라벨) 저장

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 매 iteration 마다 호출됨
        # 


        video_path, label = self.data[idx]
        video_tensor = torch.load(video_path)  # (30, 3, 224, 224) 형태 유지

        # print(f"📌 __getitem__ 호출됨: idx={idx}, label={label}")

        
        if self.transform:
            video_tensor = self.transform(video_tensor)

        return video_tensor, label  # (30, 3, 224, 224), 정수 라벨

def get_dataloaders(root_dir, batch_size=8, shuffle=True, train_split=0.8, val_split=0.1):

    dataset = VideoDataset(root_dir) #데이터셋 만들기
    total_size = len(dataset)
    
    # 데이터셋 인덱스 섞기 (랜덤 샘플링)
    indices = list(range(total_size))
    random.shuffle(indices)

    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle) #데이터를 불러올때 미니배치로 불러오기 편하게 해주는게 dataloaders
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    root_dir = "processed_videos"
    train_loader, val_loader, test_loader = get_dataloaders(root_dir)

    # 배치 샘플 확인
    sample_batch = next(iter(train_loader))
    sample_data, sample_label = sample_batch
    print(f"Batch Shape: {sample_data.shape}")  # (batch, 30, 3, 224, 224)
    print(f"Labels: {sample_label}")  # 배치 내 클래스 라벨 확인
