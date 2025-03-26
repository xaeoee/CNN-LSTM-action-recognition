import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random

class MultimodalDataset(Dataset):
    def __init__(self, video_root, landmark_root, use_video=True, use_landmark=True, transform=None):
        self.use_video = use_video
        self.use_landmark = use_landmark
        self.video_root = video_root
        self.landmark_root = landmark_root
        self.transform = transform
        self.data = []

        # 클래스 이름은 video_root 기준으로 정렬
        self.class_labels = sorted(os.listdir(video_root))

        for label_idx, class_name in enumerate(self.class_labels):
            video_class_dir = os.path.join(video_root, class_name)
            landmark_class_dir = os.path.join(landmark_root, class_name)

            if not os.path.isdir(video_class_dir) or not os.path.isdir(landmark_class_dir):
                continue

            video_files = sorted([f for f in os.listdir(video_class_dir) if f.endswith('.pt')])
            landmark_files = sorted([f for f in os.listdir(landmark_class_dir) if f.endswith('.npy')])

            # 이름 매칭이 정확히 되도록 zip (전처리 과정에서 이름이 동일해야 함)
            for vf, lf in zip(video_files, landmark_files):
                video_path = os.path.join(video_class_dir, vf)
                landmark_path = os.path.join(landmark_class_dir, lf)
                self.data.append((video_path, landmark_path, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, landmark_path, label = self.data[idx]

        video_tensor = None
        landmark_tensor = None

        if self.use_video:
            video_tensor = torch.load(video_path)  # (50, 3, 224, 224)
            if self.transform:
                video_tensor = self.transform(video_tensor)

        if self.use_landmark:
            landmark_array = np.load(landmark_path)  # (50, 24)
            landmark_tensor = torch.tensor(landmark_array, dtype=torch.float32)

        return video_tensor, landmark_tensor, label


def get_multimodal_dataloaders(video_root, landmark_root, batch_size=8, shuffle=True, train_split=0.8, val_split=0.1, use_video=True, use_landmark=True):
    dataset = MultimodalDataset(video_root, landmark_root, use_video, use_landmark)
    total_size = len(dataset)

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

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    video_root = "video_data"
    landmark_root = "landmark_data"

    # 영상 + 랜드마크 같이 사용할 경우
    train_loader, val_loader, test_loader = get_multimodal_dataloaders(video_root, landmark_root)

    # 배치 테스트
    video_batch, landmark_batch, labels = next(iter(train_loader))
    print(f"Video shape: {video_batch.shape if video_batch is not None else 'Not used'}")
    print(f"Landmark shape: {landmark_batch.shape if landmark_batch is not None else 'Not used'}")
    print(f"Labels: {labels}")
