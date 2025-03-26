import numpy as np
import torch

# a = np.array([1,2])
# print(a.shape)
# print(a.ndim)

# b = np.array([[1,2], [3,4]])
# print(b.shape)
# print(b.ndim)

# data = [[1,2], [3,4]]
# x_data = torch.tensor(data)

# print(x_data)
# print(x_data.shape)
# print(type(x_data))
# print(x_data.ndim)
# print(x_data.dtype)

# shape = (2, 3)

# rand_tensor = torch.rand(shape)

# ones_tensor = torch.ones(shape)

# zeros_tensor = torch.zeros(shape)

# print(rand_tensor, ones_tensor, zeros_tensor)

# numpy 배열로부터 tensor 초기화

# data = [[1,2],[3,4]]

# np_array = np.load("/home/jaeyoung/CNN-LSTM-action-recognition/landmark_data/007/S001C001P001R001A007_rgb.npy")

# x_np = torch.from_numpy(np_array)

# print(x_np.dtype)


data = [[1,2],[3,4]]

x_data = torch.tensor(data, dtype=torch.float32)

# print(x_data)

# 다른 dtype으로 변환

x_data = x_data.to(torch.uint8)
# print(x_data)/

# 다른 device로 tensor옮기기

# gpu가능한지

# print(torch.cuda.is_available())

# gpu 로 tensor옮기기


x_data = x_data.to("cuda")


# 어떤 device상에 tensor가 있는지 확인하기
# print(x_data.device)

#  indexing

tensor = torch.ones(4,4)

print(f"First row: {tensor[0]}")

# tensor = tensor.to(torch.uint8)

# print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:,0]}")

print(f"lasst column: {tensor[:, -1]}")

tensor[:, 1] = 0

print(tensor)


data = np.load("/home/jaeyoung/CNN-LSTM-action-recognition/landmark_data/036/S001C001P001R001A036_rgb.npy")

tensor = torch.from_numpy(data)

print(tensor)
print(tensor.shape)
print(tensor.ndim)
print(tensor[-1])

