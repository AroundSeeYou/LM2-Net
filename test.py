from datetime import datetime

import torch
from torch import nn

import numpy as np

from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from LM2_Net import zh_net
from MV2 import efficientnetv2_s, EfficientNetV2
from ResNet import ResNet
from ResNet50 import resnet50, resnet101, resnet34
from t_sne_3d import plot_tsne3d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT_TEST = r'datSets/original/test'

# 将图像RGB三个通道的像素值分别减去0.5,再除以0.5.从而将所有的像素值固定在[-1,1]范围内
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.RandomVerticalFlip(),  # 随机垂直旋转
    transforms.ToTensor(),
    normalize])

train_dataset = ImageFolder(ROOT_TEST, transform=test_transform)

test_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)



# 调用net里面的定义的网络模型， 如果GPU可用则将模型转到GPU
model = zh_net()
model.eval()  # 设置为评估模式
# 加载模型train.py里面训练的模型
model.load_state_dict(torch.load('save_model/env_last_original.pth'))



# 假设你有模型的输出和真实标签
predicted_labels = []
true_labels = []
with torch.no_grad():  # 关闭梯度计算
    correct = 0
    total = 0
    for batch, (images, labels) in enumerate(test_dataset):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, axis=1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())


accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')



# # 对val_dataset里面的照片进行推理验证
# for i in range(50):
#     x, y = val_dataset[i][0], val_dataset[i][1]
#     # show(x).show()
#     x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False).to(device)
#     x = torch.tensor(x).to(device)
#     with torch.no_grad():
#         pred = model(x)
#         predicted, actual = classes[torch.argmax(pred[0])], classes[y]
#         print(f'Predicted: "{predicted}", Actual: "{actual}"')