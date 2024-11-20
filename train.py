from datetime import datetime

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn

import numpy as np

from torch.optim import lr_scheduler
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from LM2_Net import  LM2_Net

from networks.rs18 import ResNet18


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT_TRAIN = r'datSets/AppleLeaf9/train'
ROOT_TEST = r'datSets/AppleLeaf9/val'

# 将图像RGB三个通道的像素值分别减去0.5,再除以0.5.从而将所有的像素值固定在[-1,1]范围内
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 裁剪为224*224
    transforms.RandomVerticalFlip(),  # 随机垂直旋转
    transforms.ToTensor(),  # 将0-255范围内的像素转为0-1范围内的tensor
    normalize])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])

train_dataset = ImageFolder(ROOT_TRAIN, transform=train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform=val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=16)

# 如果显卡可用，则用显卡进行训练
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
model_cnf = [[2, 3, 1, 1, 24, 24, 0, 0],
             [4, 3, 2, 4, 24, 48, 0, 0],
             [4, 3, 2, 4, 48, 64, 0, 0],
             [6, 3, 2, 4, 64, 128, 1, 0.25],
             [9, 3, 1, 6, 128, 160, 1, 0.25],
             [15, 3, 2, 6, 160, 256, 1, 0.25]]
# 调用net里面的定义的网络模型， 如果GPU可用则将模型转到GPU
# model = EfficientNetV2(model_cnf).to(device)
# model = EfficientNetV2(model_cnf).to(device)
model = LM2_Net().to(device)
# 定义损失函数（交叉熵损失）
loss_fn = nn.BCEWithLogitsLoss()
# loss_fn = nn.MSELoss()
# loss_fn = nn.MultiLabelSoftMarginLoss()
# loss_fn = nn.BCEWithLogitsLoss()

# 定义优化器（SGD）
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=1e-4)

Emotion_kinds = 9
conf_matrixs = torch.zeros(Emotion_kinds, Emotion_kinds)

# 学习率每隔10epoch变为原来的0.1
# lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
def confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

# 定义训练函数
predicted_labels1 = []
true_labels1 = []
def train(epoch, dataloader, model, loss_fn, optimizer):
    data_collect = torch.empty(0, 3).to(device)

    label_collect = torch.empty(0, 3).to(device)
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        # 前向传播
        image, y = x.to(device), y.to(device)
        output = model(image)

        ou = torch.max(output, 1)
        # print(ou)
        # print(y)
        # data_collect = torch.cat((data_collect, ou), dim=0)
        # label_collect = torch.cat((label_collect, y), dim=0)

        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis=1)
        cur_acc = torch.sum(y == pred) / output.shape[0]

        # 反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
        _, predicted = torch.max(output, axis=1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')



    data_collect = data_collect.to('cpu').detach().numpy()
    label_collect = label_collect.to('cpu').detach().numpy()
    # plot_tsne3d(features=data_collect, labels=label_collect, epoch=epoch,
    #             fileNameDir=os.path.join('result', "Resnet50++", 'image'), num_classes=3)

    train_loss = loss / n
    tran_acc = current / n
    print('train_loss:' + str(train_loss))

    f = open('rs18_ccta_AppleLeaf9_train' + '.txt', 'a')
    f.write("epoch\":\"" + "{}\"\n".format(epoch + 1))
    f.write("train_loss\":\"" + "{}\"\n".format(train_loss))
    f.write("Accuracy\":\"" + "{}\"\n".format(accuracy))
    f.write("Precision\":\"" + "{}\"\n".format(precision))
    f.write("Recall\":\"" + "{}\"\n".format(recall))
    f.write("F1 Score\":\"" + "{}\"\n".format(f1))
    f.write("\n")

    return train_loss, tran_acc


# 定义测试函数
predicted_labels = []
true_labels = []

def val(epoch, dataloader, model, loss_fn):
    # 将模型转为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]
            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
            _, predicted = torch.max(output, axis=1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(y.cpu().numpy())

        val_loss = loss / n
        val_acc = current / n
        print('val_loss:' + str(val_loss))
        # print('val_acc:' + str(val_acc))

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='macro')
        recall = recall_score(true_labels, predicted_labels, average='macro')
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        f = open('rs18_ccta_AppleLeaf9_val' + '.txt', 'a')
        f.write("epoch\":\"" + "{}\"\n".format(epoch+1))
        f.write("loss\":\"" + "{}\"\n".format(val_loss))
        f.write("Accuracy\":\"" + "{}\"\n".format(accuracy))
        f.write("Precision\":\"" + "{}\"\n".format(precision))
        f.write("Recall\":\"" + "{}\"\n".format(recall))
        f.write("F1 Score\":\"" + "{}\"\n".format(f1))
        f.write("\n")

        conf_matrix = confusion_matrix(output, y, conf_matrixs)
        conf_matrix = conf_matrix.cpu()

        conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
        corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
        per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数

        # print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), test_num))
        print(conf_matrix)

        # 获取每种Emotion的识别准确率
        print("每类总个数：", per_kinds)
        print("每类预测正确的个数：", corrects)
        print("每类的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))
        return val_loss, val_acc


# 画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    # plt.title("训练集和验证集loss值对比图")
    plt.show()


def matplot_acc(train_acc, val_acc):
    plt.plot(train_acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.legend(loc='best')
    plt.ylabel('acc', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    # plt.title("训练集和验证集精确度值对比图")
    plt.show()


# 开始训练
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 300
min_acc = 0
print('时间: ', datetime.now())
for t in range(epoch):
    # lr_scheduler.step()
    print(f"epoch{t + 1}\n--------------")
    train_loss, train_acc = train(t, train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(t,val_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    # 保存最好的模型权重文件
    if val_acc > min_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        min_acc = val_acc
        print(f'save best model,第{t + 1}轮')
        torch.save(model.state_dict(), 'save_model/rs18_ccta_best_AppleLeaf9.pth')
    # 保存最后的权重模型文件
    if t == epoch - 1:
        torch.save(model.state_dict(), 'save_model/rs18_ccta_last_AppleLeaf9.pth')
print('---------------Over---------------！')
print('时间: ', datetime.now())

matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)





