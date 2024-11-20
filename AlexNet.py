import torch
from torch import nn
import torch.nn.functional as F

from LM2_Net import MambaLayer


class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=4, padding=2)
        self.ReLU = nn.ReLU()
        self.mamba1 = MambaLayer(dim=48)
        self.c2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.s2 = nn.MaxPool2d(2)
        self.mamba2 = MambaLayer(dim=128)
        self.c3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.s3 = nn.MaxPool2d(2)
        self.mamba3 = MambaLayer(dim=192)
        self.c4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.mamba = MambaLayer(dim=128)


        self.f9 = nn.Linear(4608, 5)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.mamba1(x)
        x = self.ReLU(self.c2(x))
        x = self.mamba2(x)
        x = self.s2(x)

        x = self.ReLU(self.c3(x))
        x = self.mamba3(x)
        x = self.s3(x)

        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.mamba(x)
        x = self.s5(x)
        # x = self.flatten(x)

        x = self.flatten(x)
        x = self.f9(x)

        x = F.dropout(x, p=0.5)

        # x = self.f6(x)
        # x = F.dropout(x, p=0.5)
        # x = self.f7(x)
        # x = F.dropout(x, p=0.5)
        # x = self.f8(x)
        # x = F.dropout(x, p=0.5)
        #
        # x = self.f9(x)
        # x = F.dropout(x, p=0.5)
        return x


if __name__ == "__mian__":
    x = torch.rand([1, 3, 224, 224])
    model = MyAlexNet()
    y = model(x)




