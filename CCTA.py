import torch
import torch.nn as nn

from networks.EVC import EVCBlock


class CCTA(nn.Module):
    def __init__(self, input_channel):
        super(CCTA, self).__init__()
        self.con2d = nn.Conv2d(in_channels=input_channel*2, out_channels=input_channel, kernel_size=1, stride=1)
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=1)
        self.Softmax2d = nn.Softmax2d()
        self.ACD = ACD(input_channel, 0.8)
        self.evc = EVCBlock(in_channels=input_channel, out_channels=input_channel)

    def forward(self, inputs):
        input_dim0 = int(inputs.shape[0])
        input_dim1 = int(inputs.shape[1])
        input_dim2 = int(inputs.shape[2])
        input_dim3 = int(inputs.shape[3])

        a = inputs.permute((0, 3, 1, 2))
        a = a.reshape((input_dim0, input_dim3, input_dim2, input_dim1))
        a = self.Softmax2d(a)
        a_probs = a.permute((0, 3, 2, 1))

        b = inputs.permute((0, 2, 3, 1))
        b = b.reshape((input_dim0, input_dim2, input_dim3, input_dim1))
        b = self.Softmax2d(b)
        b_probs = b.permute((0, 3, 1, 2))
        # attention_horizontal
        # attention_vertical
        A = torch.mul(a_probs, b_probs)
        A = torch.exp(A)
        Maximum = torch.maximum(a_probs, b_probs)
        knowledgelayer = A + Maximum +inputs
        k = self.ACD([A, Maximum, knowledgelayer])
        k = self.evc(k)

        return k



class ACD(nn.Module):
    def __init__(self, units, Thr, **kwargs):
        super(ACD, self).__init__(**kwargs)
        self.units = units
        self.Thr = Thr


    def forward(self, x):
        assert isinstance(x, list)
        # H1,V1 ,H2,V2 ,X1,X2= x
        H1, V1, X1 = x

        cos1 = torch.multiply(torch.sqrt(torch.multiply(H1, H1)), torch.sqrt(torch.multiply(V1, V1)))
        cosin1 = torch.multiply(H1, V1) / cos1

        # cos2 = torch.multiply(torch.sqrt(torch.multiply(H2, H2)), torch.sqrt(torch.multiply(V2, V2)))
        # cosin2 = torch.multiply(H2, V2) / cos2

        Zeos = torch.zeros_like(X1)
        Ones = torch.ones_like(X1)

        # print(self.Thr)

        Y = torch.where(cosin1 > self.Thr, Ones, Zeos)
        # Y1 = torch.where(cosin2 > self.Thr, x=Ones, y=Zeos)


        # concatenate = torch.concatenate([Y * X1,  Y1 * X2], axis=3)
        concatenate = torch.cat([Y * X1], axis=3)

        return concatenate



# if __name__ == '__main__':
#     x = torch.rand(2,3,4,4)
#     # print(x)
#     Cam = CCTA(input_channel=3)
#     # Cam.cuda()
#     out1 = Cam(x)
#     print(out1.shape)