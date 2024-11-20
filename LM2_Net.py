import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from CCTA import CCTA
from CannyFilter import CannyFilter
from networks.EVC import EVCBlock
from networks.SEAttion import SEAttention

from networks.laplacian import HighFrequencyModule
from networks.vision_lstm import SequenceTraversal, LayerNorm
from mamba_ssm import Mamba
from networks.vision_lstm_util import DropPath
from torchutilss import visulize_features, visualize_tensors
count = 0

class ViLLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.vil = ViLBlock(
            dim=self.dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
        )


    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_vil = self.vil(x_flat)
        out = x_vil.transpose(-1, -2).reshape(B, C, *img_dims)

        return out

class ViLBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, norm_bias=False):
        super().__init__()
        self.dim = dim
        # self.direction = direction
        self.drop_path = drop_path
        self.norm_bias = norm_bias

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        # self.layer = ViLLayer(dim=dim, direction=direction)
        self.layer = MambaLayer(dim=dim)

        # self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        # print('In xlstm now')
        return x

    # def reset_parameters(self):
    #     self.layer.reset_parameters()
    #     self.norm.reset_parameters()
class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        bn_momentum = 0.1
        self.height = HighFrequencyModule(input_channel=in_channels, mode="high_boost_filtering")
        kernel_size = kernel_size
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        self.stride = stride
        self.l2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.l2h = torch.nn.Conv2d(in_channels, out_channels // 4,
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2l = torch.nn.Conv2d(in_channels, int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)
        self.h2h = torch.nn.Conv2d(in_channels,
                                   out_channels - int(alpha * out_channels),
                                   kernel_size, 1, padding, dilation, groups, bias)

        self.sigmid = nn.Sigmoid()

    def forward(self, x):
        X_h = x
        X_l = x
        X_h = self.height(X_h)

        X_h2l = self.h2g_pool(X_h)

        X_h2h = self.h2h(X_h)
        X_l2h = self.l2h(X_l)

        X_l2l = self.l2l(X_l)
        X_h2l = self.h2l(X_h2l)

        X_h2l = self.upsample(X_h2l)
        X_h = torch.cat((X_l2l, X_h2h), dim=1)
        X_l = torch.cat((X_l2h, X_h2l), dim=1)

        #
        X_h = self.sigmid(X_h)
        X_l = self.sigmid(X_l)

        # X_l = self._pre_treat_1(X_l)
        return X_h, X_l
class DWConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DWConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, padding=0, groups=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True))

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class BAM(nn.Module):
    """ Basic self-attention module
    """

    def __init__(self, in_dim, ds=8, activation=nn.ReLU):
        super(BAM, self).__init__()
        self.chanel_in = in_dim
        self.key_channel = self.chanel_in // 8
        self.activation = activation
        self.ds = ds  #
        self.pool = nn.AvgPool2d(self.ds)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, input):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = self.pool(input)
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C X (N)/(ds*ds)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)/(ds*ds)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        energy = (self.key_channel ** -.5) * energy

        attention = self.softmax(energy)  # BX (N) X (N)/(ds*ds)/(ds*ds)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = F.interpolate(out, [width * self.ds, height * self.ds])
        out = out + input

        return out


class LM2_Net(nn.Module):
    def __init__(self, freeze_bn=False):
        super(LM2_Net, self).__init__()
        # self.encoder = resnet34()   #在此处可切换backbone
        self.encoder = resnet18()
        self.decoder = Decoder()

        if freeze_bn:
            self.freeze_bn()

    def forward(self, A):
        output1 = self.encoder(A)
        result = self.decoder(output1)

        return result

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth', }


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self, model_path):
        pretrain_dict = model_zoo.load_url(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feature = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        feature.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feature.append(x)
        x = self.layer2(x)
        feature.append(x)
        x = self.layer3(x)
        feature.append(x)
        x = self.layer4(x)
        feature.append(x)
        return feature


def resnet34(pretrained=True):
    """Constructs a ResNet-34 model."""
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model._load_pretrained_model(model_urls['resnet34'])
    return model


def resnet18(pretrained=True):
    """
    output, low_level_feat:
    512, 256, 128, 64, 64
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model._load_pretrained_model(model_urls['resnet18'])
    return model

def resnet50(pretrained=True):
    """
    output, low_level_feat:
    512, 256, 128, 64, 64
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model._load_pretrained_model(model_urls['resnet50'])
    return model


class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(decoder_block, self).__init__()

        self.de_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        # self.de_block1 = OctaveConv(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.de_block2 = DWConv(out_channels, out_channels)

        self.att = CoordAtt(out_channels, out_channels)

        self.de_block3 = DWConv(out_channels, out_channels)

        self.de_block4 = nn.Conv2d(out_channels, 1, 1)
        # self.de_block4 = OctaveConv(in_channels=out_channels, out_channels=1, kernel_size=1)

        self.de_block5 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        # self.de_block5 = OctaveConv(in_channels=out_channels, out_channels=1, kernel_size=2,padding=0,stride=2)


    def forward(self, input1, input):
        x0 = torch.cat((input1, input), dim=1)
        x0 = self.de_block1(x0)
        # x = self.de_block2(x0)
        # x = self.att(x)
        # x = self.de_block3(x)
        # x = x + x0
        # al = self.de_block4(x)
        # result = self.de_block5(x)

        al = self.de_block4(x0)
        result = self.de_block5(x0)

        return al, result


class ref_seg(nn.Module):
    def __init__(self):
        super(ref_seg, self).__init__()
        self.dir_head = nn.Sequential(nn.Conv2d(32, 32, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 8, 1, 1))
        self.conv0 = nn.Conv2d(1, 8, 3, 1, 1, bias=False)
        self.conv0.weight = nn.Parameter(torch.tensor([[[[0, 0, 0], [1, 0, 0], [0, 0, 0]]],
                                                       [[[1, 0, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 1, 0], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 1], [0, 0, 0], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 1], [0, 0, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                                                       [[[0, 0, 0], [0, 0, 0], [0, 1, 0]]],
                                                       [[[0, 0, 0], [0, 0, 0], [1, 0, 0]]]]).float())

    def forward(self, x, masks_pred, edge_pred):
        direc_pred = self.dir_head(x)
        direc_pred = direc_pred.softmax(1)
        edge_mask = 1 * (torch.sigmoid(edge_pred).detach() > 0.6)
        refined_mask_pred = (self.conv0(masks_pred) * direc_pred).sum(1).unsqueeze(1) * edge_mask + masks_pred * (
                    1 - edge_mask)
        return refined_mask_pred


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)

        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.PReLU(ch_out)
        # self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        # out = self.conv2(out)
        # out = self.bn1(out)
        # out = self.relu(out)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        # out = self.bn2(out)
        out = self.relu(out)
class shallow_fea_fusion(nn.Module):
    def __init__(self, F_g, F_l, F_int, count):
        super(shallow_fea_fusion, self).__init__()
        self.count = count
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F_int),
            nn.PReLU(F_int)
        )
        # self.W_g = DeepWise_PointWise_Conv(F_g, F_int)
        # self.W_x = DeepWise_PointWise_Conv(F_l, F_int)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(F_int),
            nn.PReLU(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int * 2, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.PReLU(F_int)

        self.shallow_conv = conv_block(ch_in=F_g * count, ch_out=F_int)
        # self.shallow_conv = conv_block(ch_in=F_l * 2, ch_out=F_int)
        # self.shallow_conv = nn.Conv2d(in_channels=F_l * 2, out_channels=F_int, kernel_size=3, padding=1)
        self.conv1x1 = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0)
        # self.ccta = CCTA(64 * count) #max_iou:0.8290242073631846 epoch:26
        self.ccta = CCTA(F_int * 2)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        g1 = self.ccta(g1)
        # 上采样的 l 卷积
        x1 = self.W_x(x)

        psi = torch.cat((g1, x1), dim=1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        fea1 = x1 * psi
        fea2 = g1 * psi
        fea = torch.cat((fea1, fea2), dim=1)
        fea = self.ccta(fea)
        # fea = self.shallow_conv(fea)
        '''fea = self.ccta(fea) max_iou:0.8212976853792537 epoch:28'''
        # fea2 = self.shallow_conv(fea2)
        return fea


class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.se = SEAttention(channel=dim)



    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        # print(C)
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = self.se(out)

        return out
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.bam = BAM(512)
        self.db1 = nn.Sequential(
            nn.Conv2d(512, 512, 1), nn.BatchNorm2d(512), nn.ReLU(),
            DWConv(512, 512),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        )

        self.db2 = decoder_block(768, 256)
        self.db3 = decoder_block(384, 128)
        self.db4 = decoder_block(192, 64)
        self.db5 = decoder_block(128, 32)

        self.classifier1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))

        self.classifier2 = nn.Sequential(
            nn.Conv2d(32 + 1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.Conv2d(32, 1, 1))
        self.interpo = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine = ref_seg()
        self._init_weight()
        self.ccta = CCTA(32)
        self.ca  = CannyFilter()
        self.lap = HighFrequencyModule(32)
        self.lap1 = HighFrequencyModule(64)
        self.lap2 = HighFrequencyModule(64)
        self.lap3 = HighFrequencyModule(128)
        self.lap4 = HighFrequencyModule(512)

        self.evc = CCTA(512)
        self.mv2 = EVCBlock(in_channels=128, out_channels=128)
        self.vi = MambaLayer(dim=32)
        self.vi1 = MambaLayer(dim=64)
        self.vi2 = MambaLayer(dim=64)
        self.vi3 = MambaLayer(dim=128)
        self.vi4 = MambaLayer(dim=256)
        self.vi5 = MambaLayer(dim=512)
        self.se = SEAttention()
        # self.c = ViLLayer(dim=512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.fc = nn.Linear(32, 9)

    def forward(self, input1):


        # x1 = self.db1(input1)
        # x2 = self.lap1(x1)
        # x3 = self.lap2(x1)
        # x= x2 + x3 + x1


        input1, input2, input3, input4, input5 = input1[0], input1[1], input1[2], input1[3], input1[4]
        x = self.evc(input5)

        x = self.db1(x)
        al1, x = self.db2(input4, x)  # 256*32*32
        al2, x = self.db3(input3, x)  # 128*64*64
        al3, x = self.db4(input2, x)  # 64*128*128
        al4, x = self.db5(input1, x)  # 32*256*256
        # if epoch % 10 == 0 and type == 0:
        #     if img11 % 3000 == 0:
        #         visualize_tensors(al4, img11=img11)
        x = self.avgpool(x)
        x = self.lap(x)
        x = self.vi(x)


        x = torch.flatten(x, 1)
        x = self.fc(x)

        return  x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    # test_data1 = torch.rand(2,3,512,512).cuda()
    # test_data2 = torch.rand(2,3,512,512).cuda()
    # test_label = torch.randint(0, 2, (2,1,512,512)).cuda()
    #
    model = LM2_Net()
    # model = model.cuda()
    # output = model(test_data1)
    print(0%10)

