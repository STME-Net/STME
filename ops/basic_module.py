from __future__ import print_function, division, absolute_import
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.utils.model_zoo as model_zoo

class STEModule(nn.Module):
    def __init__(self, channel, n_segment=8):
        super(STEModule, self).__init__()
        self.channel = channel
        self.reduction = 16
        self.n_segment = n_segment

        self.conv1 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel // self.reduction, kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.conv2 = nn.Conv1d(in_channels=self.channel // self.reduction, out_channels=self.channel // self.reduction,
                               kernel_size=3, stride=1, bias=False, padding=1, groups=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel // self.reduction,
                               kernel_size=3, bias=False, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.conv4 = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel, kernel_size=1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=self.channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        x = self.conv1(x)  # [nt,c/16,h,w]
        x = self.bn1(x)

        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)  # [n,t,c/16,h,w]
        x = x.permute([0, 3, 4, 2, 1])  # [n,h,w,c/16,t]
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)  # [nhw,c/16,t]

        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute([0, 4, 3, 1, 2])
        x = x.contiguous().view(nt, c, h, w)

        x = self.conv3(x)  # [nt, c/16, h, w]
        x = self.bn3(x)

        x = self.conv4(x)  # [nt, c, h, w]
        x = self.bn4(x)

        w = self.sigmoid(x)
        output = residual * w + residual
        return output

class LMEModule(nn.Module):
    def __init__(self, channel, n_segment=8, index=1):
        super(LMEModule, self).__init__()
        self.channel = channel
        self.reduction = 16  
        self.n_segment = n_segment
        self.stride = 2 ** (index - 1)

        self.conv1 = nn.Conv2d(in_channels=self.channel, out_channels=self.channel // self.reduction, kernel_size=1,
                               bias=False)  
        self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
        self.conv2 = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel // self.reduction,
                               kernel_size=3, padding=1, groups=self.channel // self.reduction,
                               bias=False)

        self.pad_left = (0, 0, 0, 0, 0, 0, 0, 1)
        self.pad_right = (0, 0, 0, 0, 0, 0, 1, 0)

        self.share_weight_conv_left = nn.Parameter(torch.randn(self.channel // self.reduction, self.channel // self.reduction, 3, 3))
        self.share_weight_conv_right = nn.Parameter(torch.randn(self.channel // self.reduction, self.channel // self.reduction, 3, 3))
        self.bn_1 = nn.BatchNorm2d(self.channel // self.reduction)
        self.bn_2 = nn.BatchNorm2d(self.channel // self.reduction)
        self.bn_3 = nn.BatchNorm2d(self.channel // self.reduction)

        self.avgpool_left = nn.AdaptiveAvgPool2d(1)  # [h,w]->[1,1]
        self.avgpool_right = nn.AdaptiveAvgPool2d(1)
        self.conv4 = nn.Conv2d(in_channels=self.channel // self.reduction, out_channels=self.channel, kernel_size=1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(num_features=self.channel)
        self.sigmoid_left = nn.Sigmoid()
        self.sigmoid_right = nn.Sigmoid()

    def forward(self, x):
        feature = self.conv1(x)  # [nt,c/16,h,w]
        feature = self.bn1(feature)

        reshape_feature = feature.view((-1, self.n_segment) + feature.size()[1:])  # [n,t,c/16,h,w]
        t_feature_left, _ = reshape_feature.split([self.n_segment - 1, 1], dim=1)  # [n,t-1,c/16,h,w]
        _, t_feature_right = reshape_feature.split([1, self.n_segment - 1], dim=1)  # [n,t-1,c/16,h,w]

        conv_feature = self.conv2(feature)  # [nt,c/16,h,w] 
        reshape_conv_feature = conv_feature.view((-1, self.n_segment) + conv_feature.size()[1:])  # [n,t,c/16,h,w]

        t1_feature_left, _ = reshape_conv_feature.split([1, self.n_segment - 1], dim=1)  # [n,t-1,c/16,h,w]
        _, t1_feature_right = reshape_conv_feature.split([self.n_segment - 1, 1], dim=1)  # [n,t-1,c/16,h,w]

        diff_feature_left = t1_feature_left - t_feature_left  # [n,t-1,c/16,h,w]
        diff_feature_right = t1_feature_right - t_feature_right


        diff_feature_pad_left = F.pad(diff_feature_left, self.pad_left, mode="constant", value=0)  # [n,t,c/16,h,w]
        diff_feature_pad_left = diff_feature_pad_left.view((-1,) + diff_feature_pad_left.size()[2:])  # [nt,c/16,h,w]
        diff_feature_pad_right = F.pad(diff_feature_right, self.pad_left, mode="constant", value=0)  # [n,t,c/16,h,w]
        diff_feature_pad_right = diff_feature_pad_right.view((-1,) + diff_feature_pad_right.size()[2:])  # [nt,c/16,h,w]

        recep1_left = nn.functional.conv2d(diff_feature_pad_left, self.share_weight_conv_left, bias=None, stride=1,padding=1, dilation=1)
        recep1_left = self.bn_1(recep1_left)
        recep2_left = nn.functional.conv2d(diff_feature_pad_left, self.share_weight_conv_left, bias=None, stride=1,padding=2, dilation=2)
        recep2_left = self.bn_2(recep2_left)
        recep3_left = nn.functional.conv2d(diff_feature_pad_left, self.share_weight_conv_left, bias=None, stride=1,padding=3, dilation=3)
        recep3_left = self.bn_3(recep3_left)

        recep1_right = nn.functional.conv2d(diff_feature_pad_right, self.share_weight_conv_right, bias=None, stride=1,padding=1, dilation=1)
        recep1_right = self.bn_1(recep1_right)
        recep2_right = nn.functional.conv2d(diff_feature_pad_right, self.share_weight_conv_right, bias=None, stride=1,padding=2, dilation=2)
        recep2_right = self.bn_2(recep2_right)
        recep3_right = nn.functional.conv2d(diff_feature_pad_right, self.share_weight_conv_right, bias=None, stride=1,padding=3, dilation=3)
        recep3_right = self.bn_3(recep3_right)

        feature_left = 1.0 / 3.0 * recep1_left + 1.0 / 3.0 * recep2_left + 1.0 / 3.0 * recep3_left  # [nt,c/16,h,w]
        feature_right = 1.0 / 3.0 * recep1_right + 1.0 / 3.0 * recep2_right + 1.0 / 3.0 * recep3_right  # [nt,c/16,h,w]
        feature_left = self.bn4(self.conv4(self.avgpool_left(feature_left)))  # [nt,c,1,1]
        feature_right = self.bn4(self.conv4(self.avgpool_right(feature_right)))  # [nt,c,1,1]

        w_left = self.sigmoid_left(feature_left) - 0.5
        w_right = self.sigmoid_right(feature_right) - 0.5
        
        w = 0.5 * w_left + 0.5 * w_right
        output = x + x * w
        return output


class ShiftModule(nn.Module):
    """1D Temporal convolutions, the convs are initialized to act as the "Part shift" layer
    """

    def __init__(self, input_channels, n_segment=8, n_div=8, mode='shift'):
        super(ShiftModule, self).__init__()
        self.input_channels = input_channels
        self.n_segment = n_segment
        self.fold_div = n_div
        self.fold = self.input_channels // self.fold_div

        self.conv = nn.Conv1d(self.fold_div * self.fold, self.fold_div * self.fold,
                              kernel_size=3, padding=1, groups=self.fold_div * self.fold,
                              bias=False)  
        if mode == 'shift':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:self.fold, 0, 2] = 1  # shift left
            self.conv.weight.data[self.fold: 2 * self.fold, 0, 0] = 1  # shift right
            if 2 * self.fold < self.input_channels:
                self.conv.weight.data[2 * self.fold:, 0, 1] = 1  # fixed
        elif mode == 'fixed':
            self.conv.weight.requires_grad = True
            self.conv.weight.data.zero_()
            self.conv.weight.data[:, 0, 1] = 1  # fixed
        elif mode == 'norm':
            self.conv.weight.requires_grad = True

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        x = x.permute(0, 3, 4, 2, 1)
        x = x.contiguous().view(n_batch * h * w, c, self.n_segment)
        x = self.conv(x)
        x = x.view(n_batch, h, w, c, self.n_segment)
        x = x.permute(0, 4, 3, 1, 2)
        x = x.contiguous().view(nt, c, h, w)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleneckShift(nn.Module):
    expansion = 4

    def __init__(self, num_segments, inplanes, planes, stride=1, downsample=None):
        super(BottleneckShift, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes)
        self.num_segments = num_segments

        self.shift = ShiftModule(planes, n_segment=self.num_segments, n_div=8, mode='shift')
        self.lme = LMEModule(planes, n_segment=self.num_segments, index=1)
        self.ste = STEModule(planes, n_segment=self.num_segments)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.shift(out)
        out1 = self.ste(out)  
        out2 = self.lme(out)  
        out = out1 + out2

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class STMEResNet(nn.Module):

    def __init__(self, num_segments, block, layers, num_classes=1000):
        self.inplanes = 64
        self.input_space = None
        self.input_size = (224, 224, 3)
        self.mean = None
        self.std = None
        self.num_segments = num_segments

        super(STMEResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.num_segments, Bottleneck, 64, layers[0])
        self.layer2 = self._make_layer(self.num_segments, block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(self.num_segments, block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(self.num_segments, block, 512, layers[3], stride=2)

        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, num_segments, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(num_segments, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(num_segments, self.inplanes, planes))

        return nn.Sequential(*layers)

    def features(self, input):
        x = self.conv1(input)
        self.conv1_input = x.clone()
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features):
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

model_urls = {
    'stme_resnet50': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet50-19c8e357.pth',
    'stme_resnet101': 'http://data.lip6.fr/cadene/pretrainedmodels/resnet101-5d3b4d8f.pth'
}

def stme_resnet50(num_segments=8, pretrained=False, num_classes=1000):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = STMEResNet(num_segments, BottleneckShift, [3, 4, 6, 3], num_classes=num_classes)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['stme_resnet50']), strict=False)
    return model
