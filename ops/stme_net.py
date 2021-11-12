import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.init import normal_, constant_
from ops.basic_module import *

class STME_Net(nn.Module):
    def __init__(self, resnet_model, resnet_model1, apha, belta):
        super(STME_Net, self).__init__()

        self.avg_diff = nn.AvgPool2d(kernel_size=2, stride=2)  # AvgPool

        self.share_weight_conv = nn.Parameter(torch.randn(12, 12, 3, 3)) #in_channels=12,out_channels=12,kernel_size=3*3
        self.bn_1 = nn.BatchNorm2d(12)
        self.bn_2 = nn.BatchNorm2d(12)
        self.bn_3 = nn.BatchNorm2d(12)

        self.conv = nn.Conv2d(in_channels=12, out_channels=64, kernel_size=1, bias=False)
        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=4, padding=1, dilation=1, ceil_mode=False)

        self.res_stage2 =nn.Sequential(*list(resnet_model1.children())[4])

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = list(resnet_model.children())[8]

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        x1, x2, x3, x4, x5 = x[:, 0:3, :, :], x[:, 3:6, :, :], x[:, 6:9, :, :], x[:, 9:12, :, :], x[:, 12:15, :, :]
        x_c5 = self.avg_diff(torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], 1).view(-1, 12, x2.size()[2], x2.size()[3]))
        
        recep1 = nn.functional.conv2d(x_c5, self.share_weight_conv, bias=None, stride=1,padding=1, dilation=1)
        recep1 = self.bn_1(recep1)
        recep2 = nn.functional.conv2d(x_c5, self.share_weight_conv, bias=None, stride=1,padding=2, dilation=2)
        recep2 = self.bn_2(recep2)
        recep3 = nn.functional.conv2d(x_c5, self.share_weight_conv, bias=None, stride=1,padding=3, dilation=3)
        recep3 = self.bn_3(recep3)

        x_diff = 1.0 / 3.0 * recep1 + 1.0 / 3.0 * recep2 + 1.0 / 3.0 * recep3  # [64,8,112,112]

        x_diff = self.conv(x_diff)

        x_diff = self.maxpool_diff(1.0 / 1.0 * x_diff)
        stage2_diff = self.res_stage2(x_diff)


        x = self.conv1(x3)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])
        x = self.apha * x + self.belta * x_diff

        x = self.layer1_bak(x)
        stage2_diff = F.interpolate(stage2_diff, x.size()[2:])
        x = self.apha*x + self.belta*stage2_diff

        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def stme_net(base_model=None, num_segments=8, pretrained=True, **kwargs):
    if ("50" in base_model):
        resnet_model = stme_resnet50(num_segments, pretrained)
        resnet_model1 = stme_resnet50(num_segments, pretrained)

    if (num_segments is 8):
        model = STME_Net(resnet_model, resnet_model1, apha=0.5, belta=0.5)
    else:
        model = STME_Net(resnet_model, resnet_model1, apha=0.75, belta=0.25)
    return model
