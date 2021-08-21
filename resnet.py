import torch.nn as nn
import torch
from torch.nn.modules import conv
from torch.nn.modules.linear import Identity
import math


def conv3x3(inplanes, planes,stride=1, groups=1, dilation=1):
    return nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=1, bias=False, groups=groups, dilation=dilation)

def conv1x1(inplanes, planes,stride=1):
    return nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride = 1, downsaple = None, groups = 1,base_width = 64, dilation = 1, norm_layer = None) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('Basicblock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplemented('Dilation >1 not supported in Baisc')
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace = True) #inplace为True表示在原地操作，一般默认为False，表示新建一个变量存储操作1
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = norm_layer(planes)
        self.downsample = downsaple
        self.stride = stride

   

    #后激活
    # def forward(self, x):

    #     identity = x #保留原始输入

    #     out = self.conv1(x)
    #     out = self.bn1(out)
    #     out = self.relu(out)
    #     out = self.conv2(out)
    #     out = self.bn2(out)

    #     if self.downsample is not None:
    #         identity = self.downsample(x)
        
    #     out += identity
    #     out = self.relu(out)

    #     return out
    

    #预激活
    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity

        return out


#瓶颈架构，50层以上会使用到
class Bottlenect(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1, base_width = 64, dilation =1, norm_layer = None):   
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes*(base_width / 64))*groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes*self.expansion)
        self.bn3 = norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity= x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += identity

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. /n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #make_layer方法的第一个输入block是Bottleneck或BasicBlock类，第二个输入是该blocks的输出channel，第三个输入是每个blocks中包含多少个residual子结构
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes*block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def generate_model(blocks, layers):
    return ResNet(blocks, layers)

if __name__ == '__main__':
    model = generate_model(Bottlenect, [6, 4, 2, 3])
    print(model)





            
        



    




