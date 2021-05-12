import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

# class ResNetBasicblock(nn.Module):
#   expansion = 1
#   """
#   RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
#   """
#   def __init__(self, inplanes, planes, stride=1):
#     super(ResNetBasicblock, self).__init__()

#     self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#     self.conv_bn1 = nn.BatchNorm2d(planes)

#     self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#     self.conv_bn2 = nn.BatchNorm2d(planes)

#     self.downsample = None

#   def forward(self, x):
#     residual = x
#     basicblock = self.conv1(x)
#     basicblock = self.conv_bn1(basicblock)
#     basicblock = F.relu(basicblock, inplace=True)

#     basicblock = self.conv2(basicblock)
#     basicblock = self.conv_bn2(basicblock)

#     if self.downsample is not None:
#       residual = self.downsample(x)
#     return residual + basicblock

# class DepthwiseConv(nn.Module):
#     def __init__(self, inplanes, planes, stride=1):
#         super(DepthwiseConv, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, groups=inplanes)
#         self.conv_bn1 = nn.BatchNorm2d(inplanes)
#         self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1)
#         self.conv_bn2 = nn.BatchNorm2d(planes)

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv_bn1(out)
#         out = F.relu(out, inplace=True)
#         out = self.conv2(out)
#         out = self.conv_bn2(out)
#         return out

# Model starts here

class custom_cnn_12(torch.nn.Module):
    def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):
        super(custom_cnn_12,self).__init__()
        self.reshape = reshape
        self.widen_list = widen_list
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.deepen_list = deepen_list
        self.skipcon_list = skipcon_list
        self.kerneladd_list = kerneladd_list
        self.relu = torch.nn.ReLU(inplace=True)
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool2x2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv0 = torch.nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1 = torch.nn.Conv2d(64, 256, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(256, 1024, (3, 3), stride=(1, 1), padding=(1, 1))


        self.conv3 = torch.nn.Conv2d(1024, 1024, (3, 3), stride=(1, 1), padding=(1, 1), groups=1024, bias=False)
        self.conv_bn3 = nn.BatchNorm2d(1024)
        self.conv4 = torch.nn.Conv2d(1024, 256, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn4 = nn.BatchNorm2d(256)


        self.conv5 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = torch.nn.Conv2d(256, 1024, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv8 = torch.nn.Conv2d(1024, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9 = torch.nn.Conv2d(64, 256, (3, 3), stride=(1, 1), padding=(1, 1))

        # self.conv9 = ResNetBasicblock(256, 256, stride= 1)

        self.conv10 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_bn10 = nn.BatchNorm2d(256)
        self.conv11 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_bn11 = nn.BatchNorm2d(256)

        self.conv12 = torch.nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1))

        # self.conv11 = ResNetBasicblock(512, 512, stride= 1)

        self.conv13 = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_bn13 = nn.BatchNorm2d(512)
        self.conv14 = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_bn14 = nn.BatchNorm2d(512)

        self.conv15 = torch.nn.Conv2d(512, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.fc0 = torch.nn.Linear(50176, 512)
        self.fc1 = torch.nn.Linear(512, 1024)
        self.classifier = torch.nn.Linear(1024, 1000)
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 224, 224)
        X1 = self.conv0(X1)
        X1 = self.relu(X1)
        X1 = self.conv1(X1)
        X1 = self.relu(X1)
        X1 = self.conv2(X1)
        X1 = self.avgpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv3(X1)
        X1 = self.conv_bn3(X1)
        X1 = self.relu(X1)
        X1 = self.conv4(X1)
        X1 = self.conv_bn4(X1)
        X1 = self.relu(X1)
        X1 = self.conv5(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv6(X1)
        X1 = self.relu(X1)
        X1 = self.conv7(X1)
        X1 = self.relu(X1)
        X1 = self.conv8(X1)
        X1 = self.relu(X1)
        X1 = self.conv9(X1)
        X1 = self.relu(X1)
        X0_skip = X1
        X1 = self.conv10(X1)
        X1 = self.conv_bn10(X1)
        X1 = self.relu(X1)
        X1 = self.conv11(X1)
        X1 = self.conv_bn11(X1)
        X1 = X1 + X0_skip
        X1 = self.relu(X1)
        X1 = self.conv12(X1)
        X1 = self.relu(X1)
        X0_skip = X1
        X1 = self.conv13(X1)
        X1 = self.conv_bn13(X1)
        X1 = self.relu(X1)
        X1 = self.conv14(X1)
        X1 = self.conv_bn14(X1)
        X1 = X1 + X0_skip
        X1 = self.relu(X1)
        X1 = self.conv15(X1)
        X1 = self.relu(X1)
        X1 = X1.view(-1, 50176)
        X1 = self.fc0(X1)
        X1 = self.relu(X1)
        X1 = self.fc1(X1)
        X1 = self.relu(X1)
        X1 = self.classifier(X1)
        X1 = self.logsoftmax(X1)
        return X1

batch_size = 1
input_features = 150528
torch.manual_seed(1234)
X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_12(input_features).to(cuda_device)

# End Call Model
model.eval()
new_out = model(X)
