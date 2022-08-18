import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# Model starts here

class cnn_256p3p1p1p1_512p3p1p1p1_16p3p1p1p2_256p3p1p1p2_64p3p1p1p0_bn1_mlp_128_128_bn1(torch.nn.Module):
    def __init__(self, input_features, reshape = True, decompo_list = None, dummy_list = None):
        super(cnn_256p3p1p1p1_512p3p1p1p1_16p3p1p1p2_256p3p1p1p2_64p3p1p1p0_bn1_mlp_128_128_bn1,self).__init__()
        self.reshape = reshape
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.relu = torch.nn.ReLU(inplace=True)
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool2x2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv0 = torch.nn.Conv2d(3, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn0 = torch.nn.BatchNorm2d(256)
        self.conv1 = torch.nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn1 = torch.nn.BatchNorm2d(512)
        self.conv2 = torch.nn.Conv2d(512, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn4 = torch.nn.BatchNorm2d(64)
        self.fc0 = torch.nn.Linear(256, 128)
        self.fc_bn0 = torch.nn.BatchNorm1d(128)
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc_bn1 = torch.nn.BatchNorm1d(128)
        self.classifier = torch.nn.Linear(128, 10)
        self.classifier_bn = torch.nn.BatchNorm1d(10, affine=False)
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 32, 32)
        X1 = self.conv0(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn0(X1)
        X1 = self.conv1(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn1(X1)
        X1 = self.conv2(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn2(X1)
        X1 = self.conv3(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn3(X1)
        X1 = self.conv4(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn4(X1)
        X1 = X1.view(-1, 256)
        X1 = self.fc0(X1)
        X1 = self.fc_bn0(X1)
        X1 = self.relu(X1)
        X1 = self.fc1(X1)
        X1 = self.fc_bn1(X1)
        X1 = self.relu(X1)
        X1 = self.classifier(X1)
        X1 = self.classifier_bn(X1)
        X1 = self.logsoftmax(X1)
        return X1

# Model ends here

class CNN4(torch.nn.Module):
    def __init__(self, input_features):
        super(CNN4,self).__init__()
        self.relu = torch.nn.ReLU()
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.cnn1 = torch.nn.Conv2d(1, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.cnn2 = torch.nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = torch.nn.BatchNorm2d(16)
        self.cnn3 = torch.nn.Conv2d(16, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bn3 = torch.nn.BatchNorm2d(32)
        self.cnn4 = torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = torch.nn.BatchNorm2d(32)
        # self.cnn1 = torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1))
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, input):
        X1 = self.cnn1(input.reshape(-1, 1, 28, 28))

        X1 = self.relu(X1)
        X1 = self.bn1(X1)
        # X1 = self.maxpool(X1)
        X1 = self.cnn2(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.bn2(X1)
        print(X1.size())

        X1 = self.cnn3(X1)
        X1 = self.relu(X1)
        X1 = self.bn3(X1)
        print(X1.size())

        X1 = self.cnn4(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.bn4(X1)
        print(X1.size())

        return X1
class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(ResNetBasicblock, self).__init__()

    self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn_a = nn.BatchNorm2d(planes)

    self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn_b = nn.BatchNorm2d(planes)

    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = F.relu(basicblock, inplace=True)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)

    return F.relu(residual + basicblock, inplace=True)

class DepthwiseConv(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DepthwiseConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, groups=inplanes, bias=False)
        self.conv_bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv_bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.conv_bn2(out)
        return out

def run_torch():
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = 1
    input_features = 3072

    X = torch.randn(batch_size, input_features, device=cuda_device)

    # Start Call Model

    model = cnn_256p3p1p1p1_512p3p1p1p1_16p3p1p1p2_256p3p1p1p2_64p3p1p1p0_bn1_mlp_128_128_bn1(input_features).to(cuda_device)

    # End Call Model
    model.eval()
    new_out = model(X)

    #     if model_name == "mlp":
    #         mlp = MLP4(input_features).to(cuda_device)
    #         new_out = mlp(X)
    #     elif model_name == "cnn":
    #         cnn = CNN4(input_features).to(cuda_device)
    #         new_out = cnn(X)
    # else:
    #     X = torch.randn(batch_size, input_features)
    #     if model_name == "mlp":
    #         mlp = MLP4(input_features)
    #         new_out = mlp(X)
    #     elif model_name == "cnn":
    #         cnn = CNN4(input_features)
    #         new_out = cnn(X)
