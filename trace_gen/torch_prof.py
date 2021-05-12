import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# Model starts here

class cnn_16p3p1p1p1_1024p3p1p1p0_32p3p1p1p0_16p3p1p1p0_bn1_mlp_4096_256_256_256_bn0(torch.nn.Module):
    def __init__(self, input_features):
        super(cnn_16p3p1p1p1_1024p3p1p1p0_32p3p1p1p0_16p3p1p1p0_bn1_mlp_4096_256_256_256_bn0,self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = torch.nn.Conv2d(3, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn0 = torch.nn.BatchNorm2d(16)
        self.conv1 = torch.nn.Conv2d(16, 1024, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn1 = torch.nn.BatchNorm2d(1024)
        self.conv2 = torch.nn.Conv2d(1024, 32, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn2 = torch.nn.BatchNorm2d(32)
        self.conv3 = torch.nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn3 = torch.nn.BatchNorm2d(16)
        self.fc0 = torch.nn.Linear(200704, 4096)
        self.fc1 = torch.nn.Linear(4096, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        self.classifier = torch.nn.Linear(256, 1000)
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, input):
        X1 = self.conv0(input.reshape(-1, 3, 224, 224))
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn0(X1)
        X1 = self.conv1(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn1(X1)
        X1 = self.conv2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn2(X1)
        X1 = self.conv3(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn3(X1)
        X1 = X1.view(1, -1)
        X1 = self.fc0(X1)
        X1 = self.relu(X1)
        X1 = self.fc1(X1)
        X1 = self.relu(X1)
        X1 = self.fc2(X1)
        X1 = self.relu(X1)
        X1 = self.fc3(X1)
        X1 = self.relu(X1)
        X1 = self.classifier(X1)
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

def run_torch():
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = 1
    input_features = 150528

    X = torch.randn(batch_size, input_features, device=cuda_device)

    # Start Call Model

    model = cnn_16p3p1p1p1_1024p3p1p1p0_32p3p1p1p0_16p3p1p1p0_bn1_mlp_4096_256_256_256_bn0(input_features).to(cuda_device)

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
