import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# (CIFAR format) A simple three-layer MLP to test obfuscating FC dimension parameters
class custom_cnn_4(torch.nn.Module):
    def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):
        super(custom_cnn_4,self).__init__()
        self.reshape = reshape
        self.widen_list = widen_list
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.deepen_list = deepen_list
        self.skipcon_list = skipcon_list
        self.kerneladd_list = kerneladd_list
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool2x2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc0 = torch.nn.Linear(3072, 512)
        self.fc_bn0 = torch.nn.BatchNorm1d(512)
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc_bn1 = torch.nn.BatchNorm1d(256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc_bn2 = torch.nn.BatchNorm1d(128)
        self.classifier = torch.nn.Linear(128, 10)
        self.classifier_bn = torch.nn.BatchNorm1d(10, affine=False)
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        X1 = self.fc0(X1)
        X1 = self.fc_bn0(X1)
        X1 = self.relu(X1)
        X1 = self.fc1(X1)
        X1 = self.fc_bn1(X1)
        X1 = self.relu(X1)
        X1 = self.fc2(X1)
        X1 = self.fc_bn2(X1)
        X1 = self.relu(X1)
        X1 = self.classifier(X1)
        X1 = self.classifier_bn(X1)
        return X1

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

batch_size = 1
input_features = 3072

X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_4(input_features).to(cuda_device)

# End Call Model
model.eval()
new_out = model(X)
# print(new_out.size())