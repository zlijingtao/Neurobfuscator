import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# (CIFAR format) A simple two-layer CNN to test obfuscating convolution hyperparameters
class custom_cnn_2(torch.nn.Module):
    def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):
        super(custom_cnn_2,self).__init__()
        self.reshape = reshape
        self.widen_list = widen_list
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.deepen_list = deepen_list
        self.skipcon_list = skipcon_list
        self.kerneladd_list = kerneladd_list
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv0 = torch.nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1 = torch.nn.Conv2d(64, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn1 = torch.nn.BatchNorm2d(256)
        self.conv2 = torch.nn.Conv2d(256, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 32, 32)
        X1 = self.conv0(X1)
        X1 = self.relu(X1)
        X1 = self.conv1(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn1(X1)
        X1 = self.conv2(X1)
        return X1


assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

batch_size = 1
input_features = 3072

X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_2(input_features).to(cuda_device)

# End Call Model
model.eval()
new_out = model(X)
# print(new_out.size())