import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append('../')
from torch_utils import summary
assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

#This model stands for a customized model #model_id = 4 (regardless of the file name)
#This model scripts is the original model_3.py [before the decompo and dummy API are added], vanilla one. Go throught the func_modifier again to add more obfuscators.
# cnn_32p3p1p1p0_128p3p1p1p1_128p3p1p1p0_256p3p1p1p0_256p3p1p1p0_256p3p1p1p0_1024p3p1p1p0_1024p3p1p1p0_1024p3p1p1p0_bn1_mlp_256_128_128_bn1
class custom_cnn_11(torch.nn.Module):
    def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):
        super(custom_cnn_11,self).__init__()
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
        self.avgpool4x4 = torch.nn.AvgPool2d(kernel_size=4, stride=4)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        #sep1
        self.conv0 = torch.nn.Conv2d(3, 3, (5, 5), stride=(1, 1), padding=(2, 2), groups=3, bias=False)
        self.conv1 = torch.nn.Conv2d(3, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn1 = torch.nn.BatchNorm2d(64)

        #conv1
        self.conv2 = torch.nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.conv_bn2 = torch.nn.BatchNorm2d(64)

        #conv2
        self.conv3 = torch.nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.conv_bn3 = torch.nn.BatchNorm2d(64)

        #first merge
        self.conv4 = torch.nn.Conv2d(128, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn4 = torch.nn.BatchNorm2d(64)

        #sep2
        self.conv5 = torch.nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=False)
        self.conv6 = torch.nn.Conv2d(64, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn6 = torch.nn.BatchNorm2d(64)

        #second merge
        self.conv7 = torch.nn.Conv2d(128, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn7 = torch.nn.BatchNorm2d(64)
        #sep3
        self.conv8 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.conv9 = torch.nn.Conv2d(64, 128, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn9 = torch.nn.BatchNorm2d(128)

        #third merge
        self.conv10 = torch.nn.Conv2d(192, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn10 = torch.nn.BatchNorm2d(64)
        #conv3
        self.conv11 = torch.nn.Conv2d(64, 128, (5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.conv_bn11 = torch.nn.BatchNorm2d(128)

        #fourth merge
        self.conv12 = torch.nn.Conv2d(384, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn12 = torch.nn.BatchNorm2d(64)

        #sep4
        self.conv13 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
        self.conv14 = torch.nn.Conv2d(64, 128, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn14 = torch.nn.BatchNorm2d(128)

        #fifth merge (concat6)
        self.conv15 = torch.nn.Conv2d(576, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn15 = torch.nn.BatchNorm2d(64)

        #sep5
        self.conv16 = torch.nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=False)
        self.conv17 = torch.nn.Conv2d(64, 128, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn17 = torch.nn.BatchNorm2d(128)

        #sixth merge (concat7)
        self.conv18 = torch.nn.Conv2d(512, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn18 = torch.nn.BatchNorm2d(64)

        #seventh merge (concat8)
        self.conv19 = torch.nn.Conv2d(192, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn19 = torch.nn.BatchNorm2d(64)

        #conv4
        self.conv20 = torch.nn.Conv2d(64, 256, (5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.conv_bn20 = torch.nn.BatchNorm2d(256)

        #eighth merge (concat9)
        self.conv21 = torch.nn.Conv2d(512, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn21 = torch.nn.BatchNorm2d(64)

        #sep6
        self.conv22 = torch.nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=False)
        self.conv23 = torch.nn.Conv2d(64, 256, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn23 = torch.nn.BatchNorm2d(256)

        #ninth merge (concat10)
        self.conv24 = torch.nn.Conv2d(768, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn24 = torch.nn.BatchNorm2d(64)

        #conv5
        self.conv25 = torch.nn.Conv2d(64, 256, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_bn25 = torch.nn.BatchNorm2d(256)

        #tenth merge (concat11)
        self.conv26 = torch.nn.Conv2d(960, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn26 = torch.nn.BatchNorm2d(64)

        #sep7
        self.conv27 = torch.nn.Conv2d(64, 64, (5, 5), stride=(1, 1), padding=(2, 2), groups=64, bias=False)
        self.conv28 = torch.nn.Conv2d(64, 256, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn28 = torch.nn.BatchNorm2d(256)

        #tenth merge (concat11)
        self.conv29 = torch.nn.Conv2d(704, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn29 = torch.nn.BatchNorm2d(64)

        #classifier
        self.conv30 = torch.nn.Conv2d(64, 10, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)

    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 32, 32)
        X1 = self.conv0(X1)
        X1 = self.conv1(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn1(X1)
        sep1 = X1 #64x32x32
        X1 = self.conv2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn2(X1)
        C1 = X1 #64x32x32
        X1 = self.conv3(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn3(X1)
        C2 = X1 #64x32x32
        X1 = torch.cat([sep1, C2], 1)
        X1 = self.conv4(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn4(X1)
        
        X1 = self.conv5(X1)
        X1 = self.conv6(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn6(X1)
        sep2 = X1 #64x32x32
        X1 = torch.cat([sep1, sep2], 1)
        X1 = self.conv7(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn7(X1)
        X1 = self.maxpool2x2(X1)
        
        X1 = self.conv8(X1)
        X1 = self.conv9(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn9(X1)
        sep3 = X1 #128x16x16
        X1 = torch.cat([self.avgpool2x2(sep2), sep3], 1)
        X1 = self.conv10(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn10(X1)
        
        X1 = self.conv11(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn11(X1)
        C3 = X1 #128x16x16
        X1 = torch.cat([self.avgpool2x2(C2), self.avgpool2x2(sep2), sep3, C3], 1)
        X1 = self.conv12(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn12(X1)
        X1 = self.conv13(X1)
        X1 = self.conv14(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn14(X1)
        sep4 = X1 #128x16x16
        X1 = torch.cat([self.avgpool2x2(C2), C3, self.avgpool2x2(sep1), self.avgpool2x2(sep2), sep3, sep4], 1)
        X1 = self.conv15(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn15(X1)
        X1 = self.conv16(X1)
        X1 = self.conv17(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn17(X1)
        sep5 = X1 #128x16x16
        X1 = torch.cat([self.avgpool2x2(sep1), self.avgpool2x2(sep2), sep3, sep4, sep5], 1)
        X1 = self.conv18(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn18(X1)
        X1 = self.maxpool2x2(X1)
        pool2 = X1 #64x8x8
        X1 = torch.cat([self.avgpool2x2(C3), pool2], 1)
        X1 = self.conv19(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn19(X1)
        X1 = self.conv20(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn20(X1)
        C4 = X1 #256x8x8
        X1 = torch.cat([self.avgpool4x4(sep2), self.avgpool2x2(sep4), self.avgpool4x4(C2), C4], 1)
        X1 = self.conv21(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn21(X1)
        X1 = self.conv22(X1)
        X1 = self.conv23(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn23(X1)
        sep6 = X1 #256x8x8
        X1 = torch.cat([self.avgpool2x2(sep3), self.avgpool4x4(C1), self.avgpool4x4(C2), C4, sep6], 1) #256 + 256 + 256
        X1 = self.conv24(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn24(X1)
        X1 = self.conv25(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn25(X1)
        C5 = X1 #256x8x8
        X1 = torch.cat([self.avgpool4x4(C2), self.avgpool4x4(sep1), self.avgpool4x4(sep2), self.avgpool2x2(sep3), self.avgpool2x2(sep4), sep6, C5], 1) #256 + 256 + 128 + 128 + 192
        X1 = self.conv26(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn26(X1)
        X1 = self.conv27(X1)
        X1 = self.conv28(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn28(X1)
        sep7 = X1 #256x8x8
        X1 = torch.cat([self.avgpool2x2(sep4), self.avgpool4x4(sep2), sep6, sep7], 1) #256 + 256 + 128 + 64
        X1 = self.conv29(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn29(X1)
        X1 = self.conv30(X1)
        X1 = self.avgpool(X1)
        X1 = X1.view(X1.shape[:2])
        X1 = self.logsoftmax(X1)
        return X1

batch_size = 1
input_features = 3072
torch.manual_seed(1234)
X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_11(input_features).to(cuda_device)

# End Call Model
model.eval()
new_out = model(X)
# model = custom_cnn_11(input_features, reshape = False).to(cuda_device)
# print(summary(model, (3,32,32)))