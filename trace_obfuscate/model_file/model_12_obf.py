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

        params = [3, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[0] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[0] / 4) * 4)
            if self.widen_list[1] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[1] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[1] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[1])
                params[3] = params[3] + 2* int(self.kerneladd_list[1])
                params[6] = params[6] + int(self.kerneladd_list[1])
                params[7] = params[7] + int(self.kerneladd_list[1])
        if self.decompo_list != None:
            if self.decompo_list[1] == 1:
                self.conv1_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv1_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[1] == 2:
                self.conv1_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv1_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv1_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv1_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv1 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv1 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[1] == 1:
                self.conv1_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[1] == 1:
                self.conv1_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn1 = torch.nn.BatchNorm2d(params[1])


        #conv1

        params = [64, 64, 5, 5, 1, 1, 2, 2]
        if self.widen_list != None:
            if self.widen_list[1] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[1] / 4) * 4)
            if self.widen_list[2] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[2] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[2] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[2])
                params[3] = params[3] + 2* int(self.kerneladd_list[2])
                params[6] = params[6] + int(self.kerneladd_list[2])
                params[7] = params[7] + int(self.kerneladd_list[2])
        if self.decompo_list != None:
            if self.decompo_list[2] == 1:
                self.conv2_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv2_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[2] == 2:
                self.conv2_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv2_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv2_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv2_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[2] == 3:
                self.conv2_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv2_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[2] == 4:
                self.conv2_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv2_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv2_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv2_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv2 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv2 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[2] == 1:
                self.conv2_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[2] == 1:
                self.conv2_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn2 = torch.nn.BatchNorm2d(params[1])


        #conv2

        params = [64, 64, 5, 5, 1, 1, 2, 2]
        if self.widen_list != None:
            if self.widen_list[2] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[2] / 4) * 4)
            if self.widen_list[3] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[3] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[3] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[3])
                params[3] = params[3] + 2* int(self.kerneladd_list[3])
                params[6] = params[6] + int(self.kerneladd_list[3])
                params[7] = params[7] + int(self.kerneladd_list[3])
        if self.decompo_list != None:
            if self.decompo_list[3] == 1:
                self.conv3_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv3_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[3] == 2:
                self.conv3_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv3_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv3_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv3_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[3] == 3:
                self.conv3_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv3_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[3] == 4:
                self.conv3_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv3_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv3_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv3_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv3 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv3 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[3] == 1:
                self.conv3_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[3] == 1:
                self.conv3_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn3 = torch.nn.BatchNorm2d(params[1])


        #first merge

        params = [128, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[3] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[3] / 4) * 4)
            if self.widen_list[4] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[4] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[4] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[4])
                params[3] = params[3] + 2* int(self.kerneladd_list[4])
                params[6] = params[6] + int(self.kerneladd_list[4])
                params[7] = params[7] + int(self.kerneladd_list[4])
        if self.decompo_list != None:
            if self.decompo_list[4] == 1:
                self.conv4_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv4_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[4] == 2:
                self.conv4_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv4_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv4_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv4_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[4] == 3:
                self.conv4_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv4_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[4] == 4:
                self.conv4_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv4_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv4_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv4_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv4 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv4 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[4] == 1:
                self.conv4_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[4] == 1:
                self.conv4_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn4 = torch.nn.BatchNorm2d(params[1])


        #sep2

        params = [64, 64, 5, 5, 1, 1, 2, 2]
        params.append(64)
        if self.widen_list != None:
            if self.widen_list[4] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[4] / 4) * 4)
            if self.widen_list[5] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[5] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[5] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[5])
                params[3] = params[3] + 2* int(self.kerneladd_list[5])
                params[6] = params[6] + int(self.kerneladd_list[5])
                params[7] = params[7] + int(self.kerneladd_list[5])
        if self.decompo_list != None:
            self.decompo_list[5] = 0
            if self.decompo_list[5] == 1:
                self.conv5_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv5_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[5] == 2:
                self.conv5_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv5_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv5_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv5_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[5] == 3:
                self.conv5_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv5_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[5] == 4:
                self.conv5_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv5_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv5_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv5_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv5 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv5 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[5] == 1:
                self.conv5_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[5] == 1:
                self.conv5_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [64, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[5] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[5] / 4) * 4)
            if self.widen_list[6] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[6] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[6] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[6])
                params[3] = params[3] + 2* int(self.kerneladd_list[6])
                params[6] = params[6] + int(self.kerneladd_list[6])
                params[7] = params[7] + int(self.kerneladd_list[6])
        if self.decompo_list != None:
            if self.decompo_list[6] == 1:
                self.conv6_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv6_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[6] == 2:
                self.conv6_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv6_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv6_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv6_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[6] == 3:
                self.conv6_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv6_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[6] == 4:
                self.conv6_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv6_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv6_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv6_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv6 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv6 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[6] == 1:
                self.conv6_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[6] == 1:
                self.conv6_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn6 = torch.nn.BatchNorm2d(params[1])


        #second merge

        params = [128, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[6] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[6] / 4) * 4)
            if self.widen_list[7] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[7] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[7] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[7])
                params[3] = params[3] + 2* int(self.kerneladd_list[7])
                params[6] = params[6] + int(self.kerneladd_list[7])
                params[7] = params[7] + int(self.kerneladd_list[7])
        if self.decompo_list != None:
            if self.decompo_list[7] == 1:
                self.conv7_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv7_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[7] == 2:
                self.conv7_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv7_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv7_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv7_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[7] == 3:
                self.conv7_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv7_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[7] == 4:
                self.conv7_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv7_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv7_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv7_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv7 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv7 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[7] == 1:
                self.conv7_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[7] == 1:
                self.conv7_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn7 = torch.nn.BatchNorm2d(params[1])

        #sep3

        params = [64, 64, 3, 3, 1, 1, 1, 1]
        params.append(64)
        if self.widen_list != None:
            if self.widen_list[7] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[7] / 4) * 4)
            if self.widen_list[8] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[8] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[8] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[8])
                params[3] = params[3] + 2* int(self.kerneladd_list[8])
                params[6] = params[6] + int(self.kerneladd_list[8])
                params[7] = params[7] + int(self.kerneladd_list[8])
        if self.decompo_list != None:
            self.decompo_list[8] = 0
            if self.decompo_list[8] == 1:
                self.conv8_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv8_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[8] == 2:
                self.conv8_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv8_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv8_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv8_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[8] == 3:
                self.conv8_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv8_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[8] == 4:
                self.conv8_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv8_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv8_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv8_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv8 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv8 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[8] == 1:
                self.conv8_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[8] == 1:
                self.conv8_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [64, 128, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[8] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[8] / 4) * 4)
            if self.widen_list[9] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[9] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[9] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[9])
                params[3] = params[3] + 2* int(self.kerneladd_list[9])
                params[6] = params[6] + int(self.kerneladd_list[9])
                params[7] = params[7] + int(self.kerneladd_list[9])
        if self.decompo_list != None:
            if self.decompo_list[9] == 1:
                self.conv9_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv9_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[9] == 2:
                self.conv9_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv9_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv9_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv9_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[9] == 3:
                self.conv9_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv9_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[9] == 4:
                self.conv9_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv9_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv9_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv9_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv9 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv9 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[9] == 1:
                self.conv9_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[9] == 1:
                self.conv9_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn9 = torch.nn.BatchNorm2d(params[1])


        #third merge

        params = [192, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[9] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[9] / 4) * 4)
            if self.widen_list[10] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[10] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[10] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[10])
                params[3] = params[3] + 2* int(self.kerneladd_list[10])
                params[6] = params[6] + int(self.kerneladd_list[10])
                params[7] = params[7] + int(self.kerneladd_list[10])
        if self.decompo_list != None:
            if self.decompo_list[10] == 1:
                self.conv10_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv10_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[10] == 2:
                self.conv10_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv10_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv10_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv10_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[10] == 3:
                self.conv10_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv10_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[10] == 4:
                self.conv10_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv10_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv10_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv10_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv10 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv10 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[10] == 1:
                self.conv10_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[10] == 1:
                self.conv10_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn10 = torch.nn.BatchNorm2d(params[1])

        #conv3

        params = [64, 128, 5, 5, 1, 1, 2, 2]
        if self.widen_list != None:
            if self.widen_list[10] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[10] / 4) * 4)
            if self.widen_list[11] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[11] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[11] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[11])
                params[3] = params[3] + 2* int(self.kerneladd_list[11])
                params[6] = params[6] + int(self.kerneladd_list[11])
                params[7] = params[7] + int(self.kerneladd_list[11])
        if self.decompo_list != None:
            if self.decompo_list[11] == 1:
                self.conv11_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv11_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[11] == 2:
                self.conv11_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv11_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv11_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv11_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[11] == 3:
                self.conv11_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv11_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[11] == 4:
                self.conv11_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv11_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv11_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv11_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv11 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv11 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[11] == 1:
                self.conv11_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[11] == 1:
                self.conv11_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn11 = torch.nn.BatchNorm2d(params[1])


        #fourth merge

        params = [384, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[11] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[11] / 4) * 4)
            if self.widen_list[12] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[12] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[12] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[12])
                params[3] = params[3] + 2* int(self.kerneladd_list[12])
                params[6] = params[6] + int(self.kerneladd_list[12])
                params[7] = params[7] + int(self.kerneladd_list[12])
        if self.decompo_list != None:
            if self.decompo_list[12] == 1:
                self.conv12_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv12_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[12] == 2:
                self.conv12_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv12_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv12_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv12_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[12] == 3:
                self.conv12_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv12_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[12] == 4:
                self.conv12_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv12_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv12_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv12_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv12 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv12 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[12] == 1:
                self.conv12_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[12] == 1:
                self.conv12_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn12 = torch.nn.BatchNorm2d(params[1])


        #sep4

        params = [64, 64, 3, 3, 1, 1, 1, 1]
        params.append(64)
        if self.widen_list != None:
            if self.widen_list[12] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[12] / 4) * 4)
            if self.widen_list[13] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[13] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[13] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[13])
                params[3] = params[3] + 2* int(self.kerneladd_list[13])
                params[6] = params[6] + int(self.kerneladd_list[13])
                params[7] = params[7] + int(self.kerneladd_list[13])
        if self.decompo_list != None:
            self.decompo_list[13] = 0
            if self.decompo_list[13] == 1:
                self.conv13_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv13_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[13] == 2:
                self.conv13_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv13_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv13_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv13_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[13] == 3:
                self.conv13_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv13_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[13] == 4:
                self.conv13_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv13_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv13_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv13_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv13 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv13 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[13] == 1:
                self.conv13_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[13] == 1:
                self.conv13_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [64, 128, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[13] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[13] / 4) * 4)
            if self.widen_list[14] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[14] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[14] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[14])
                params[3] = params[3] + 2* int(self.kerneladd_list[14])
                params[6] = params[6] + int(self.kerneladd_list[14])
                params[7] = params[7] + int(self.kerneladd_list[14])
        if self.decompo_list != None:
            if self.decompo_list[14] == 1:
                self.conv14_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv14_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[14] == 2:
                self.conv14_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv14_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv14_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv14_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[14] == 3:
                self.conv14_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv14_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[14] == 4:
                self.conv14_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv14_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv14_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv14_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv14 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv14 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[14] == 1:
                self.conv14_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[14] == 1:
                self.conv14_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn14 = torch.nn.BatchNorm2d(params[1])


        #fifth merge (concat6)

        params = [576, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[14] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[14] / 4) * 4)
            if self.widen_list[15] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[15] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[15] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[15])
                params[3] = params[3] + 2* int(self.kerneladd_list[15])
                params[6] = params[6] + int(self.kerneladd_list[15])
                params[7] = params[7] + int(self.kerneladd_list[15])
        if self.decompo_list != None:
            if self.decompo_list[15] == 1:
                self.conv15_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv15_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[15] == 2:
                self.conv15_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv15_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv15_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv15_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[15] == 3:
                self.conv15_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv15_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[15] == 4:
                self.conv15_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv15_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv15_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv15_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv15 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv15 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[15] == 1:
                self.conv15_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[15] == 1:
                self.conv15_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn15 = torch.nn.BatchNorm2d(params[1])


        #sep5

        params = [64, 64, 5, 5, 1, 1, 2, 2]
        params.append(64)
        if self.widen_list != None:
            if self.widen_list[15] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[15] / 4) * 4)
            if self.widen_list[16] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[16] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[16] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[16])
                params[3] = params[3] + 2* int(self.kerneladd_list[16])
                params[6] = params[6] + int(self.kerneladd_list[16])
                params[7] = params[7] + int(self.kerneladd_list[16])
        if self.decompo_list != None:
            self.decompo_list[16] = 0
            if self.decompo_list[16] == 1:
                self.conv16_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv16_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[16] == 2:
                self.conv16_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv16_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv16_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv16_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[16] == 3:
                self.conv16_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv16_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[16] == 4:
                self.conv16_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv16_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv16_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv16_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv16 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv16 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[16] == 1:
                self.conv16_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[16] == 1:
                self.conv16_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [64, 128, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[16] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[16] / 4) * 4)
            if self.widen_list[17] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[17] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[17] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[17])
                params[3] = params[3] + 2* int(self.kerneladd_list[17])
                params[6] = params[6] + int(self.kerneladd_list[17])
                params[7] = params[7] + int(self.kerneladd_list[17])
        if self.decompo_list != None:
            if self.decompo_list[17] == 1:
                self.conv17_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv17_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[17] == 2:
                self.conv17_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv17_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv17_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv17_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[17] == 3:
                self.conv17_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv17_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[17] == 4:
                self.conv17_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv17_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv17_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv17_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv17 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv17 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[17] == 1:
                self.conv17_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[17] == 1:
                self.conv17_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn17 = torch.nn.BatchNorm2d(params[1])


        #sixth merge (concat7)

        params = [512, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[17] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[17] / 4) * 4)
            if self.widen_list[18] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[18] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[18] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[18])
                params[3] = params[3] + 2* int(self.kerneladd_list[18])
                params[6] = params[6] + int(self.kerneladd_list[18])
                params[7] = params[7] + int(self.kerneladd_list[18])
        if self.decompo_list != None:
            if self.decompo_list[18] == 1:
                self.conv18_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv18_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[18] == 2:
                self.conv18_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv18_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv18_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv18_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[18] == 3:
                self.conv18_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv18_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[18] == 4:
                self.conv18_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv18_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv18_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv18_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv18 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv18 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[18] == 1:
                self.conv18_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[18] == 1:
                self.conv18_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn18 = torch.nn.BatchNorm2d(params[1])


        #seventh merge (concat8)

        params = [192, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[18] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[18] / 4) * 4)
            if self.widen_list[19] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[19] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[19] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[19])
                params[3] = params[3] + 2* int(self.kerneladd_list[19])
                params[6] = params[6] + int(self.kerneladd_list[19])
                params[7] = params[7] + int(self.kerneladd_list[19])
        if self.decompo_list != None:
            if self.decompo_list[19] == 1:
                self.conv19_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv19_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[19] == 2:
                self.conv19_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv19_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv19_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv19_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[19] == 3:
                self.conv19_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv19_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[19] == 4:
                self.conv19_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv19_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv19_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv19_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv19 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv19 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[19] == 1:
                self.conv19_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[19] == 1:
                self.conv19_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn19 = torch.nn.BatchNorm2d(params[1])


        #conv4

        params = [64, 256, 5, 5, 1, 1, 2, 2]
        if self.widen_list != None:
            if self.widen_list[19] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[19] / 4) * 4)
            if self.widen_list[20] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[20] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[20] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[20])
                params[3] = params[3] + 2* int(self.kerneladd_list[20])
                params[6] = params[6] + int(self.kerneladd_list[20])
                params[7] = params[7] + int(self.kerneladd_list[20])
        if self.decompo_list != None:
            if self.decompo_list[20] == 1:
                self.conv20_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv20_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[20] == 2:
                self.conv20_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv20_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv20_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv20_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[20] == 3:
                self.conv20_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv20_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[20] == 4:
                self.conv20_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv20_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv20_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv20_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv20 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv20 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[20] == 1:
                self.conv20_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[20] == 1:
                self.conv20_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn20 = torch.nn.BatchNorm2d(params[1])


        #eighth merge (concat9)

        params = [512, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[20] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[20] / 4) * 4)
            if self.widen_list[21] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[21] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[21] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[21])
                params[3] = params[3] + 2* int(self.kerneladd_list[21])
                params[6] = params[6] + int(self.kerneladd_list[21])
                params[7] = params[7] + int(self.kerneladd_list[21])
        if self.decompo_list != None:
            if self.decompo_list[21] == 1:
                self.conv21_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv21_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[21] == 2:
                self.conv21_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv21_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv21_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv21_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[21] == 3:
                self.conv21_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv21_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[21] == 4:
                self.conv21_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv21_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv21_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv21_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv21 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv21 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[21] == 1:
                self.conv21_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[21] == 1:
                self.conv21_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn21 = torch.nn.BatchNorm2d(params[1])


        #sep6

        params = [64, 64, 5, 5, 1, 1, 2, 2]
        params.append(64)
        if self.widen_list != None:
            if self.widen_list[21] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[21] / 4) * 4)
            if self.widen_list[22] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[22] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[22] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[22])
                params[3] = params[3] + 2* int(self.kerneladd_list[22])
                params[6] = params[6] + int(self.kerneladd_list[22])
                params[7] = params[7] + int(self.kerneladd_list[22])
        if self.decompo_list != None:
            self.decompo_list[22] = 0
            if self.decompo_list[22] == 1:
                self.conv22_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv22_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[22] == 2:
                self.conv22_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv22_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv22_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv22_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[22] == 3:
                self.conv22_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv22_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[22] == 4:
                self.conv22_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv22_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv22_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv22_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv22 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv22 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[22] == 1:
                self.conv22_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[22] == 1:
                self.conv22_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [64, 256, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[22] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[22] / 4) * 4)
            if self.widen_list[23] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[23] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[23] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[23])
                params[3] = params[3] + 2* int(self.kerneladd_list[23])
                params[6] = params[6] + int(self.kerneladd_list[23])
                params[7] = params[7] + int(self.kerneladd_list[23])
        if self.decompo_list != None:
            if self.decompo_list[23] == 1:
                self.conv23_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv23_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[23] == 2:
                self.conv23_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv23_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv23_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv23_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[23] == 3:
                self.conv23_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv23_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[23] == 4:
                self.conv23_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv23_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv23_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv23_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv23 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv23 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[23] == 1:
                self.conv23_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[23] == 1:
                self.conv23_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn23 = torch.nn.BatchNorm2d(params[1])


        #ninth merge (concat10)

        params = [768, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[23] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[23] / 4) * 4)
            if self.widen_list[24] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[24] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[24] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[24])
                params[3] = params[3] + 2* int(self.kerneladd_list[24])
                params[6] = params[6] + int(self.kerneladd_list[24])
                params[7] = params[7] + int(self.kerneladd_list[24])
        if self.decompo_list != None:
            if self.decompo_list[24] == 1:
                self.conv24_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv24_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[24] == 2:
                self.conv24_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv24_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv24_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv24_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[24] == 3:
                self.conv24_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv24_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[24] == 4:
                self.conv24_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv24_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv24_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv24_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv24 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv24 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[24] == 1:
                self.conv24_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[24] == 1:
                self.conv24_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn24 = torch.nn.BatchNorm2d(params[1])


        #conv5

        params = [64, 256, 3, 3, 1, 1, 1, 1]
        if self.widen_list != None:
            if self.widen_list[24] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[24] / 4) * 4)
            if self.widen_list[25] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[25] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[25] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[25])
                params[3] = params[3] + 2* int(self.kerneladd_list[25])
                params[6] = params[6] + int(self.kerneladd_list[25])
                params[7] = params[7] + int(self.kerneladd_list[25])
        if self.decompo_list != None:
            if self.decompo_list[25] == 1:
                self.conv25_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv25_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[25] == 2:
                self.conv25_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv25_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv25_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv25_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[25] == 3:
                self.conv25_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv25_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[25] == 4:
                self.conv25_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv25_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv25_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv25_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv25 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv25 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[25] == 1:
                self.conv25_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[25] == 1:
                self.conv25_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn25 = torch.nn.BatchNorm2d(params[1])


        #tenth merge (concat11)

        params = [960, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[25] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[25] / 4) * 4)
            if self.widen_list[26] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[26] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[26] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[26])
                params[3] = params[3] + 2* int(self.kerneladd_list[26])
                params[6] = params[6] + int(self.kerneladd_list[26])
                params[7] = params[7] + int(self.kerneladd_list[26])
        if self.decompo_list != None:
            if self.decompo_list[26] == 1:
                self.conv26_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv26_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[26] == 2:
                self.conv26_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv26_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv26_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv26_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[26] == 3:
                self.conv26_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv26_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[26] == 4:
                self.conv26_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv26_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv26_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv26_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv26 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv26 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[26] == 1:
                self.conv26_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[26] == 1:
                self.conv26_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn26 = torch.nn.BatchNorm2d(params[1])


        #sep7

        params = [64, 64, 5, 5, 1, 1, 2, 2]
        params.append(64)
        if self.widen_list != None:
            if self.widen_list[26] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[26] / 4) * 4)
            if self.widen_list[27] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[27] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[27] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[27])
                params[3] = params[3] + 2* int(self.kerneladd_list[27])
                params[6] = params[6] + int(self.kerneladd_list[27])
                params[7] = params[7] + int(self.kerneladd_list[27])
        if self.decompo_list != None:
            self.decompo_list[27] = 0
            if self.decompo_list[27] == 1:
                self.conv27_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv27_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[27] == 2:
                self.conv27_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv27_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv27_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv27_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[27] == 3:
                self.conv27_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv27_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[27] == 4:
                self.conv27_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv27_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv27_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv27_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv27 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv27 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[27] == 1:
                self.conv27_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[27] == 1:
                self.conv27_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [64, 256, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[27] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[27] / 4) * 4)
            if self.widen_list[28] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[28] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[28] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[28])
                params[3] = params[3] + 2* int(self.kerneladd_list[28])
                params[6] = params[6] + int(self.kerneladd_list[28])
                params[7] = params[7] + int(self.kerneladd_list[28])
        if self.decompo_list != None:
            if self.decompo_list[28] == 1:
                self.conv28_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv28_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[28] == 2:
                self.conv28_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv28_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv28_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv28_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[28] == 3:
                self.conv28_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv28_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[28] == 4:
                self.conv28_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv28_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv28_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv28_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv28 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv28 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[28] == 1:
                self.conv28_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[28] == 1:
                self.conv28_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn28 = torch.nn.BatchNorm2d(params[1])


        #tenth merge (concat11)

        params = [704, 64, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[28] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[28] / 4) * 4)
            if self.widen_list[29] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[29] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[29] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[29])
                params[3] = params[3] + 2* int(self.kerneladd_list[29])
                params[6] = params[6] + int(self.kerneladd_list[29])
                params[7] = params[7] + int(self.kerneladd_list[29])
        if self.decompo_list != None:
            if self.decompo_list[29] == 1:
                self.conv29_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv29_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[29] == 2:
                self.conv29_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv29_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv29_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv29_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[29] == 3:
                self.conv29_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv29_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[29] == 4:
                self.conv29_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv29_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv29_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv29_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv29 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv29 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[29] == 1:
                self.conv29_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[29] == 1:
                self.conv29_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn29 = torch.nn.BatchNorm2d(params[1])


        #classifier

        params = [64, 10, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[29] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[29] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[30] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[30])
                params[3] = params[3] + 2* int(self.kerneladd_list[30])
                params[6] = params[6] + int(self.kerneladd_list[30])
                params[7] = params[7] + int(self.kerneladd_list[30])
        if self.decompo_list != None:
            if self.decompo_list[30] == 1:
                self.conv30_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv30_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[30] == 3:
                self.conv30_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv30_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[30] == 4:
                self.conv30_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv30_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv30_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv30_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv30 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv30 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[30] == 1:
                self.conv30_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[30] == 1:
                self.conv30_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 32, 32)
        X1 = self.conv0(X1)

        if self.dummy_list != None:
            if self.dummy_list[0] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[0]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[1] == 1:
                X1_0 = self.conv1_0(X1)
                X1_1 = self.conv1_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[1] == 2:
                X1_0 = self.conv1_0(X1)
                X1_1 = self.conv1_1(X1)
                X1_2 = self.conv1_2(X1)
                X1_3 = self.conv1_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            else:
                X1 = self.conv1(X1)
        else:
            X1 = self.conv1(X1)

        if self.dummy_list != None:
            if self.dummy_list[1] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[1]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[1] == 1:
                X1 = self.conv1_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[1] == 1:
                X1_skip = X1
                X1 = self.conv1_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn1(X1)
        sep1 = X1 #64x32x32

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[2] == 1:
                X1_0 = self.conv2_0(X1)
                X1_1 = self.conv2_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[2] == 2:
                X1_0 = self.conv2_0(X1)
                X1_1 = self.conv2_1(X1)
                X1_2 = self.conv2_2(X1)
                X1_3 = self.conv2_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[2] == 3:
                X1_0 = self.conv2_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv2_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[2] == 4:
                X1_0 = self.conv2_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv2_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv2_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv2_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv2(X1)
        else:
            X1 = self.conv2(X1)

        if self.dummy_list != None:
            if self.dummy_list[2] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[2]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[2] == 1:
                X1 = self.conv2_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[2] == 1:
                X1_skip = X1
                X1 = self.conv2_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn2(X1)
        C1 = X1 #64x32x32

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[3] == 1:
                X1_0 = self.conv3_0(X1)
                X1_1 = self.conv3_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[3] == 2:
                X1_0 = self.conv3_0(X1)
                X1_1 = self.conv3_1(X1)
                X1_2 = self.conv3_2(X1)
                X1_3 = self.conv3_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[3] == 3:
                X1_0 = self.conv3_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv3_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[3] == 4:
                X1_0 = self.conv3_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv3_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv3_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv3_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv3(X1)
        else:
            X1 = self.conv3(X1)

        if self.dummy_list != None:
            if self.dummy_list[3] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[3]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[3] == 1:
                X1 = self.conv3_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[3] == 1:
                X1_skip = X1
                X1 = self.conv3_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn3(X1)
        C2 = X1 #64x32x32
        X1 = torch.cat([sep1, C2], 1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[4] == 1:
                X1_0 = self.conv4_0(X1)
                X1_1 = self.conv4_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[4] == 2:
                X1_0 = self.conv4_0(X1)
                X1_1 = self.conv4_1(X1)
                X1_2 = self.conv4_2(X1)
                X1_3 = self.conv4_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[4] == 3:
                X1_0 = self.conv4_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv4_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[4] == 4:
                X1_0 = self.conv4_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv4_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv4_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv4_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv4(X1)
        else:
            X1 = self.conv4(X1)

        if self.dummy_list != None:
            if self.dummy_list[4] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[4]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[4] == 1:
                X1 = self.conv4_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[4] == 1:
                X1_skip = X1
                X1 = self.conv4_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn4(X1)
        

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[5] == 1:
                X1_0 = self.conv5_0(X1)
                X1_1 = self.conv5_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[5] == 2:
                X1_0 = self.conv5_0(X1)
                X1_1 = self.conv5_1(X1)
                X1_2 = self.conv5_2(X1)
                X1_3 = self.conv5_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[5] == 3:
                X1_0 = self.conv5_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv5_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[5] == 4:
                X1_0 = self.conv5_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv5_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv5_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv5_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv5(X1)
        else:
            X1 = self.conv5(X1)

        if self.dummy_list != None:
            if self.dummy_list[5] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[5]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[6] == 1:
                X1_0 = self.conv6_0(X1)
                X1_1 = self.conv6_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[6] == 2:
                X1_0 = self.conv6_0(X1)
                X1_1 = self.conv6_1(X1)
                X1_2 = self.conv6_2(X1)
                X1_3 = self.conv6_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[6] == 3:
                X1_0 = self.conv6_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv6_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[6] == 4:
                X1_0 = self.conv6_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv6_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv6_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv6_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv6(X1)
        else:
            X1 = self.conv6(X1)

        if self.dummy_list != None:
            if self.dummy_list[6] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[6]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[6] == 1:
                X1 = self.conv6_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[6] == 1:
                X1_skip = X1
                X1 = self.conv6_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn6(X1)
        sep2 = X1 #64x32x32
        X1 = torch.cat([sep1, sep2], 1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[7] == 1:
                X1_0 = self.conv7_0(X1)
                X1_1 = self.conv7_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[7] == 2:
                X1_0 = self.conv7_0(X1)
                X1_1 = self.conv7_1(X1)
                X1_2 = self.conv7_2(X1)
                X1_3 = self.conv7_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[7] == 3:
                X1_0 = self.conv7_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv7_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[7] == 4:
                X1_0 = self.conv7_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv7_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv7_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv7_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv7(X1)
        else:
            X1 = self.conv7(X1)

        if self.dummy_list != None:
            if self.dummy_list[7] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[7]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[7] == 1:
                X1 = self.conv7_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[7] == 1:
                X1_skip = X1
                X1 = self.conv7_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn7(X1)
        X1 = self.maxpool2x2(X1)
        

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[8] == 1:
                X1_0 = self.conv8_0(X1)
                X1_1 = self.conv8_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[8] == 2:
                X1_0 = self.conv8_0(X1)
                X1_1 = self.conv8_1(X1)
                X1_2 = self.conv8_2(X1)
                X1_3 = self.conv8_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[8] == 3:
                X1_0 = self.conv8_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv8_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[8] == 4:
                X1_0 = self.conv8_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv8_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv8_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv8_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv8(X1)
        else:
            X1 = self.conv8(X1)

        if self.dummy_list != None:
            if self.dummy_list[8] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[8]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[9] == 1:
                X1_0 = self.conv9_0(X1)
                X1_1 = self.conv9_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[9] == 2:
                X1_0 = self.conv9_0(X1)
                X1_1 = self.conv9_1(X1)
                X1_2 = self.conv9_2(X1)
                X1_3 = self.conv9_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[9] == 3:
                X1_0 = self.conv9_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv9_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[9] == 4:
                X1_0 = self.conv9_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv9_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv9_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv9_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv9(X1)
        else:
            X1 = self.conv9(X1)

        if self.dummy_list != None:
            if self.dummy_list[9] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[9]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[9] == 1:
                X1 = self.conv9_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[9] == 1:
                X1_skip = X1
                X1 = self.conv9_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn9(X1)
        sep3 = X1 #128x16x16
        X1 = torch.cat([self.avgpool2x2(sep2), sep3], 1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[10] == 1:
                X1_0 = self.conv10_0(X1)
                X1_1 = self.conv10_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[10] == 2:
                X1_0 = self.conv10_0(X1)
                X1_1 = self.conv10_1(X1)
                X1_2 = self.conv10_2(X1)
                X1_3 = self.conv10_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[10] == 3:
                X1_0 = self.conv10_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv10_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[10] == 4:
                X1_0 = self.conv10_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv10_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv10_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv10_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv10(X1)
        else:
            X1 = self.conv10(X1)

        if self.dummy_list != None:
            if self.dummy_list[10] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[10]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[10] == 1:
                X1 = self.conv10_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[10] == 1:
                X1_skip = X1
                X1 = self.conv10_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn10(X1)
        

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[11] == 1:
                X1_0 = self.conv11_0(X1)
                X1_1 = self.conv11_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[11] == 2:
                X1_0 = self.conv11_0(X1)
                X1_1 = self.conv11_1(X1)
                X1_2 = self.conv11_2(X1)
                X1_3 = self.conv11_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[11] == 3:
                X1_0 = self.conv11_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv11_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[11] == 4:
                X1_0 = self.conv11_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv11_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv11_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv11_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv11(X1)
        else:
            X1 = self.conv11(X1)

        if self.dummy_list != None:
            if self.dummy_list[11] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[11]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[11] == 1:
                X1 = self.conv11_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[11] == 1:
                X1_skip = X1
                X1 = self.conv11_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn11(X1)
        C3 = X1 #128x16x16
        X1 = torch.cat([self.avgpool2x2(C2), self.avgpool2x2(sep2), sep3, C3], 1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[12] == 1:
                X1_0 = self.conv12_0(X1)
                X1_1 = self.conv12_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[12] == 2:
                X1_0 = self.conv12_0(X1)
                X1_1 = self.conv12_1(X1)
                X1_2 = self.conv12_2(X1)
                X1_3 = self.conv12_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[12] == 3:
                X1_0 = self.conv12_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv12_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[12] == 4:
                X1_0 = self.conv12_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv12_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv12_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv12_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv12(X1)
        else:
            X1 = self.conv12(X1)

        if self.dummy_list != None:
            if self.dummy_list[12] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[12]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[12] == 1:
                X1 = self.conv12_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[12] == 1:
                X1_skip = X1
                X1 = self.conv12_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn12(X1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[13] == 1:
                X1_0 = self.conv13_0(X1)
                X1_1 = self.conv13_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[13] == 2:
                X1_0 = self.conv13_0(X1)
                X1_1 = self.conv13_1(X1)
                X1_2 = self.conv13_2(X1)
                X1_3 = self.conv13_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[13] == 3:
                X1_0 = self.conv13_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv13_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[13] == 4:
                X1_0 = self.conv13_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv13_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv13_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv13_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv13(X1)
        else:
            X1 = self.conv13(X1)

        if self.dummy_list != None:
            if self.dummy_list[13] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[13]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[14] == 1:
                X1_0 = self.conv14_0(X1)
                X1_1 = self.conv14_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[14] == 2:
                X1_0 = self.conv14_0(X1)
                X1_1 = self.conv14_1(X1)
                X1_2 = self.conv14_2(X1)
                X1_3 = self.conv14_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[14] == 3:
                X1_0 = self.conv14_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv14_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[14] == 4:
                X1_0 = self.conv14_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv14_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv14_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv14_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv14(X1)
        else:
            X1 = self.conv14(X1)

        if self.dummy_list != None:
            if self.dummy_list[14] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[14]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[14] == 1:
                X1 = self.conv14_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[14] == 1:
                X1_skip = X1
                X1 = self.conv14_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn14(X1)
        sep4 = X1 #128x16x16
        X1 = torch.cat([self.avgpool2x2(C2), C3, self.avgpool2x2(sep1), self.avgpool2x2(sep2), sep3, sep4], 1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[15] == 1:
                X1_0 = self.conv15_0(X1)
                X1_1 = self.conv15_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[15] == 2:
                X1_0 = self.conv15_0(X1)
                X1_1 = self.conv15_1(X1)
                X1_2 = self.conv15_2(X1)
                X1_3 = self.conv15_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[15] == 3:
                X1_0 = self.conv15_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv15_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[15] == 4:
                X1_0 = self.conv15_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv15_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv15_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv15_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv15(X1)
        else:
            X1 = self.conv15(X1)

        if self.dummy_list != None:
            if self.dummy_list[15] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[15]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[15] == 1:
                X1 = self.conv15_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[15] == 1:
                X1_skip = X1
                X1 = self.conv15_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn15(X1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[16] == 1:
                X1_0 = self.conv16_0(X1)
                X1_1 = self.conv16_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[16] == 2:
                X1_0 = self.conv16_0(X1)
                X1_1 = self.conv16_1(X1)
                X1_2 = self.conv16_2(X1)
                X1_3 = self.conv16_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[16] == 3:
                X1_0 = self.conv16_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv16_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[16] == 4:
                X1_0 = self.conv16_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv16_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv16_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv16_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv16(X1)
        else:
            X1 = self.conv16(X1)

        if self.dummy_list != None:
            if self.dummy_list[16] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[16]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[17] == 1:
                X1_0 = self.conv17_0(X1)
                X1_1 = self.conv17_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[17] == 2:
                X1_0 = self.conv17_0(X1)
                X1_1 = self.conv17_1(X1)
                X1_2 = self.conv17_2(X1)
                X1_3 = self.conv17_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[17] == 3:
                X1_0 = self.conv17_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv17_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[17] == 4:
                X1_0 = self.conv17_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv17_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv17_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv17_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv17(X1)
        else:
            X1 = self.conv17(X1)

        if self.dummy_list != None:
            if self.dummy_list[17] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[17]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[17] == 1:
                X1 = self.conv17_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[17] == 1:
                X1_skip = X1
                X1 = self.conv17_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn17(X1)
        sep5 = X1 #128x16x16
        X1 = torch.cat([self.avgpool2x2(sep1), self.avgpool2x2(sep2), sep3, sep4, sep5], 1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[18] == 1:
                X1_0 = self.conv18_0(X1)
                X1_1 = self.conv18_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[18] == 2:
                X1_0 = self.conv18_0(X1)
                X1_1 = self.conv18_1(X1)
                X1_2 = self.conv18_2(X1)
                X1_3 = self.conv18_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[18] == 3:
                X1_0 = self.conv18_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv18_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[18] == 4:
                X1_0 = self.conv18_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv18_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv18_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv18_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv18(X1)
        else:
            X1 = self.conv18(X1)

        if self.dummy_list != None:
            if self.dummy_list[18] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[18]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[18] == 1:
                X1 = self.conv18_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[18] == 1:
                X1_skip = X1
                X1 = self.conv18_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn18(X1)
        X1 = self.maxpool2x2(X1)
        pool2 = X1 #64x8x8
        X1 = torch.cat([self.avgpool2x2(C3), pool2], 1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[19] == 1:
                X1_0 = self.conv19_0(X1)
                X1_1 = self.conv19_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[19] == 2:
                X1_0 = self.conv19_0(X1)
                X1_1 = self.conv19_1(X1)
                X1_2 = self.conv19_2(X1)
                X1_3 = self.conv19_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[19] == 3:
                X1_0 = self.conv19_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv19_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[19] == 4:
                X1_0 = self.conv19_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv19_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv19_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv19_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv19(X1)
        else:
            X1 = self.conv19(X1)

        if self.dummy_list != None:
            if self.dummy_list[19] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[19]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[19] == 1:
                X1 = self.conv19_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[19] == 1:
                X1_skip = X1
                X1 = self.conv19_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn19(X1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[20] == 1:
                X1_0 = self.conv20_0(X1)
                X1_1 = self.conv20_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[20] == 2:
                X1_0 = self.conv20_0(X1)
                X1_1 = self.conv20_1(X1)
                X1_2 = self.conv20_2(X1)
                X1_3 = self.conv20_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[20] == 3:
                X1_0 = self.conv20_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv20_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[20] == 4:
                X1_0 = self.conv20_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv20_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv20_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv20_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv20(X1)
        else:
            X1 = self.conv20(X1)

        if self.dummy_list != None:
            if self.dummy_list[20] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[20]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[20] == 1:
                X1 = self.conv20_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[20] == 1:
                X1_skip = X1
                X1 = self.conv20_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn20(X1)
        C4 = X1 #256x8x8
        X1 = torch.cat([self.avgpool4x4(sep2), self.avgpool2x2(sep4), self.avgpool4x4(C2), C4], 1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[21] == 1:
                X1_0 = self.conv21_0(X1)
                X1_1 = self.conv21_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[21] == 2:
                X1_0 = self.conv21_0(X1)
                X1_1 = self.conv21_1(X1)
                X1_2 = self.conv21_2(X1)
                X1_3 = self.conv21_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[21] == 3:
                X1_0 = self.conv21_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv21_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[21] == 4:
                X1_0 = self.conv21_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv21_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv21_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv21_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv21(X1)
        else:
            X1 = self.conv21(X1)

        if self.dummy_list != None:
            if self.dummy_list[21] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[21]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[21] == 1:
                X1 = self.conv21_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[21] == 1:
                X1_skip = X1
                X1 = self.conv21_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn21(X1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[22] == 1:
                X1_0 = self.conv22_0(X1)
                X1_1 = self.conv22_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[22] == 2:
                X1_0 = self.conv22_0(X1)
                X1_1 = self.conv22_1(X1)
                X1_2 = self.conv22_2(X1)
                X1_3 = self.conv22_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[22] == 3:
                X1_0 = self.conv22_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv22_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[22] == 4:
                X1_0 = self.conv22_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv22_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv22_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv22_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv22(X1)
        else:
            X1 = self.conv22(X1)

        if self.dummy_list != None:
            if self.dummy_list[22] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[22]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[23] == 1:
                X1_0 = self.conv23_0(X1)
                X1_1 = self.conv23_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[23] == 2:
                X1_0 = self.conv23_0(X1)
                X1_1 = self.conv23_1(X1)
                X1_2 = self.conv23_2(X1)
                X1_3 = self.conv23_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[23] == 3:
                X1_0 = self.conv23_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv23_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[23] == 4:
                X1_0 = self.conv23_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv23_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv23_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv23_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv23(X1)
        else:
            X1 = self.conv23(X1)

        if self.dummy_list != None:
            if self.dummy_list[23] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[23]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[23] == 1:
                X1 = self.conv23_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[23] == 1:
                X1_skip = X1
                X1 = self.conv23_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn23(X1)
        sep6 = X1 #256x8x8
        X1 = torch.cat([self.avgpool2x2(sep3), self.avgpool4x4(C1), self.avgpool4x4(C2), C4, sep6], 1) #256 + 256 + 256

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[24] == 1:
                X1_0 = self.conv24_0(X1)
                X1_1 = self.conv24_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[24] == 2:
                X1_0 = self.conv24_0(X1)
                X1_1 = self.conv24_1(X1)
                X1_2 = self.conv24_2(X1)
                X1_3 = self.conv24_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[24] == 3:
                X1_0 = self.conv24_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv24_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[24] == 4:
                X1_0 = self.conv24_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv24_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv24_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv24_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv24(X1)
        else:
            X1 = self.conv24(X1)

        if self.dummy_list != None:
            if self.dummy_list[24] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[24]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[24] == 1:
                X1 = self.conv24_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[24] == 1:
                X1_skip = X1
                X1 = self.conv24_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn24(X1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[25] == 1:
                X1_0 = self.conv25_0(X1)
                X1_1 = self.conv25_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[25] == 2:
                X1_0 = self.conv25_0(X1)
                X1_1 = self.conv25_1(X1)
                X1_2 = self.conv25_2(X1)
                X1_3 = self.conv25_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[25] == 3:
                X1_0 = self.conv25_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv25_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[25] == 4:
                X1_0 = self.conv25_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv25_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv25_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv25_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv25(X1)
        else:
            X1 = self.conv25(X1)

        if self.dummy_list != None:
            if self.dummy_list[25] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[25]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[25] == 1:
                X1 = self.conv25_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[25] == 1:
                X1_skip = X1
                X1 = self.conv25_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn25(X1)
        C5 = X1 #256x8x8
        X1 = torch.cat([self.avgpool4x4(C2), self.avgpool4x4(sep1), self.avgpool4x4(sep2), self.avgpool2x2(sep3), self.avgpool2x2(sep4), sep6, C5], 1) #256 + 256 + 128 + 128 + 192

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[26] == 1:
                X1_0 = self.conv26_0(X1)
                X1_1 = self.conv26_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[26] == 2:
                X1_0 = self.conv26_0(X1)
                X1_1 = self.conv26_1(X1)
                X1_2 = self.conv26_2(X1)
                X1_3 = self.conv26_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[26] == 3:
                X1_0 = self.conv26_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv26_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[26] == 4:
                X1_0 = self.conv26_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv26_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv26_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv26_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv26(X1)
        else:
            X1 = self.conv26(X1)

        if self.dummy_list != None:
            if self.dummy_list[26] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[26]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[26] == 1:
                X1 = self.conv26_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[26] == 1:
                X1_skip = X1
                X1 = self.conv26_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn26(X1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[27] == 1:
                X1_0 = self.conv27_0(X1)
                X1_1 = self.conv27_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[27] == 2:
                X1_0 = self.conv27_0(X1)
                X1_1 = self.conv27_1(X1)
                X1_2 = self.conv27_2(X1)
                X1_3 = self.conv27_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[27] == 3:
                X1_0 = self.conv27_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv27_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[27] == 4:
                X1_0 = self.conv27_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv27_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv27_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv27_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv27(X1)
        else:
            X1 = self.conv27(X1)

        if self.dummy_list != None:
            if self.dummy_list[27] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[27]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[28] == 1:
                X1_0 = self.conv28_0(X1)
                X1_1 = self.conv28_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[28] == 2:
                X1_0 = self.conv28_0(X1)
                X1_1 = self.conv28_1(X1)
                X1_2 = self.conv28_2(X1)
                X1_3 = self.conv28_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[28] == 3:
                X1_0 = self.conv28_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv28_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[28] == 4:
                X1_0 = self.conv28_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv28_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv28_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv28_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv28(X1)
        else:
            X1 = self.conv28(X1)

        if self.dummy_list != None:
            if self.dummy_list[28] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[28]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[28] == 1:
                X1 = self.conv28_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[28] == 1:
                X1_skip = X1
                X1 = self.conv28_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn28(X1)
        sep7 = X1 #256x8x8
        X1 = torch.cat([self.avgpool2x2(sep4), self.avgpool4x4(sep2), sep6, sep7], 1) #256 + 256 + 128 + 64

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[29] == 1:
                X1_0 = self.conv29_0(X1)
                X1_1 = self.conv29_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[29] == 2:
                X1_0 = self.conv29_0(X1)
                X1_1 = self.conv29_1(X1)
                X1_2 = self.conv29_2(X1)
                X1_3 = self.conv29_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[29] == 3:
                X1_0 = self.conv29_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv29_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[29] == 4:
                X1_0 = self.conv29_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv29_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv29_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv29_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv29(X1)
        else:
            X1 = self.conv29(X1)

        if self.dummy_list != None:
            if self.dummy_list[29] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[29]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[29] == 1:
                X1 = self.conv29_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[29] == 1:
                X1_skip = X1
                X1 = self.conv29_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.conv_bn29(X1)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[30] == 1:
                X1_0 = self.conv30_0(X1)
                X1_1 = self.conv30_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[30] == 3:
                X1_0 = self.conv30_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv30_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[30] == 4:
                X1_0 = self.conv30_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv30_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv30_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv30_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv30(X1)
        else:
            X1 = self.conv30(X1)

        if self.dummy_list != None:
            if self.dummy_list[30] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[30]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

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