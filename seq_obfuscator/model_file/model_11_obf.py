import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

#This model stands for MobileNet ImageNet
#model_id = 10 (regardless of the file name)

class custom_cnn_10(torch.nn.Module):
    def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):
        super(custom_cnn_10,self).__init__()
        self.reshape = reshape
        self.widen_list = widen_list
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.deepen_list = deepen_list
        self.skipcon_list = skipcon_list
        self.kerneladd_list = kerneladd_list
        self.relu = torch.nn.ReLU6(inplace=True)
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        

        params = [3, 32, 3, 3, 2, 2, 1, 1]
        if self.widen_list != None:
            if self.widen_list[0] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[0] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[0] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[0])
                params[3] = params[3] + 2* int(self.kerneladd_list[0])
                params[6] = params[6] + int(self.kerneladd_list[0])
                params[7] = params[7] + int(self.kerneladd_list[0])
        if self.decompo_list != None:
            if self.decompo_list[0] == 1:
                self.conv0_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv0_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[0] == 2:
                self.conv0_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv0_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv0_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv0_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv0 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv0 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[0] == 1:
                self.conv0_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[0] == 1:
                self.conv0_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn0 = torch.nn.BatchNorm2d(params[1])



        # [1, 16, 1, 1]


        params = [32, 32, 3, 3, 1, 1, 1, 1]
        params.append(32)
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
            self.decompo_list[1] = 0
            if self.decompo_list[1] == 1:
                self.conv1_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv1_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[1] == 2:
                self.conv1_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv1_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv1_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv1_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[1] == 3:
                self.conv1_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv1_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[1] == 4:
                self.conv1_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv1_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv1_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv1_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv1 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv1 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[1] == 1:
                self.conv1_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[1] == 1:
                self.conv1_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn1 = torch.nn.BatchNorm2d(params[1])


        params = [32, 16, 1, 1, 1, 1, 0, 0]
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


        # [6, 24, 2, 2]


        params = [16, 96, 1, 1, 1, 1, 0, 0]
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



        params = [96, 96, 3, 3, 2, 2, 1, 1]
        params.append(96)
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
            self.decompo_list[4] = 0
            if self.decompo_list[4] == 1:
                self.conv4_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv4_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[4] == 2:
                self.conv4_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv4_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv4_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv4_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[4] == 3:
                self.conv4_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv4_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[4] == 4:
                self.conv4_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv4_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv4_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv4_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv4 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv4 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[4] == 1:
                self.conv4_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[4] == 1:
                self.conv4_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn4 = torch.nn.BatchNorm2d(params[1])



        params = [96, 24, 1, 1, 1, 1, 0, 0]
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
            if self.decompo_list[5] == 1:
                self.conv5_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv5_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[5] == 2:
                self.conv5_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv5_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv5_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv5_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[5] == 3:
                self.conv5_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv5_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[5] == 4:
                self.conv5_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv5_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv5_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv5_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv5 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv5 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[5] == 1:
                self.conv5_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[5] == 1:
                self.conv5_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn5 = torch.nn.BatchNorm2d(params[1])



        params = [24, 144, 1, 1, 1, 1, 0, 0]
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



        params = [144, 144, 3, 3, 1, 1, 1, 1]
        params.append(144)
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
            self.decompo_list[7] = 0
            if self.decompo_list[7] == 1:
                self.conv7_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv7_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[7] == 2:
                self.conv7_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv7_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv7_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv7_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[7] == 3:
                self.conv7_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv7_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[7] == 4:
                self.conv7_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv7_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv7_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv7_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv7 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv7 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[7] == 1:
                self.conv7_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[7] == 1:
                self.conv7_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn7 = torch.nn.BatchNorm2d(params[1])



        params = [144, 24, 1, 1, 1, 1, 0, 0]
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
            if self.decompo_list[8] == 1:
                self.conv8_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv8_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[8] == 2:
                self.conv8_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv8_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv8_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv8_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[8] == 3:
                self.conv8_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv8_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[8] == 4:
                self.conv8_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv8_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv8_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv8_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv8 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv8 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[8] == 1:
                self.conv8_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[8] == 1:
                self.conv8_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn8 = torch.nn.BatchNorm2d(params[1])


        # [6, 32, 3, 2],


        params = [24, 144, 1, 1, 1, 1, 0, 0]
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



        params = [144, 144, 3, 3, 2, 2, 1, 1]
        params.append(144)
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
            self.decompo_list[10] = 0
            if self.decompo_list[10] == 1:
                self.conv10_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv10_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[10] == 2:
                self.conv10_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv10_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv10_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv10_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[10] == 3:
                self.conv10_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv10_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[10] == 4:
                self.conv10_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv10_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv10_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv10_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv10 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv10 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[10] == 1:
                self.conv10_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[10] == 1:
                self.conv10_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn10 = torch.nn.BatchNorm2d(params[1])



        params = [144, 32, 1, 1, 1, 1, 0, 0]
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



        params = [32, 192, 1, 1, 1, 1, 0, 0]
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



        params = [192, 192, 3, 3, 1, 1, 1, 1]
        params.append(192)
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


        self.conv_bn13 = torch.nn.BatchNorm2d(params[1])



        params = [192, 32, 1, 1, 1, 1, 0, 0]
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



        params = [32, 192, 1, 1, 1, 1, 0, 0]
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



        params = [192, 192, 3, 3, 1, 1, 1, 1]
        params.append(192)
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


        self.conv_bn16 = torch.nn.BatchNorm2d(params[1])



        params = [192, 32, 1, 1, 1, 1, 0, 0]
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


        # [6, 64, 4, 2],


        params = [32, 192, 1, 1, 1, 1, 0, 0]
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



        params = [192, 192, 3, 3, 2, 2, 1, 1]
        params.append(192)
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
            self.decompo_list[19] = 0
            if self.decompo_list[19] == 1:
                self.conv19_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv19_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[19] == 2:
                self.conv19_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv19_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv19_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv19_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[19] == 3:
                self.conv19_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv19_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[19] == 4:
                self.conv19_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv19_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv19_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv19_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv19 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv19 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[19] == 1:
                self.conv19_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[19] == 1:
                self.conv19_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn19 = torch.nn.BatchNorm2d(params[1])



        params = [192, 64, 1, 1, 1, 1, 0, 0]
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



        params = [64, 384, 1, 1, 1, 1, 0, 0]
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



        params = [384, 384, 3, 3, 1, 1, 1, 1]
        params.append(384)
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


        self.conv_bn22 = torch.nn.BatchNorm2d(params[1])



        params = [384, 64, 1, 1, 1, 1, 0, 0]
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



        params = [64, 384, 1, 1, 1, 1, 0, 0]
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



        params = [384, 384, 3, 3, 1, 1, 1, 1]
        params.append(384)
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
            self.decompo_list[25] = 0
            if self.decompo_list[25] == 1:
                self.conv25_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv25_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[25] == 2:
                self.conv25_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv25_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv25_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv25_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[25] == 3:
                self.conv25_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv25_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[25] == 4:
                self.conv25_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv25_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv25_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv25_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv25 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv25 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[25] == 1:
                self.conv25_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[25] == 1:
                self.conv25_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn25 = torch.nn.BatchNorm2d(params[1])



        params = [384, 64, 1, 1, 1, 1, 0, 0]
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



        params = [64, 384, 1, 1, 1, 1, 0, 0]
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
            if self.decompo_list[27] == 1:
                self.conv27_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv27_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[27] == 2:
                self.conv27_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv27_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv27_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv27_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[27] == 3:
                self.conv27_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv27_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[27] == 4:
                self.conv27_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv27_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv27_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv27_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv27 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv27 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[27] == 1:
                self.conv27_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[27] == 1:
                self.conv27_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn27 = torch.nn.BatchNorm2d(params[1])



        params = [384, 384, 3, 3, 1, 1, 1, 1]
        params.append(384)
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
            self.decompo_list[28] = 0
            if self.decompo_list[28] == 1:
                self.conv28_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv28_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[28] == 2:
                self.conv28_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv28_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv28_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv28_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[28] == 3:
                self.conv28_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv28_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[28] == 4:
                self.conv28_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv28_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv28_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv28_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv28 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv28 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[28] == 1:
                self.conv28_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[28] == 1:
                self.conv28_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn28 = torch.nn.BatchNorm2d(params[1])



        params = [384, 64, 1, 1, 1, 1, 0, 0]
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



        # [6, 96, 3, 1],


        params = [64, 384, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[29] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[29] / 4) * 4)
            if self.widen_list[30] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[30] / 4) * 4)
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
            elif self.decompo_list[30] == 2:
                self.conv30_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv30_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv30_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv30_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
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


        self.conv_bn30 = torch.nn.BatchNorm2d(params[1])



        params = [384, 384, 3, 3, 1, 1, 1, 1]
        params.append(384)
        if self.widen_list != None:
            if self.widen_list[30] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[30] / 4) * 4)
            if self.widen_list[31] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[31] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[31] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[31])
                params[3] = params[3] + 2* int(self.kerneladd_list[31])
                params[6] = params[6] + int(self.kerneladd_list[31])
                params[7] = params[7] + int(self.kerneladd_list[31])
        if self.decompo_list != None:
            self.decompo_list[31] = 0
            if self.decompo_list[31] == 1:
                self.conv31_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv31_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[31] == 2:
                self.conv31_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv31_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv31_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv31_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[31] == 3:
                self.conv31_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv31_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[31] == 4:
                self.conv31_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv31_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv31_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv31_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv31 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv31 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[31] == 1:
                self.conv31_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[31] == 1:
                self.conv31_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn31 = torch.nn.BatchNorm2d(params[1])



        params = [384, 96, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[31] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[31] / 4) * 4)
            if self.widen_list[32] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[32] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[32] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[32])
                params[3] = params[3] + 2* int(self.kerneladd_list[32])
                params[6] = params[6] + int(self.kerneladd_list[32])
                params[7] = params[7] + int(self.kerneladd_list[32])
        if self.decompo_list != None:
            if self.decompo_list[32] == 1:
                self.conv32_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv32_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[32] == 2:
                self.conv32_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv32_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv32_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv32_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[32] == 3:
                self.conv32_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv32_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[32] == 4:
                self.conv32_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv32_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv32_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv32_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv32 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv32 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[32] == 1:
                self.conv32_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[32] == 1:
                self.conv32_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn32 = torch.nn.BatchNorm2d(params[1])



        params = [96, 576, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[32] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[32] / 4) * 4)
            if self.widen_list[33] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[33] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[33] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[33])
                params[3] = params[3] + 2* int(self.kerneladd_list[33])
                params[6] = params[6] + int(self.kerneladd_list[33])
                params[7] = params[7] + int(self.kerneladd_list[33])
        if self.decompo_list != None:
            if self.decompo_list[33] == 1:
                self.conv33_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv33_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[33] == 2:
                self.conv33_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv33_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv33_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv33_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[33] == 3:
                self.conv33_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv33_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[33] == 4:
                self.conv33_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv33_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv33_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv33_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv33 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv33 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[33] == 1:
                self.conv33_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[33] == 1:
                self.conv33_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn33 = torch.nn.BatchNorm2d(params[1])



        params = [576, 576, 3, 3, 1, 1, 1, 1]
        params.append(576)
        if self.widen_list != None:
            if self.widen_list[33] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[33] / 4) * 4)
            if self.widen_list[34] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[34] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[34] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[34])
                params[3] = params[3] + 2* int(self.kerneladd_list[34])
                params[6] = params[6] + int(self.kerneladd_list[34])
                params[7] = params[7] + int(self.kerneladd_list[34])
        if self.decompo_list != None:
            self.decompo_list[34] = 0
            if self.decompo_list[34] == 1:
                self.conv34_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv34_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[34] == 2:
                self.conv34_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv34_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv34_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv34_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[34] == 3:
                self.conv34_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv34_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[34] == 4:
                self.conv34_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv34_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv34_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv34_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv34 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv34 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[34] == 1:
                self.conv34_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[34] == 1:
                self.conv34_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn34 = torch.nn.BatchNorm2d(params[1])



        params = [576, 96, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[34] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[34] / 4) * 4)
            if self.widen_list[35] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[35] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[35] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[35])
                params[3] = params[3] + 2* int(self.kerneladd_list[35])
                params[6] = params[6] + int(self.kerneladd_list[35])
                params[7] = params[7] + int(self.kerneladd_list[35])
        if self.decompo_list != None:
            if self.decompo_list[35] == 1:
                self.conv35_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv35_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[35] == 2:
                self.conv35_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv35_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv35_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv35_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[35] == 3:
                self.conv35_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv35_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[35] == 4:
                self.conv35_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv35_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv35_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv35_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv35 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv35 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[35] == 1:
                self.conv35_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[35] == 1:
                self.conv35_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn35 = torch.nn.BatchNorm2d(params[1])



        params = [96, 576, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[35] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[35] / 4) * 4)
            if self.widen_list[36] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[36] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[36] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[36])
                params[3] = params[3] + 2* int(self.kerneladd_list[36])
                params[6] = params[6] + int(self.kerneladd_list[36])
                params[7] = params[7] + int(self.kerneladd_list[36])
        if self.decompo_list != None:
            if self.decompo_list[36] == 1:
                self.conv36_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv36_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[36] == 2:
                self.conv36_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv36_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv36_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv36_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[36] == 3:
                self.conv36_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv36_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[36] == 4:
                self.conv36_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv36_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv36_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv36_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv36 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv36 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[36] == 1:
                self.conv36_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[36] == 1:
                self.conv36_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn36 = torch.nn.BatchNorm2d(params[1])



        params = [576, 576, 3, 3, 1, 1, 1, 1]
        params.append(576)
        if self.widen_list != None:
            if self.widen_list[36] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[36] / 4) * 4)
            if self.widen_list[37] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[37] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[37] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[37])
                params[3] = params[3] + 2* int(self.kerneladd_list[37])
                params[6] = params[6] + int(self.kerneladd_list[37])
                params[7] = params[7] + int(self.kerneladd_list[37])
        if self.decompo_list != None:
            self.decompo_list[37] = 0
            if self.decompo_list[37] == 1:
                self.conv37_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv37_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[37] == 2:
                self.conv37_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv37_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv37_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv37_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[37] == 3:
                self.conv37_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv37_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[37] == 4:
                self.conv37_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv37_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv37_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv37_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv37 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv37 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[37] == 1:
                self.conv37_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[37] == 1:
                self.conv37_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn37 = torch.nn.BatchNorm2d(params[1])



        params = [576, 96, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[37] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[37] / 4) * 4)
            if self.widen_list[38] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[38] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[38] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[38])
                params[3] = params[3] + 2* int(self.kerneladd_list[38])
                params[6] = params[6] + int(self.kerneladd_list[38])
                params[7] = params[7] + int(self.kerneladd_list[38])
        if self.decompo_list != None:
            if self.decompo_list[38] == 1:
                self.conv38_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv38_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[38] == 2:
                self.conv38_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv38_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv38_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv38_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[38] == 3:
                self.conv38_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv38_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[38] == 4:
                self.conv38_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv38_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv38_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv38_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv38 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv38 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[38] == 1:
                self.conv38_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[38] == 1:
                self.conv38_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn38 = torch.nn.BatchNorm2d(params[1])


        # [6, 160, 3, 2],


        params = [96, 576, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[38] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[38] / 4) * 4)
            if self.widen_list[39] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[39] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[39] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[39])
                params[3] = params[3] + 2* int(self.kerneladd_list[39])
                params[6] = params[6] + int(self.kerneladd_list[39])
                params[7] = params[7] + int(self.kerneladd_list[39])
        if self.decompo_list != None:
            if self.decompo_list[39] == 1:
                self.conv39_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv39_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[39] == 2:
                self.conv39_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv39_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv39_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv39_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[39] == 3:
                self.conv39_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv39_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[39] == 4:
                self.conv39_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv39_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv39_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv39_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv39 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv39 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[39] == 1:
                self.conv39_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[39] == 1:
                self.conv39_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn39 = torch.nn.BatchNorm2d(params[1])



        params = [576, 576, 3, 3, 2, 2, 1, 1]
        params.append(576)
        if self.widen_list != None:
            if self.widen_list[39] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[39] / 4) * 4)
            if self.widen_list[40] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[40] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[40] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[40])
                params[3] = params[3] + 2* int(self.kerneladd_list[40])
                params[6] = params[6] + int(self.kerneladd_list[40])
                params[7] = params[7] + int(self.kerneladd_list[40])
        if self.decompo_list != None:
            self.decompo_list[40] = 0
            if self.decompo_list[40] == 1:
                self.conv40_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv40_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[40] == 2:
                self.conv40_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv40_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv40_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv40_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[40] == 3:
                self.conv40_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv40_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[40] == 4:
                self.conv40_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv40_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv40_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv40_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv40 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv40 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[40] == 1:
                self.conv40_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[40] == 1:
                self.conv40_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn40 = torch.nn.BatchNorm2d(params[1])



        params = [576, 160, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[40] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[40] / 4) * 4)
            if self.widen_list[41] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[41] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[41] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[41])
                params[3] = params[3] + 2* int(self.kerneladd_list[41])
                params[6] = params[6] + int(self.kerneladd_list[41])
                params[7] = params[7] + int(self.kerneladd_list[41])
        if self.decompo_list != None:
            if self.decompo_list[41] == 1:
                self.conv41_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv41_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[41] == 2:
                self.conv41_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv41_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv41_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv41_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[41] == 3:
                self.conv41_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv41_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[41] == 4:
                self.conv41_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv41_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv41_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv41_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv41 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv41 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[41] == 1:
                self.conv41_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[41] == 1:
                self.conv41_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn41 = torch.nn.BatchNorm2d(params[1])



        params = [160, 960, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[41] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[41] / 4) * 4)
            if self.widen_list[42] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[42] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[42] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[42])
                params[3] = params[3] + 2* int(self.kerneladd_list[42])
                params[6] = params[6] + int(self.kerneladd_list[42])
                params[7] = params[7] + int(self.kerneladd_list[42])
        if self.decompo_list != None:
            if self.decompo_list[42] == 1:
                self.conv42_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv42_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[42] == 2:
                self.conv42_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv42_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv42_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv42_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[42] == 3:
                self.conv42_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv42_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[42] == 4:
                self.conv42_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv42_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv42_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv42_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv42 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv42 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[42] == 1:
                self.conv42_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[42] == 1:
                self.conv42_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn42 = torch.nn.BatchNorm2d(params[1])



        params = [960, 960, 3, 3, 1, 1, 1, 1]
        params.append(960)
        if self.widen_list != None:
            if self.widen_list[42] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[42] / 4) * 4)
            if self.widen_list[43] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[43] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[43] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[43])
                params[3] = params[3] + 2* int(self.kerneladd_list[43])
                params[6] = params[6] + int(self.kerneladd_list[43])
                params[7] = params[7] + int(self.kerneladd_list[43])
        if self.decompo_list != None:
            self.decompo_list[43] = 0
            if self.decompo_list[43] == 1:
                self.conv43_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv43_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[43] == 2:
                self.conv43_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv43_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv43_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv43_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[43] == 3:
                self.conv43_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv43_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[43] == 4:
                self.conv43_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv43_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv43_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv43_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv43 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv43 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[43] == 1:
                self.conv43_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[43] == 1:
                self.conv43_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn43 = torch.nn.BatchNorm2d(params[1])



        params = [960, 160, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[43] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[43] / 4) * 4)
            if self.widen_list[44] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[44] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[44] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[44])
                params[3] = params[3] + 2* int(self.kerneladd_list[44])
                params[6] = params[6] + int(self.kerneladd_list[44])
                params[7] = params[7] + int(self.kerneladd_list[44])
        if self.decompo_list != None:
            if self.decompo_list[44] == 1:
                self.conv44_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv44_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[44] == 2:
                self.conv44_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv44_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv44_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv44_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[44] == 3:
                self.conv44_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv44_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[44] == 4:
                self.conv44_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv44_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv44_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv44_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv44 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv44 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[44] == 1:
                self.conv44_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[44] == 1:
                self.conv44_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn44 = torch.nn.BatchNorm2d(params[1])



        params = [160, 960, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[44] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[44] / 4) * 4)
            if self.widen_list[45] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[45] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[45] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[45])
                params[3] = params[3] + 2* int(self.kerneladd_list[45])
                params[6] = params[6] + int(self.kerneladd_list[45])
                params[7] = params[7] + int(self.kerneladd_list[45])
        if self.decompo_list != None:
            if self.decompo_list[45] == 1:
                self.conv45_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv45_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[45] == 2:
                self.conv45_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv45_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv45_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv45_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[45] == 3:
                self.conv45_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv45_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[45] == 4:
                self.conv45_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv45_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv45_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv45_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv45 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv45 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[45] == 1:
                self.conv45_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[45] == 1:
                self.conv45_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn45 = torch.nn.BatchNorm2d(params[1])



        params = [960, 960, 3, 3, 1, 1, 1, 1]
        params.append(960)
        if self.widen_list != None:
            if self.widen_list[45] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[45] / 4) * 4)
            if self.widen_list[46] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[46] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[46] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[46])
                params[3] = params[3] + 2* int(self.kerneladd_list[46])
                params[6] = params[6] + int(self.kerneladd_list[46])
                params[7] = params[7] + int(self.kerneladd_list[46])
        if self.decompo_list != None:
            self.decompo_list[46] = 0
            if self.decompo_list[46] == 1:
                self.conv46_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv46_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[46] == 2:
                self.conv46_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv46_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv46_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv46_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[46] == 3:
                self.conv46_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv46_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[46] == 4:
                self.conv46_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv46_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv46_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv46_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv46 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv46 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[46] == 1:
                self.conv46_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[46] == 1:
                self.conv46_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn46 = torch.nn.BatchNorm2d(params[1])



        params = [960, 160, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[46] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[46] / 4) * 4)
            if self.widen_list[47] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[47] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[47] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[47])
                params[3] = params[3] + 2* int(self.kerneladd_list[47])
                params[6] = params[6] + int(self.kerneladd_list[47])
                params[7] = params[7] + int(self.kerneladd_list[47])
        if self.decompo_list != None:
            if self.decompo_list[47] == 1:
                self.conv47_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv47_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[47] == 2:
                self.conv47_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv47_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv47_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv47_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[47] == 3:
                self.conv47_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv47_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[47] == 4:
                self.conv47_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv47_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv47_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv47_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv47 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv47 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[47] == 1:
                self.conv47_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[47] == 1:
                self.conv47_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn47 = torch.nn.BatchNorm2d(params[1])


        # [6, 320, 1, 1],


        params = [160, 960, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[47] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[47] / 4) * 4)
            if self.widen_list[48] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[48] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[48] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[48])
                params[3] = params[3] + 2* int(self.kerneladd_list[48])
                params[6] = params[6] + int(self.kerneladd_list[48])
                params[7] = params[7] + int(self.kerneladd_list[48])
        if self.decompo_list != None:
            if self.decompo_list[48] == 1:
                self.conv48_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv48_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[48] == 2:
                self.conv48_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv48_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv48_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv48_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[48] == 3:
                self.conv48_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv48_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[48] == 4:
                self.conv48_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv48_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv48_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv48_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv48 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv48 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[48] == 1:
                self.conv48_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[48] == 1:
                self.conv48_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn48 = torch.nn.BatchNorm2d(params[1])



        params = [960, 960, 3, 3, 1, 1, 1, 1]
        params.append(960)
        if self.widen_list != None:
            if self.widen_list[48] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[48] / 4) * 4)
            if self.widen_list[49] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[49] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[49] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[49])
                params[3] = params[3] + 2* int(self.kerneladd_list[49])
                params[6] = params[6] + int(self.kerneladd_list[49])
                params[7] = params[7] + int(self.kerneladd_list[49])
        if self.decompo_list != None:
            self.decompo_list[49] = 0
            if self.decompo_list[49] == 1:
                self.conv49_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv49_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[49] == 2:
                self.conv49_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv49_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv49_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv49_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[49] == 3:
                self.conv49_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv49_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            elif self.decompo_list[49] == 4:
                self.conv49_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv49_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv49_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
                self.conv49_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
            else:
                self.conv49 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        else:
            self.conv49 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), groups=params[8], bias=False)
        if self.deepen_list != None:
            if self.deepen_list[49] == 1:
                self.conv49_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[49] == 1:
                self.conv49_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn49 = torch.nn.BatchNorm2d(params[1])



        params = [960, 320, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[49] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[49] / 4) * 4)
            if self.widen_list[50] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[50] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[50] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[50])
                params[3] = params[3] + 2* int(self.kerneladd_list[50])
                params[6] = params[6] + int(self.kerneladd_list[50])
                params[7] = params[7] + int(self.kerneladd_list[50])
        if self.decompo_list != None:
            if self.decompo_list[50] == 1:
                self.conv50_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv50_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[50] == 2:
                self.conv50_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv50_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv50_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv50_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[50] == 3:
                self.conv50_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv50_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            elif self.decompo_list[50] == 4:
                self.conv50_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv50_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv50_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
                self.conv50_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
            else:
                self.conv50 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        else:
            self.conv50 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]), bias=False)
        if self.deepen_list != None:
            if self.deepen_list[50] == 1:
                self.conv50_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[50] == 1:
                self.conv50_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn50 = torch.nn.BatchNorm2d(params[1])



        params = [320, 1280, 1, 1, 1, 1, 0, 0]
        if self.widen_list != None:
            if self.widen_list[50] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[50] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[51] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[51])
                params[3] = params[3] + 2* int(self.kerneladd_list[51])
                params[6] = params[6] + int(self.kerneladd_list[51])
                params[7] = params[7] + int(self.kerneladd_list[51])
        if self.decompo_list != None:
            if self.decompo_list[51] == 1:
                self.conv51_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv51_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[51] == 2:
                self.conv51_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv51_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv51_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv51_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[51] == 3:
                self.conv51_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv51_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[51] == 4:
                self.conv51_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv51_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv51_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv51_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv51 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv51 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[51] == 1:
                self.conv51_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[51] == 1:
                self.conv51_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        self.conv_bn51 = torch.nn.BatchNorm2d(params[1])



        params = [1280, 1000]
        if self.widen_list != None:
            pass
        if self.decompo_list != None:
            if self.decompo_list[52] == 1:
                self.classifier_0 = torch.nn.Linear(params[0], int(params[1]/2))
                self.classifier_1 = torch.nn.Linear(params[0], int(params[1]/2))
            elif self.decompo_list[52] == 2:
                self.classifier_0 = torch.nn.Linear(params[0], int(params[1]/4))
                self.classifier_1 = torch.nn.Linear(params[0], int(params[1]/4))
                self.classifier_2 = torch.nn.Linear(params[0], int(params[1]/4))
                self.classifier_3 = torch.nn.Linear(params[0], int(params[1]/4))
            elif self.decompo_list[52] == 3:
                self.classifier_0 = torch.nn.Linear(int(params[0]/2), params[1])
                self.classifier_1 = torch.nn.Linear(int(params[0]/2), params[1])
            elif self.decompo_list[52] == 4:
                self.classifier_0 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_1 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_2 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_3 = torch.nn.Linear(int(params[0]/4), params[1])
            else:
                self.classifier = torch.nn.Linear(params[0], params[1])
        else:
            self.classifier = torch.nn.Linear(params[0], params[1])


        if self.deepen_list != None:
            if self.deepen_list[52] == 1:
                self.classifier_dp = torch.nn.Linear(params[1], params[1])
        if self.skipcon_list != None:
            if self.skipcon_list[52] == 1:
                self.classifier_sk = torch.nn.Linear(params[1], params[1])

        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 224, 224)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[0] == 1:
                X1_0 = self.conv0_0(X1)
                X1_1 = self.conv0_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[0] == 2:
                X1_0 = self.conv0_0(X1)
                X1_1 = self.conv0_1(X1)
                X1_2 = self.conv0_2(X1)
                X1_3 = self.conv0_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            else:
                X1 = self.conv0(X1)
        else:
            X1 = self.conv0(X1)

        if self.dummy_list != None:
            if self.dummy_list[0] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[0]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn0(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[0] == 1:
                X1 = self.conv0_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[0] == 1:
                X1_skip = X1
                X1 = self.conv0_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        

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
            elif self.decompo_list[1] == 3:
                X1_0 = self.conv1_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv1_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[1] == 4:
                X1_0 = self.conv1_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv1_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv1_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv1_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv1(X1)
        else:
            X1 = self.conv1(X1)

        if self.dummy_list != None:
            if self.dummy_list[1] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[1]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn1(X1)

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

        X1 = self.conv_bn2(X1)
        
        # [6, 24, 2, 2]


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

        X1 = self.conv_bn3(X1)

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

        X1 = self.conv_bn4(X1)

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

        X1 = self.conv_bn5(X1)

        X0_skip = X1

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

        X1 = self.conv_bn6(X1)

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

        X1 = self.conv_bn7(X1)

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

        X1 = self.conv_bn8(X1)
        X1 = X1 + X0_skip

        # [6, 32, 3, 2],


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

        X1 = self.conv_bn9(X1)

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

        X1 = self.conv_bn10(X1)

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

        X1 = self.conv_bn11(X1)

        X0_skip = X1

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

        X1 = self.conv_bn12(X1)

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

        X1 = self.conv_bn13(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[13] == 1:
                X1 = self.conv13_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[13] == 1:
                X1_skip = X1
                X1 = self.conv13_sk(X1)
                X1 = self.relu(X1 + X1_skip)


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

        X1 = self.conv_bn14(X1)
        X1 = X1 + X0_skip

        X0_skip = X1

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

        X1 = self.conv_bn15(X1)

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

        X1 = self.conv_bn16(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[16] == 1:
                X1 = self.conv16_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[16] == 1:
                X1_skip = X1
                X1 = self.conv16_sk(X1)
                X1 = self.relu(X1 + X1_skip)


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

        X1 = self.conv_bn17(X1)
        X1 = X1 + X0_skip

        # [6, 64, 4, 2],


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

        X1 = self.conv_bn18(X1)

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

        X1 = self.conv_bn19(X1)

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

        X1 = self.conv_bn20(X1)

        X0_skip = X1

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

        X1 = self.conv_bn21(X1)

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

        X1 = self.conv_bn22(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[22] == 1:
                X1 = self.conv22_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[22] == 1:
                X1_skip = X1
                X1 = self.conv22_sk(X1)
                X1 = self.relu(X1 + X1_skip)


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

        X1 = self.conv_bn23(X1)
        X1 = X1 + X0_skip

        X0_skip = X1

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

        X1 = self.conv_bn24(X1)

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

        X1 = self.conv_bn25(X1)

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

        X1 = self.conv_bn26(X1)
        X1 = X1 + X0_skip

        X0_skip = X1

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

        X1 = self.conv_bn27(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[27] == 1:
                X1 = self.conv27_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[27] == 1:
                X1_skip = X1
                X1 = self.conv27_sk(X1)
                X1 = self.relu(X1 + X1_skip)


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

        X1 = self.conv_bn28(X1)

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

        X1 = self.conv_bn29(X1)
        X1 = X1 + X0_skip

        # [6, 96, 3, 1],


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[30] == 1:
                X1_0 = self.conv30_0(X1)
                X1_1 = self.conv30_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[30] == 2:
                X1_0 = self.conv30_0(X1)
                X1_1 = self.conv30_1(X1)
                X1_2 = self.conv30_2(X1)
                X1_3 = self.conv30_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
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

        X1 = self.conv_bn30(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[30] == 1:
                X1 = self.conv30_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[30] == 1:
                X1_skip = X1
                X1 = self.conv30_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[31] == 1:
                X1_0 = self.conv31_0(X1)
                X1_1 = self.conv31_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[31] == 2:
                X1_0 = self.conv31_0(X1)
                X1_1 = self.conv31_1(X1)
                X1_2 = self.conv31_2(X1)
                X1_3 = self.conv31_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[31] == 3:
                X1_0 = self.conv31_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv31_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[31] == 4:
                X1_0 = self.conv31_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv31_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv31_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv31_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv31(X1)
        else:
            X1 = self.conv31(X1)

        if self.dummy_list != None:
            if self.dummy_list[31] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[31]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn31(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[31] == 1:
                X1 = self.conv31_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[31] == 1:
                X1_skip = X1
                X1 = self.conv31_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[32] == 1:
                X1_0 = self.conv32_0(X1)
                X1_1 = self.conv32_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[32] == 2:
                X1_0 = self.conv32_0(X1)
                X1_1 = self.conv32_1(X1)
                X1_2 = self.conv32_2(X1)
                X1_3 = self.conv32_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[32] == 3:
                X1_0 = self.conv32_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv32_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[32] == 4:
                X1_0 = self.conv32_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv32_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv32_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv32_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv32(X1)
        else:
            X1 = self.conv32(X1)

        if self.dummy_list != None:
            if self.dummy_list[32] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[32]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn32(X1)

        X0_skip = X1

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[33] == 1:
                X1_0 = self.conv33_0(X1)
                X1_1 = self.conv33_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[33] == 2:
                X1_0 = self.conv33_0(X1)
                X1_1 = self.conv33_1(X1)
                X1_2 = self.conv33_2(X1)
                X1_3 = self.conv33_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[33] == 3:
                X1_0 = self.conv33_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv33_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[33] == 4:
                X1_0 = self.conv33_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv33_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv33_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv33_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv33(X1)
        else:
            X1 = self.conv33(X1)

        if self.dummy_list != None:
            if self.dummy_list[33] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[33]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn33(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[33] == 1:
                X1 = self.conv33_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[33] == 1:
                X1_skip = X1
                X1 = self.conv33_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[34] == 1:
                X1_0 = self.conv34_0(X1)
                X1_1 = self.conv34_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[34] == 2:
                X1_0 = self.conv34_0(X1)
                X1_1 = self.conv34_1(X1)
                X1_2 = self.conv34_2(X1)
                X1_3 = self.conv34_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[34] == 3:
                X1_0 = self.conv34_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv34_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[34] == 4:
                X1_0 = self.conv34_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv34_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv34_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv34_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv34(X1)
        else:
            X1 = self.conv34(X1)

        if self.dummy_list != None:
            if self.dummy_list[34] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[34]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn34(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[34] == 1:
                X1 = self.conv34_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[34] == 1:
                X1_skip = X1
                X1 = self.conv34_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[35] == 1:
                X1_0 = self.conv35_0(X1)
                X1_1 = self.conv35_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[35] == 2:
                X1_0 = self.conv35_0(X1)
                X1_1 = self.conv35_1(X1)
                X1_2 = self.conv35_2(X1)
                X1_3 = self.conv35_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[35] == 3:
                X1_0 = self.conv35_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv35_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[35] == 4:
                X1_0 = self.conv35_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv35_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv35_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv35_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv35(X1)
        else:
            X1 = self.conv35(X1)

        if self.dummy_list != None:
            if self.dummy_list[35] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[35]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn35(X1)
        X1 = X1 + X0_skip

        X0_skip = X1

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[36] == 1:
                X1_0 = self.conv36_0(X1)
                X1_1 = self.conv36_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[36] == 2:
                X1_0 = self.conv36_0(X1)
                X1_1 = self.conv36_1(X1)
                X1_2 = self.conv36_2(X1)
                X1_3 = self.conv36_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[36] == 3:
                X1_0 = self.conv36_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv36_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[36] == 4:
                X1_0 = self.conv36_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv36_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv36_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv36_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv36(X1)
        else:
            X1 = self.conv36(X1)

        if self.dummy_list != None:
            if self.dummy_list[36] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[36]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn36(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[36] == 1:
                X1 = self.conv36_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[36] == 1:
                X1_skip = X1
                X1 = self.conv36_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[37] == 1:
                X1_0 = self.conv37_0(X1)
                X1_1 = self.conv37_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[37] == 2:
                X1_0 = self.conv37_0(X1)
                X1_1 = self.conv37_1(X1)
                X1_2 = self.conv37_2(X1)
                X1_3 = self.conv37_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[37] == 3:
                X1_0 = self.conv37_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv37_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[37] == 4:
                X1_0 = self.conv37_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv37_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv37_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv37_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv37(X1)
        else:
            X1 = self.conv37(X1)

        if self.dummy_list != None:
            if self.dummy_list[37] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[37]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn37(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[37] == 1:
                X1 = self.conv37_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[37] == 1:
                X1_skip = X1
                X1 = self.conv37_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[38] == 1:
                X1_0 = self.conv38_0(X1)
                X1_1 = self.conv38_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[38] == 2:
                X1_0 = self.conv38_0(X1)
                X1_1 = self.conv38_1(X1)
                X1_2 = self.conv38_2(X1)
                X1_3 = self.conv38_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[38] == 3:
                X1_0 = self.conv38_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv38_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[38] == 4:
                X1_0 = self.conv38_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv38_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv38_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv38_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv38(X1)
        else:
            X1 = self.conv38(X1)

        if self.dummy_list != None:
            if self.dummy_list[38] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[38]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn38(X1)
        X1 = X1 + X0_skip

        # [6, 160, 3, 2],


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[39] == 1:
                X1_0 = self.conv39_0(X1)
                X1_1 = self.conv39_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[39] == 2:
                X1_0 = self.conv39_0(X1)
                X1_1 = self.conv39_1(X1)
                X1_2 = self.conv39_2(X1)
                X1_3 = self.conv39_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[39] == 3:
                X1_0 = self.conv39_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv39_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[39] == 4:
                X1_0 = self.conv39_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv39_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv39_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv39_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv39(X1)
        else:
            X1 = self.conv39(X1)

        if self.dummy_list != None:
            if self.dummy_list[39] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[39]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn39(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[39] == 1:
                X1 = self.conv39_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[39] == 1:
                X1_skip = X1
                X1 = self.conv39_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[40] == 1:
                X1_0 = self.conv40_0(X1)
                X1_1 = self.conv40_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[40] == 2:
                X1_0 = self.conv40_0(X1)
                X1_1 = self.conv40_1(X1)
                X1_2 = self.conv40_2(X1)
                X1_3 = self.conv40_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[40] == 3:
                X1_0 = self.conv40_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv40_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[40] == 4:
                X1_0 = self.conv40_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv40_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv40_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv40_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv40(X1)
        else:
            X1 = self.conv40(X1)

        if self.dummy_list != None:
            if self.dummy_list[40] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[40]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn40(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[40] == 1:
                X1 = self.conv40_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[40] == 1:
                X1_skip = X1
                X1 = self.conv40_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[41] == 1:
                X1_0 = self.conv41_0(X1)
                X1_1 = self.conv41_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[41] == 2:
                X1_0 = self.conv41_0(X1)
                X1_1 = self.conv41_1(X1)
                X1_2 = self.conv41_2(X1)
                X1_3 = self.conv41_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[41] == 3:
                X1_0 = self.conv41_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv41_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[41] == 4:
                X1_0 = self.conv41_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv41_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv41_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv41_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv41(X1)
        else:
            X1 = self.conv41(X1)

        if self.dummy_list != None:
            if self.dummy_list[41] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[41]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn41(X1)

        X0_skip = X1

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[42] == 1:
                X1_0 = self.conv42_0(X1)
                X1_1 = self.conv42_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[42] == 2:
                X1_0 = self.conv42_0(X1)
                X1_1 = self.conv42_1(X1)
                X1_2 = self.conv42_2(X1)
                X1_3 = self.conv42_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[42] == 3:
                X1_0 = self.conv42_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv42_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[42] == 4:
                X1_0 = self.conv42_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv42_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv42_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv42_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv42(X1)
        else:
            X1 = self.conv42(X1)

        if self.dummy_list != None:
            if self.dummy_list[42] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[42]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn42(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[42] == 1:
                X1 = self.conv42_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[42] == 1:
                X1_skip = X1
                X1 = self.conv42_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[43] == 1:
                X1_0 = self.conv43_0(X1)
                X1_1 = self.conv43_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[43] == 2:
                X1_0 = self.conv43_0(X1)
                X1_1 = self.conv43_1(X1)
                X1_2 = self.conv43_2(X1)
                X1_3 = self.conv43_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[43] == 3:
                X1_0 = self.conv43_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv43_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[43] == 4:
                X1_0 = self.conv43_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv43_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv43_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv43_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv43(X1)
        else:
            X1 = self.conv43(X1)

        if self.dummy_list != None:
            if self.dummy_list[43] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[43]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn43(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[43] == 1:
                X1 = self.conv43_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[43] == 1:
                X1_skip = X1
                X1 = self.conv43_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[44] == 1:
                X1_0 = self.conv44_0(X1)
                X1_1 = self.conv44_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[44] == 2:
                X1_0 = self.conv44_0(X1)
                X1_1 = self.conv44_1(X1)
                X1_2 = self.conv44_2(X1)
                X1_3 = self.conv44_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[44] == 3:
                X1_0 = self.conv44_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv44_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[44] == 4:
                X1_0 = self.conv44_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv44_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv44_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv44_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv44(X1)
        else:
            X1 = self.conv44(X1)

        if self.dummy_list != None:
            if self.dummy_list[44] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[44]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn44(X1)
        X1 = X1 + X0_skip

        X0_skip = X1

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[45] == 1:
                X1_0 = self.conv45_0(X1)
                X1_1 = self.conv45_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[45] == 2:
                X1_0 = self.conv45_0(X1)
                X1_1 = self.conv45_1(X1)
                X1_2 = self.conv45_2(X1)
                X1_3 = self.conv45_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[45] == 3:
                X1_0 = self.conv45_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv45_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[45] == 4:
                X1_0 = self.conv45_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv45_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv45_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv45_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv45(X1)
        else:
            X1 = self.conv45(X1)

        if self.dummy_list != None:
            if self.dummy_list[45] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[45]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn45(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[45] == 1:
                X1 = self.conv45_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[45] == 1:
                X1_skip = X1
                X1 = self.conv45_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[46] == 1:
                X1_0 = self.conv46_0(X1)
                X1_1 = self.conv46_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[46] == 2:
                X1_0 = self.conv46_0(X1)
                X1_1 = self.conv46_1(X1)
                X1_2 = self.conv46_2(X1)
                X1_3 = self.conv46_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[46] == 3:
                X1_0 = self.conv46_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv46_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[46] == 4:
                X1_0 = self.conv46_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv46_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv46_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv46_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv46(X1)
        else:
            X1 = self.conv46(X1)

        if self.dummy_list != None:
            if self.dummy_list[46] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[46]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn46(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[46] == 1:
                X1 = self.conv46_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[46] == 1:
                X1_skip = X1
                X1 = self.conv46_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[47] == 1:
                X1_0 = self.conv47_0(X1)
                X1_1 = self.conv47_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[47] == 2:
                X1_0 = self.conv47_0(X1)
                X1_1 = self.conv47_1(X1)
                X1_2 = self.conv47_2(X1)
                X1_3 = self.conv47_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[47] == 3:
                X1_0 = self.conv47_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv47_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[47] == 4:
                X1_0 = self.conv47_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv47_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv47_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv47_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv47(X1)
        else:
            X1 = self.conv47(X1)

        if self.dummy_list != None:
            if self.dummy_list[47] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[47]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn47(X1)
        X1 = X1 + X0_skip

        # [6, 320, 1, 1],


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[48] == 1:
                X1_0 = self.conv48_0(X1)
                X1_1 = self.conv48_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[48] == 2:
                X1_0 = self.conv48_0(X1)
                X1_1 = self.conv48_1(X1)
                X1_2 = self.conv48_2(X1)
                X1_3 = self.conv48_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[48] == 3:
                X1_0 = self.conv48_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv48_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[48] == 4:
                X1_0 = self.conv48_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv48_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv48_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv48_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv48(X1)
        else:
            X1 = self.conv48(X1)

        if self.dummy_list != None:
            if self.dummy_list[48] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[48]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn48(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[48] == 1:
                X1 = self.conv48_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[48] == 1:
                X1_skip = X1
                X1 = self.conv48_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[49] == 1:
                X1_0 = self.conv49_0(X1)
                X1_1 = self.conv49_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[49] == 2:
                X1_0 = self.conv49_0(X1)
                X1_1 = self.conv49_1(X1)
                X1_2 = self.conv49_2(X1)
                X1_3 = self.conv49_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[49] == 3:
                X1_0 = self.conv49_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv49_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[49] == 4:
                X1_0 = self.conv49_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv49_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv49_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv49_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv49(X1)
        else:
            X1 = self.conv49(X1)

        if self.dummy_list != None:
            if self.dummy_list[49] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[49]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn49(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[49] == 1:
                X1 = self.conv49_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[49] == 1:
                X1_skip = X1
                X1 = self.conv49_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[50] == 1:
                X1_0 = self.conv50_0(X1)
                X1_1 = self.conv50_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[50] == 2:
                X1_0 = self.conv50_0(X1)
                X1_1 = self.conv50_1(X1)
                X1_2 = self.conv50_2(X1)
                X1_3 = self.conv50_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[50] == 3:
                X1_0 = self.conv50_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv50_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[50] == 4:
                X1_0 = self.conv50_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv50_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv50_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv50_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv50(X1)
        else:
            X1 = self.conv50(X1)

        if self.dummy_list != None:
            if self.dummy_list[50] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[50]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn50(X1)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[51] == 1:
                X1_0 = self.conv51_0(X1)
                X1_1 = self.conv51_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[51] == 2:
                X1_0 = self.conv51_0(X1)
                X1_1 = self.conv51_1(X1)
                X1_2 = self.conv51_2(X1)
                X1_3 = self.conv51_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[51] == 3:
                X1_0 = self.conv51_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_1 = self.conv51_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[51] == 4:
                X1_0 = self.conv51_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])
                X1_1 = self.conv51_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])
                X1_2 = self.conv51_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])
                X1_3 = self.conv51_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.conv51(X1)
        else:
            X1 = self.conv51(X1)

        if self.dummy_list != None:
            if self.dummy_list[51] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[51]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.conv_bn51(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[51] == 1:
                X1 = self.conv51_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[51] == 1:
                X1_skip = X1
                X1 = self.conv51_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1 = nn.functional.adaptive_avg_pool2d(X1, 1)
        X1 = X1.view(-1, 1280)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[52] == 1:
                X1_0 = self.classifier_0(X1)
                X1_1 = self.classifier_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[52] == 2:
                X1_0 = self.classifier_0(X1)
                X1_1 = self.classifier_1(X1)
                X1_2 = self.classifier_2(X1)
                X1_3 = self.classifier_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[52] == 3:
                X1_0 = self.classifier_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])
                X1_1 = self.classifier_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[52] == 4:
                X1_0 = self.classifier_0(X1[:, :int(torch.floor_divide(X1_shape[1],4))])
                X1_1 = self.classifier_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2))])
                X1_2 = self.classifier_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4))])
                X1_3 = self.classifier_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.classifier(X1)
        else:
             X1 = self.classifier(X1)

        if self.dummy_list != None:
            if self.dummy_list[52] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[52]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.logsoftmax(X1)
        return X1



batch_size = 1
input_features = 150528
torch.manual_seed(1234)
X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_10(input_features).to(cuda_device)

# End Call Model
model.eval()
new_out = model(X)