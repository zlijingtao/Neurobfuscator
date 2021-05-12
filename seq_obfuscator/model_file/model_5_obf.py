import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

#This model stands for vgg-11 
#model_id = 4 (regardless of the file name)

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
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        params = [3, 64, 3, 3, 1, 1, 1, 1]
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
                self.conv0_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv0_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[0] == 2:
                self.conv0_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv0_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv0_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv0_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv0 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv0 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[0] == 1:
                self.conv0_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[0] == 1:
                self.conv0_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [64, 128, 3, 3, 1, 1, 1, 1]
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
                self.conv1_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv1_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[1] == 2:
                self.conv1_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv1_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv1_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv1_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[1] == 3:
                self.conv1_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv1_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[1] == 4:
                self.conv1_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv1_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv1_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv1_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv1 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv1 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[1] == 1:
                self.conv1_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[1] == 1:
                self.conv1_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [128, 256, 3, 3, 1, 1, 1, 1]
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
                self.conv2_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv2_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[2] == 2:
                self.conv2_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv2_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv2_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv2_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[2] == 3:
                self.conv2_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv2_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[2] == 4:
                self.conv2_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv2_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv2_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv2_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv2 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv2 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[2] == 1:
                self.conv2_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[2] == 1:
                self.conv2_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [256, 256, 3, 3, 1, 1, 1, 1]
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
                self.conv3_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv3_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[3] == 2:
                self.conv3_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv3_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv3_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv3_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[3] == 3:
                self.conv3_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv3_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[3] == 4:
                self.conv3_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv3_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv3_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv3_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv3 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv3 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[3] == 1:
                self.conv3_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[3] == 1:
                self.conv3_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [256, 512, 3, 3, 1, 1, 1, 1]
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
                self.conv4_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv4_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[4] == 2:
                self.conv4_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv4_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv4_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv4_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[4] == 3:
                self.conv4_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv4_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[4] == 4:
                self.conv4_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv4_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv4_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv4_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv4 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv4 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[4] == 1:
                self.conv4_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[4] == 1:
                self.conv4_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [512, 512, 3, 3, 1, 1, 1, 1]
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
                self.conv5_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv5_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[5] == 2:
                self.conv5_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv5_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv5_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv5_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[5] == 3:
                self.conv5_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv5_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[5] == 4:
                self.conv5_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv5_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv5_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv5_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv5 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv5 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[5] == 1:
                self.conv5_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[5] == 1:
                self.conv5_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [512, 512, 3, 3, 1, 1, 1, 1]
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
                self.conv6_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv6_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[6] == 2:
                self.conv6_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv6_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv6_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv6_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[6] == 3:
                self.conv6_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv6_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[6] == 4:
                self.conv6_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv6_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv6_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv6_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv6 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv6 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[6] == 1:
                self.conv6_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[6] == 1:
                self.conv6_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [512, 512, 3, 3, 1, 1, 1, 1]
        if self.widen_list != None:
            if self.widen_list[6] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[6] / 4) * 4)
        if self.kerneladd_list != None:
            if self.kerneladd_list[7] > 0:
                params[2] = params[2] + 2* int(self.kerneladd_list[7])
                params[3] = params[3] + 2* int(self.kerneladd_list[7])
                params[6] = params[6] + int(self.kerneladd_list[7])
                params[7] = params[7] + int(self.kerneladd_list[7])
        if self.decompo_list != None:
            if self.decompo_list[7] == 1:
                self.conv7_0 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv7_1 = torch.nn.Conv2d(params[0], int(params[1]/2), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[7] == 2:
                self.conv7_0 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv7_1 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv7_2 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv7_3 = torch.nn.Conv2d(params[0], int(params[1]/4), (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[7] == 3:
                self.conv7_0 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv7_1 = torch.nn.Conv2d(int(params[0]/2), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            elif self.decompo_list[7] == 4:
                self.conv7_0 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv7_1 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv7_2 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
                self.conv7_3 = torch.nn.Conv2d(int(params[0]/4), params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
            else:
                self.conv7 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        else:
            self.conv7 = torch.nn.Conv2d(params[0], params[1], (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7]))
        if self.deepen_list != None:
            if self.deepen_list[7] == 1:
                self.conv7_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))
        if self.skipcon_list != None:
            if self.skipcon_list[7] == 1:
                self.conv7_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))


        params = [512, 512]
        if self.widen_list != None:
            if self.widen_list[8] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[8] / 4) * 4)
        if self.decompo_list != None:
            if self.decompo_list[8] == 1:
                self.fc0_0 = torch.nn.Linear(params[0], int(params[1]/2))
                self.fc0_1 = torch.nn.Linear(params[0], int(params[1]/2))
            elif self.decompo_list[8] == 2:
                self.fc0_0 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc0_1 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc0_2 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc0_3 = torch.nn.Linear(params[0], int(params[1]/4))
            elif self.decompo_list[8] == 3:
                self.fc0_0 = torch.nn.Linear(int(params[0]/2), params[1])
                self.fc0_1 = torch.nn.Linear(int(params[0]/2), params[1])
            elif self.decompo_list[8] == 4:
                self.fc0_0 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc0_1 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc0_2 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc0_3 = torch.nn.Linear(int(params[0]/4), params[1])
            else:
                self.fc0 = torch.nn.Linear(params[0], params[1])
        else:
            self.fc0 = torch.nn.Linear(params[0], params[1])


        if self.deepen_list != None:
            if self.deepen_list[8] == 1:
                self.fc0_dp = torch.nn.Linear(params[1], params[1])
        if self.skipcon_list != None:
            if self.skipcon_list[8] == 1:
                self.fc0_sk = torch.nn.Linear(params[1], params[1])


        params = [512, 512]
        if self.widen_list != None:
            if self.widen_list[8] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[8] / 4) * 4)
            if self.widen_list[9] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[9] / 4) * 4)
        if self.decompo_list != None:
            if self.decompo_list[9] == 1:
                self.fc1_0 = torch.nn.Linear(params[0], int(params[1]/2))
                self.fc1_1 = torch.nn.Linear(params[0], int(params[1]/2))
            elif self.decompo_list[9] == 2:
                self.fc1_0 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc1_1 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc1_2 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc1_3 = torch.nn.Linear(params[0], int(params[1]/4))
            elif self.decompo_list[9] == 3:
                self.fc1_0 = torch.nn.Linear(int(params[0]/2), params[1])
                self.fc1_1 = torch.nn.Linear(int(params[0]/2), params[1])
            elif self.decompo_list[9] == 4:
                self.fc1_0 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc1_1 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc1_2 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc1_3 = torch.nn.Linear(int(params[0]/4), params[1])
            else:
                self.fc1 = torch.nn.Linear(params[0], params[1])
        else:
            self.fc1 = torch.nn.Linear(params[0], params[1])


        if self.deepen_list != None:
            if self.deepen_list[9] == 1:
                self.fc1_dp = torch.nn.Linear(params[1], params[1])
        if self.skipcon_list != None:
            if self.skipcon_list[9] == 1:
                self.fc1_sk = torch.nn.Linear(params[1], params[1])


        params = [512, 10]
        if self.widen_list != None:
            if self.widen_list[9] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[9] / 4) * 4)
            pass
        if self.decompo_list != None:
            if self.decompo_list[10] == 1:
                self.classifier_0 = torch.nn.Linear(params[0], int(params[1]/2))
                self.classifier_1 = torch.nn.Linear(params[0], int(params[1]/2))
            elif self.decompo_list[10] == 3:
                self.classifier_0 = torch.nn.Linear(int(params[0]/2), params[1])
                self.classifier_1 = torch.nn.Linear(int(params[0]/2), params[1])
            elif self.decompo_list[10] == 4:
                self.classifier_0 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_1 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_2 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_3 = torch.nn.Linear(int(params[0]/4), params[1])
            else:
                self.classifier = torch.nn.Linear(params[0], params[1])
        else:
            self.classifier = torch.nn.Linear(params[0], params[1])


        if self.deepen_list != None:
            if self.deepen_list[10] == 1:
                self.classifier_dp = torch.nn.Linear(params[1], params[1])
        if self.skipcon_list != None:
            if self.skipcon_list[10] == 1:
                self.classifier_sk = torch.nn.Linear(params[1], params[1])

        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 32, 32)

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

        X1 = self.maxpool2x2(X1)

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

        X1 = self.maxpool2x2(X1)

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

        X1 = self.maxpool2x2(X1)

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


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[5] == 1:
                X1 = self.conv5_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[5] == 1:
                X1_skip = X1
                X1 = self.conv5_sk(X1)
                X1 = self.relu(X1 + X1_skip)

        X1 = self.maxpool2x2(X1)

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

        X1 = self.maxpool2x2(X1)
        X1 = X1.view(-1, 512)

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[8] == 1:
                X1_0 = self.fc0_0(X1)
                X1_1 = self.fc0_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[8] == 2:
                X1_0 = self.fc0_0(X1)
                X1_1 = self.fc0_1(X1)
                X1_2 = self.fc0_2(X1)
                X1_3 = self.fc0_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[8] == 3:
                X1_0 = self.fc0_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])
                X1_1 = self.fc0_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[8] == 4:
                X1_0 = self.fc0_0(X1[:, :int(torch.floor_divide(X1_shape[1],4))])
                X1_1 = self.fc0_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2))])
                X1_2 = self.fc0_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4))])
                X1_3 = self.fc0_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.fc0(X1)
        else:
             X1 = self.fc0(X1)

        if self.dummy_list != None:
            if self.dummy_list[8] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[8]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[8] == 1:
                X1 = self.fc0_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[8] == 1:
                X1_skip = X1
                X1 = self.fc0_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[9] == 1:
                X1_0 = self.fc1_0(X1)
                X1_1 = self.fc1_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[9] == 2:
                X1_0 = self.fc1_0(X1)
                X1_1 = self.fc1_1(X1)
                X1_2 = self.fc1_2(X1)
                X1_3 = self.fc1_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[9] == 3:
                X1_0 = self.fc1_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])
                X1_1 = self.fc1_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[9] == 4:
                X1_0 = self.fc1_0(X1[:, :int(torch.floor_divide(X1_shape[1],4))])
                X1_1 = self.fc1_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2))])
                X1_2 = self.fc1_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4))])
                X1_3 = self.fc1_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.fc1(X1)
        else:
             X1 = self.fc1(X1)

        if self.dummy_list != None:
            if self.dummy_list[9] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[9]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))


        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[9] == 1:
                X1 = self.fc1_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[9] == 1:
                X1_skip = X1
                X1 = self.fc1_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[10] == 1:
                X1_0 = self.classifier_0(X1)
                X1_1 = self.classifier_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[10] == 3:
                X1_0 = self.classifier_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])
                X1_1 = self.classifier_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[10] == 4:
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
            if self.dummy_list[10] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[10]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.logsoftmax(X1)
        return X1



batch_size = 1
input_features = 3072
torch.manual_seed(1234)
X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_4(input_features).to(cuda_device)

# End Call Model
model.eval()
new_out = model(X)