import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# (ImageNet format) A simple two-layer CNN to test obfuscating convolution dimension parameters
class custom_cnn_1(torch.nn.Module):
    def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):
        super(custom_cnn_1,self).__init__()
        self.reshape = reshape
        self.widen_list = widen_list
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.deepen_list = deepen_list
        self.skipcon_list = skipcon_list
        self.kerneladd_list = kerneladd_list
        self.relu = torch.nn.ReLU(inplace=True)

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


        params = [64, 256, 3, 3, 1, 1, 1, 1]
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


        self.conv_bn1 = torch.nn.BatchNorm2d(params[1])


        params = [256, 64, 3, 3, 1, 1, 1, 1]
        if self.widen_list != None:
            if self.widen_list[1] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[1] / 4) * 4)
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

        return X1

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

batch_size = 1
input_features = 150528

X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_1(input_features).to(cuda_device)

# End Call Model
model.eval()
new_out = model(X)
# print(new_out.size())