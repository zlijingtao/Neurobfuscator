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

        params = [3072, 512]
        if self.widen_list != None:
            if self.widen_list[0] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[0] / 4) * 4)
        if self.decompo_list != None:
            if self.decompo_list[0] == 1:
                self.fc0_0 = torch.nn.Linear(params[0], int(params[1]/2))
                self.fc0_1 = torch.nn.Linear(params[0], int(params[1]/2))
            elif self.decompo_list[0] == 2:
                self.fc0_0 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc0_1 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc0_2 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc0_3 = torch.nn.Linear(params[0], int(params[1]/4))
            elif self.decompo_list[0] == 3:
                self.fc0_0 = torch.nn.Linear(int(params[0]/2), params[1])
                self.fc0_1 = torch.nn.Linear(int(params[0]/2), params[1])
            elif self.decompo_list[0] == 4:
                self.fc0_0 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc0_1 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc0_2 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc0_3 = torch.nn.Linear(int(params[0]/4), params[1])
            else:
                self.fc0 = torch.nn.Linear(params[0], params[1])
        else:
            self.fc0 = torch.nn.Linear(params[0], params[1])


        if self.deepen_list != None:
            if self.deepen_list[0] == 1:
                self.fc0_dp = torch.nn.Linear(params[1], params[1])
        if self.skipcon_list != None:
            if self.skipcon_list[0] == 1:
                self.fc0_sk = torch.nn.Linear(params[1], params[1])


        self.fc_bn0 = torch.nn.BatchNorm1d(params[1])


        params = [512, 256]
        if self.widen_list != None:
            if self.widen_list[0] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[0] / 4) * 4)
            if self.widen_list[1] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[1] / 4) * 4)
        if self.decompo_list != None:
            if self.decompo_list[1] == 1:
                self.fc1_0 = torch.nn.Linear(params[0], int(params[1]/2))
                self.fc1_1 = torch.nn.Linear(params[0], int(params[1]/2))
            elif self.decompo_list[1] == 2:
                self.fc1_0 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc1_1 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc1_2 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc1_3 = torch.nn.Linear(params[0], int(params[1]/4))
            elif self.decompo_list[1] == 3:
                self.fc1_0 = torch.nn.Linear(int(params[0]/2), params[1])
                self.fc1_1 = torch.nn.Linear(int(params[0]/2), params[1])
            elif self.decompo_list[1] == 4:
                self.fc1_0 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc1_1 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc1_2 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc1_3 = torch.nn.Linear(int(params[0]/4), params[1])
            else:
                self.fc1 = torch.nn.Linear(params[0], params[1])
        else:
            self.fc1 = torch.nn.Linear(params[0], params[1])


        if self.deepen_list != None:
            if self.deepen_list[1] == 1:
                self.fc1_dp = torch.nn.Linear(params[1], params[1])
        if self.skipcon_list != None:
            if self.skipcon_list[1] == 1:
                self.fc1_sk = torch.nn.Linear(params[1], params[1])


        self.fc_bn1 = torch.nn.BatchNorm1d(params[1])


        params = [256, 128]
        if self.widen_list != None:
            if self.widen_list[1] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[1] / 4) * 4)
            if self.widen_list[2] > 1.0:
                params[1] = int(np.floor(params[1] * self.widen_list[2] / 4) * 4)
        if self.decompo_list != None:
            if self.decompo_list[2] == 1:
                self.fc2_0 = torch.nn.Linear(params[0], int(params[1]/2))
                self.fc2_1 = torch.nn.Linear(params[0], int(params[1]/2))
            elif self.decompo_list[2] == 2:
                self.fc2_0 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc2_1 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc2_2 = torch.nn.Linear(params[0], int(params[1]/4))
                self.fc2_3 = torch.nn.Linear(params[0], int(params[1]/4))
            elif self.decompo_list[2] == 3:
                self.fc2_0 = torch.nn.Linear(int(params[0]/2), params[1])
                self.fc2_1 = torch.nn.Linear(int(params[0]/2), params[1])
            elif self.decompo_list[2] == 4:
                self.fc2_0 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc2_1 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc2_2 = torch.nn.Linear(int(params[0]/4), params[1])
                self.fc2_3 = torch.nn.Linear(int(params[0]/4), params[1])
            else:
                self.fc2 = torch.nn.Linear(params[0], params[1])
        else:
            self.fc2 = torch.nn.Linear(params[0], params[1])


        if self.deepen_list != None:
            if self.deepen_list[2] == 1:
                self.fc2_dp = torch.nn.Linear(params[1], params[1])
        if self.skipcon_list != None:
            if self.skipcon_list[2] == 1:
                self.fc2_sk = torch.nn.Linear(params[1], params[1])


        self.fc_bn2 = torch.nn.BatchNorm1d(params[1])


        params = [128, 10]
        if self.widen_list != None:
            if self.widen_list[2] > 1.0:
                params[0] = int(np.floor(params[0] * self.widen_list[2] / 4) * 4)
            pass
        if self.decompo_list != None:
            if self.decompo_list[3] == 1:
                self.classifier_0 = torch.nn.Linear(params[0], int(params[1]/2))
                self.classifier_1 = torch.nn.Linear(params[0], int(params[1]/2))
            elif self.decompo_list[3] == 3:
                self.classifier_0 = torch.nn.Linear(int(params[0]/2), params[1])
                self.classifier_1 = torch.nn.Linear(int(params[0]/2), params[1])
            elif self.decompo_list[3] == 4:
                self.classifier_0 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_1 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_2 = torch.nn.Linear(int(params[0]/4), params[1])
                self.classifier_3 = torch.nn.Linear(int(params[0]/4), params[1])
            else:
                self.classifier = torch.nn.Linear(params[0], params[1])
        else:
            self.classifier = torch.nn.Linear(params[0], params[1])


        if self.deepen_list != None:
            if self.deepen_list[3] == 1:
                self.classifier_dp = torch.nn.Linear(params[1], params[1])
        if self.skipcon_list != None:
            if self.skipcon_list[3] == 1:
                self.classifier_sk = torch.nn.Linear(params[1], params[1])


        self.classifier_bn = torch.nn.BatchNorm1d(params[1], affine=False)

        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):

        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[0] == 1:
                X1_0 = self.fc0_0(X1)
                X1_1 = self.fc0_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[0] == 2:
                X1_0 = self.fc0_0(X1)
                X1_1 = self.fc0_1(X1)
                X1_2 = self.fc0_2(X1)
                X1_3 = self.fc0_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[0] == 3:
                X1_0 = self.fc0_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])
                X1_1 = self.fc0_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[0] == 4:
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
            if self.dummy_list[0] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[0]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.fc_bn0(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[0] == 1:
                X1 = self.fc0_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[0] == 1:
                X1_skip = X1
                X1 = self.fc0_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[1] == 1:
                X1_0 = self.fc1_0(X1)
                X1_1 = self.fc1_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[1] == 2:
                X1_0 = self.fc1_0(X1)
                X1_1 = self.fc1_1(X1)
                X1_2 = self.fc1_2(X1)
                X1_3 = self.fc1_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[1] == 3:
                X1_0 = self.fc1_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])
                X1_1 = self.fc1_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[1] == 4:
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
            if self.dummy_list[1] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[1]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.fc_bn1(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[1] == 1:
                X1 = self.fc1_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[1] == 1:
                X1_skip = X1
                X1 = self.fc1_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[2] == 1:
                X1_0 = self.fc2_0(X1)
                X1_1 = self.fc2_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[2] == 2:
                X1_0 = self.fc2_0(X1)
                X1_1 = self.fc2_1(X1)
                X1_2 = self.fc2_2(X1)
                X1_3 = self.fc2_3(X1)
                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)
            elif self.decompo_list[2] == 3:
                X1_0 = self.fc2_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])
                X1_1 = self.fc2_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[2] == 4:
                X1_0 = self.fc2_0(X1[:, :int(torch.floor_divide(X1_shape[1],4))])
                X1_1 = self.fc2_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2))])
                X1_2 = self.fc2_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4))])
                X1_3 = self.fc2_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):])
                X1 = X1_0 + X1_1 + X1_2 + X1_3
            else:
                X1 = self.fc2(X1)
        else:
             X1 = self.fc2(X1)

        if self.dummy_list != None:
            if self.dummy_list[2] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[2]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

        X1 = self.fc_bn2(X1)

        if True:
            X1 = self.relu(X1)
        if self.deepen_list != None:
            if self.deepen_list[2] == 1:
                X1 = self.fc2_dp(X1)
                X1 = self.relu(X1)
        if self.skipcon_list != None:
            if self.skipcon_list[2] == 1:
                X1_skip = X1
                X1 = self.fc2_sk(X1)
                X1 = self.relu(X1 + X1_skip)


        X1_shape = X1.size()
        if self.decompo_list != None:
            if self.decompo_list[3] == 1:
                X1_0 = self.classifier_0(X1)
                X1_1 = self.classifier_1(X1)
                X1 = torch.cat((X1_0, X1_1), 1)
            elif self.decompo_list[3] == 3:
                X1_0 = self.classifier_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])
                X1_1 = self.classifier_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])
                X1 = torch.add(X1_0, X1_1)
            elif self.decompo_list[3] == 4:
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
            if self.dummy_list[3] > 0:
                dummy = np.zeros(X1.size()).astype("float32")
                for i in range(self.dummy_list[3]):
                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))

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