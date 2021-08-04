'''Tutorial on scripting a model'''

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU



#Step 1, copy this template and start modify things
#Step 2, rename this file to model_xx.py, xx is a number >=0, for example xx = 5.


class custom_cnn_4(torch.nn.Module): #Step 3, modify the class name to xx - 1, if model_5.py the class name should be custom_cnn_4.
    def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):
        super(custom_cnn_4,self).__init__() #Step 3, modify the class name accordingly
        self.reshape = reshape
        self.widen_list = widen_list
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.deepen_list = deepen_list
        self.skipcon_list = skipcon_list
        self.kerneladd_list = kerneladd_list
        ### #Step 4, start to scripting all the nn modules you need. 
        ### for nn.Conv2d and nn.Linear module the number has to be sequential starting from 0.
        self.relu = torch.nn.ReLU(inplace=True)
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0 = torch.nn.Conv2d(3, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv1 = torch.nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = torch.nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5 = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7 = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.fc0 = torch.nn.Linear(512, 512)
        self.fc1 = torch.nn.Linear(512, 512)
        self.classifier = torch.nn.Linear(512, 10)
        ###
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 32, 32) #Step 5, make sure the input is reshaped correctly
        ### #Step 6, start to scripting down the computation graph (nn module level). This shows a sequential computation graph, for skip-connection, see example below:
        
        X1 = self.conv0(X1)
        X1 = self.relu(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.conv1(X1)
        X1 = self.relu(X1)
        X1 = self.maxpool2x2(X1)
        
        X0_skip = X1 ### Skip connection example, save the intermediate activation to X0_skip (can be replaced by any variable name)
        X1 = self.conv2(X1)
        X1 = self.relu(X1)
        X1 = self.conv3(X1)
        X1 = self.relu(X1)
        X1 = self.conv4(X1)
        X1 = X1 + X0_skip ### Adds the skipped part

        X1 = self.relu(X1)
        X1 = self.conv5(X1)
        X1 = self.relu(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.conv6(X1)
        X1 = self.relu(X1)
        X1 = self.conv7(X1)
        X1 = self.relu(X1)
        X1 = self.maxpool2x2(X1)
        X1 = X1.view(-1, 512)
        X1 = self.fc0(X1)
        X1 = self.relu(X1)
        X1 = self.fc1(X1)
        X1 = self.relu(X1)
        X1 = self.classifier(X1)
        X1 = self.logsoftmax(X1)
        return X1

batch_size = 1
input_features = 3072 #Step 7: modify the number of features (3072 = 3 x 32x 32, of CIFAR-10 image shape)
torch.manual_seed(1234)
X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_4(input_features).to(cuda_device) #Step 8: modify the class name accordingly

# End Call Model
model.eval()
new_out = model(X)

### #Step 9: Directly run this script can check for basic error