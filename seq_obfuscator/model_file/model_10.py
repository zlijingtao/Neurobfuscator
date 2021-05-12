import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

assert torch.cuda.is_available()
cuda_device = torch.device("cuda")  # device object representing GPU

#This model stands for resnet-20 ImageNet
#model_id = 9 (regardless of the file name)

class custom_cnn_9(torch.nn.Module):
    def __init__(self, input_features, reshape = True, widen_list = None, decompo_list = None, dummy_list = None, deepen_list = None, skipcon_list = None, kerneladd_list = None):
        super(custom_cnn_9,self).__init__()
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
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.conv0 = torch.nn.Conv2d(3, 64, (7, 7), stride=(2, 2), padding=(3, 3))
        self.conv_bn0 = torch.nn.BatchNorm2d(64)

        self.conv1 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn3 = torch.nn.BatchNorm2d(64)
        self.conv4 = torch.nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn4 = torch.nn.BatchNorm2d(64)


        self.conv5 = torch.nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_bn5 = torch.nn.BatchNorm2d(128)
        self.conv6 = torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn6 = torch.nn.BatchNorm2d(128)
        self.conv7 = torch.nn.Conv2d(64, 128, (1, 1), stride=(2, 2), padding=(0, 0))
        self.conv_bn7 = torch.nn.BatchNorm2d(128)
        self.conv8 = torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn8 = torch.nn.BatchNorm2d(128)
        self.conv9 = torch.nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn9 = torch.nn.BatchNorm2d(128)

        self.conv10 = torch.nn.Conv2d(128, 256, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_bn10 = torch.nn.BatchNorm2d(256)
        self.conv11 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn11 = torch.nn.BatchNorm2d(256)
        self.conv12 = torch.nn.Conv2d(128, 256, (1, 1), stride=(2, 2), padding=(0, 0))
        self.conv_bn12 = torch.nn.BatchNorm2d(256)
        self.conv13 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn13 = torch.nn.BatchNorm2d(256)
        self.conv14 = torch.nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn14 = torch.nn.BatchNorm2d(256)

        self.conv15 = torch.nn.Conv2d(256, 512, (3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_bn15 = torch.nn.BatchNorm2d(512)
        self.conv16 = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn16 = torch.nn.BatchNorm2d(512)
        self.conv17 = torch.nn.Conv2d(256, 512, (1, 1), stride=(2, 2), padding=(0, 0))
        self.conv_bn17 = torch.nn.BatchNorm2d(512)
        self.conv18 = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn18 = torch.nn.BatchNorm2d(512)
        self.conv19 = torch.nn.Conv2d(512, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn19 = torch.nn.BatchNorm2d(512)

        self.classifier = torch.nn.Linear(512, 1000)
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 224, 224)
        X1 = self.conv0(X1)
        X1 = self.conv_bn0(X1)
        X1 = self.relu(X1)
        X1 = self.maxpool2x2(X1)

        X0_skip = X1
        X1 = self.conv1(X1)
        X1 = self.conv_bn1(X1)
        X1 = self.relu(X1)
        X1 = self.conv2(X1)
        X1 = self.conv_bn2(X1)
        X1 += X0_skip
        X1 = self.relu(X1)

        X0_skip = X1
        X1 = self.conv3(X1)
        X1 = self.conv_bn3(X1)
        X1 = self.relu(X1)
        X1 = self.conv4(X1)
        X1 = self.conv_bn4(X1)
        X1 += X0_skip
        X1 = self.relu(X1)

        X0_skip = X1
        X1 = self.conv5(X1)
        X1 = self.conv_bn5(X1)
        X1 = self.relu(X1)
        X1 = self.conv6(X1)
        X1 = self.conv_bn6(X1)
        X1_saved = X1
        X1 = X0_skip
        X1 = self.conv7(X1)
        X1 = self.conv_bn7(X1)
        X1 += X1_saved
        X1 = self.relu(X1)

        X0_skip = X1
        X1 = self.conv8(X1)
        X1 = self.conv_bn8(X1)
        X1 = self.relu(X1)
        X1 = self.conv9(X1)
        X1 = self.conv_bn9(X1)
        X1 += X0_skip
        X1 = self.relu(X1)

        X0_skip = X1
        X1 = self.conv10(X1)
        X1 = self.conv_bn10(X1)
        X1 = self.relu(X1)
        X1 = self.conv11(X1)
        X1 = self.conv_bn11(X1)
        X1_saved = X1
        X1 = X0_skip
        X1 = self.conv12(X1)
        X1 = self.conv_bn12(X1)
        X1 += X1_saved
        X1 = self.relu(X1)

        X0_skip = X1
        X1 = self.conv13(X1)
        X1 = self.conv_bn13(X1)
        X1 = self.relu(X1)
        X1 = self.conv14(X1)
        X1 = self.conv_bn14(X1)
        X1 += X0_skip
        X1 = self.relu(X1)

        X0_skip = X1
        X1 = self.conv15(X1)
        X1 = self.conv_bn15(X1)
        X1 = self.relu(X1)
        X1 = self.conv16(X1)
        X1 = self.conv_bn16(X1)
        X1_saved = X1
        X1 = X0_skip
        X1 = self.conv17(X1)
        X1 = self.conv_bn17(X1)
        X1 += X1_saved
        X1 = self.relu(X1)

        X0_skip = X1
        X1 = self.conv18(X1)
        X1 = self.conv_bn18(X1)
        X1 = self.relu(X1)
        X1 = self.conv19(X1)
        X1 = self.conv_bn19(X1)
        X1 += X0_skip
        X1 = self.relu(X1)

        X1 = self.avgpool(X1)
        X1 = X1.view(-1, 512)
        X1 = self.classifier(X1)
        X1 = self.logsoftmax(X1)
        return X1



batch_size = 1
input_features = 150528
torch.manual_seed(1234)
X = torch.randn(batch_size, input_features, device=cuda_device)

# Start Call Model

model = custom_cnn_9(input_features).to(cuda_device)

# End Call Model
model.eval()
new_out = model(X)