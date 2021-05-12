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

        
        self.conv0 = torch.nn.Conv2d(3, 32, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv_bn0 = torch.nn.BatchNorm2d(32)


        # [1, 16, 1, 1]

        self.conv1 = torch.nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
        self.conv_bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 16, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn2 = torch.nn.BatchNorm2d(16)

        # [6, 24, 2, 2]

        self.conv3 = torch.nn.Conv2d(16, 96, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn3 = torch.nn.BatchNorm2d(96)

        self.conv4 = torch.nn.Conv2d(96, 96, (3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
        self.conv_bn4 = torch.nn.BatchNorm2d(96)

        self.conv5 = torch.nn.Conv2d(96, 24, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn5 = torch.nn.BatchNorm2d(24)

        self.conv6 = torch.nn.Conv2d(24, 144, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn6 = torch.nn.BatchNorm2d(144)

        self.conv7 = torch.nn.Conv2d(144, 144, (3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
        self.conv_bn7 = torch.nn.BatchNorm2d(144)

        self.conv8 = torch.nn.Conv2d(144, 24, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn8 = torch.nn.BatchNorm2d(24)

        # [6, 32, 3, 2],

        self.conv9 = torch.nn.Conv2d(24, 144, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn9 = torch.nn.BatchNorm2d(144)

        self.conv10 = torch.nn.Conv2d(144, 144, (3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
        self.conv_bn10 = torch.nn.BatchNorm2d(144)

        self.conv11 = torch.nn.Conv2d(144, 32, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn11 = torch.nn.BatchNorm2d(32)

        self.conv12 = torch.nn.Conv2d(32, 192, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn12 = torch.nn.BatchNorm2d(192)

        self.conv13 = torch.nn.Conv2d(192, 192, (3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.conv_bn13 = torch.nn.BatchNorm2d(192)

        self.conv14 = torch.nn.Conv2d(192, 32, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn14 = torch.nn.BatchNorm2d(32)

        self.conv15 = torch.nn.Conv2d(32, 192, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn15 = torch.nn.BatchNorm2d(192)

        self.conv16 = torch.nn.Conv2d(192, 192, (3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
        self.conv_bn16 = torch.nn.BatchNorm2d(192)

        self.conv17 = torch.nn.Conv2d(192, 32, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn17 = torch.nn.BatchNorm2d(32)

        # [6, 64, 4, 2],

        self.conv18 = torch.nn.Conv2d(32, 192, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn18 = torch.nn.BatchNorm2d(192)

        self.conv19 = torch.nn.Conv2d(192, 192, (3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
        self.conv_bn19 = torch.nn.BatchNorm2d(192)

        self.conv20 = torch.nn.Conv2d(192, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn20 = torch.nn.BatchNorm2d(64)

        self.conv21 = torch.nn.Conv2d(64, 384, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn21 = torch.nn.BatchNorm2d(384)

        self.conv22 = torch.nn.Conv2d(384, 384, (3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.conv_bn22 = torch.nn.BatchNorm2d(384)

        self.conv23 = torch.nn.Conv2d(384, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn23 = torch.nn.BatchNorm2d(64)

        self.conv24 = torch.nn.Conv2d(64, 384, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn24 = torch.nn.BatchNorm2d(384)

        self.conv25 = torch.nn.Conv2d(384, 384, (3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.conv_bn25 = torch.nn.BatchNorm2d(384)

        self.conv26 = torch.nn.Conv2d(384, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn26 = torch.nn.BatchNorm2d(64)

        self.conv27 = torch.nn.Conv2d(64, 384, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn27 = torch.nn.BatchNorm2d(384)

        self.conv28 = torch.nn.Conv2d(384, 384, (3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.conv_bn28 = torch.nn.BatchNorm2d(384)

        self.conv29 = torch.nn.Conv2d(384, 64, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn29 = torch.nn.BatchNorm2d(64) 


        # [6, 96, 3, 1],

        self.conv30 = torch.nn.Conv2d(64, 384, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn30 = torch.nn.BatchNorm2d(384)

        self.conv31 = torch.nn.Conv2d(384, 384, (3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
        self.conv_bn31 = torch.nn.BatchNorm2d(384)

        self.conv32 = torch.nn.Conv2d(384, 96, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn32 = torch.nn.BatchNorm2d(96)

        self.conv33 = torch.nn.Conv2d(96, 576, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn33 = torch.nn.BatchNorm2d(576)

        self.conv34 = torch.nn.Conv2d(576, 576, (3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.conv_bn34 = torch.nn.BatchNorm2d(576)

        self.conv35 = torch.nn.Conv2d(576, 96, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn35 = torch.nn.BatchNorm2d(96)

        self.conv36 = torch.nn.Conv2d(96, 576, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn36 = torch.nn.BatchNorm2d(576)

        self.conv37 = torch.nn.Conv2d(576, 576, (3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
        self.conv_bn37 = torch.nn.BatchNorm2d(576)

        self.conv38 = torch.nn.Conv2d(576, 96, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn38 = torch.nn.BatchNorm2d(96)

        # [6, 160, 3, 2],

        self.conv39 = torch.nn.Conv2d(96, 576, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn39 = torch.nn.BatchNorm2d(576)

        self.conv40 = torch.nn.Conv2d(576, 576, (3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
        self.conv_bn40 = torch.nn.BatchNorm2d(576)

        self.conv41 = torch.nn.Conv2d(576, 160, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn41 = torch.nn.BatchNorm2d(160)

        self.conv42 = torch.nn.Conv2d(160, 960, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn42 = torch.nn.BatchNorm2d(960)

        self.conv43 = torch.nn.Conv2d(960, 960, (3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.conv_bn43 = torch.nn.BatchNorm2d(960)

        self.conv44 = torch.nn.Conv2d(960, 160, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn44 = torch.nn.BatchNorm2d(160)

        self.conv45 = torch.nn.Conv2d(160, 960, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn45 = torch.nn.BatchNorm2d(960)

        self.conv46 = torch.nn.Conv2d(960, 960, (3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.conv_bn46 = torch.nn.BatchNorm2d(960)

        self.conv47 = torch.nn.Conv2d(960, 160, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn47 = torch.nn.BatchNorm2d(160)

        # [6, 320, 1, 1],

        self.conv48 = torch.nn.Conv2d(160, 960, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn48 = torch.nn.BatchNorm2d(960)

        self.conv49 = torch.nn.Conv2d(960, 960, (3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
        self.conv_bn49 = torch.nn.BatchNorm2d(960)

        self.conv50 = torch.nn.Conv2d(960, 320, (1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.conv_bn50 = torch.nn.BatchNorm2d(320)

        self.conv51 = torch.nn.Conv2d(320, 1280, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn51 = torch.nn.BatchNorm2d(1280)

        self.classifier = torch.nn.Linear(1280, 1000)
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
        
        X1 = self.conv1(X1)
        X1 = self.conv_bn1(X1)
        X1 = self.relu(X1)
        X1 = self.conv2(X1)
        X1 = self.conv_bn2(X1)
        
        # [6, 24, 2, 2]

        X1 = self.conv3(X1)
        X1 = self.conv_bn3(X1)
        X1 = self.relu(X1)
        X1 = self.conv4(X1)
        X1 = self.conv_bn4(X1)
        X1 = self.relu(X1)
        X1 = self.conv5(X1)
        X1 = self.conv_bn5(X1)

        X0_skip = X1
        X1 = self.conv6(X1)
        X1 = self.conv_bn6(X1)
        X1 = self.relu(X1)
        X1 = self.conv7(X1)
        X1 = self.conv_bn7(X1)
        X1 = self.relu(X1)
        X1 = self.conv8(X1)
        X1 = self.conv_bn8(X1)
        X1 = X1 + X0_skip

        # [6, 32, 3, 2],

        X1 = self.conv9(X1)
        X1 = self.conv_bn9(X1)
        X1 = self.relu(X1)
        X1 = self.conv10(X1)
        X1 = self.conv_bn10(X1)
        X1 = self.relu(X1)
        X1 = self.conv11(X1)
        X1 = self.conv_bn11(X1)

        X0_skip = X1
        X1 = self.conv12(X1)
        X1 = self.conv_bn12(X1)
        X1 = self.relu(X1)
        X1 = self.conv13(X1)
        X1 = self.conv_bn13(X1)
        X1 = self.relu(X1)
        X1 = self.conv14(X1)
        X1 = self.conv_bn14(X1)
        X1 = X1 + X0_skip

        X0_skip = X1
        X1 = self.conv15(X1)
        X1 = self.conv_bn15(X1)
        X1 = self.relu(X1)
        X1 = self.conv16(X1)
        X1 = self.conv_bn16(X1)
        X1 = self.relu(X1)
        X1 = self.conv17(X1)
        X1 = self.conv_bn17(X1)
        X1 = X1 + X0_skip

        # [6, 64, 4, 2],

        X1 = self.conv18(X1)
        X1 = self.conv_bn18(X1)
        X1 = self.relu(X1)
        X1 = self.conv19(X1)
        X1 = self.conv_bn19(X1)
        X1 = self.relu(X1)
        X1 = self.conv20(X1)
        X1 = self.conv_bn20(X1)

        X0_skip = X1
        X1 = self.conv21(X1)
        X1 = self.conv_bn21(X1)
        X1 = self.relu(X1)
        X1 = self.conv22(X1)
        X1 = self.conv_bn22(X1)
        X1 = self.relu(X1)
        X1 = self.conv23(X1)
        X1 = self.conv_bn23(X1)
        X1 = X1 + X0_skip

        X0_skip = X1
        X1 = self.conv24(X1)
        X1 = self.conv_bn24(X1)
        X1 = self.relu(X1)
        X1 = self.conv25(X1)
        X1 = self.conv_bn25(X1)
        X1 = self.relu(X1)
        X1 = self.conv26(X1)
        X1 = self.conv_bn26(X1)
        X1 = X1 + X0_skip

        X0_skip = X1
        X1 = self.conv27(X1)
        X1 = self.conv_bn27(X1)
        X1 = self.relu(X1)
        X1 = self.conv28(X1)
        X1 = self.conv_bn28(X1)
        X1 = self.relu(X1)
        X1 = self.conv29(X1)
        X1 = self.conv_bn29(X1)
        X1 = X1 + X0_skip

        # [6, 96, 3, 1],

        X1 = self.conv30(X1)
        X1 = self.conv_bn30(X1)
        X1 = self.relu(X1)
        X1 = self.conv31(X1)
        X1 = self.conv_bn31(X1)
        X1 = self.relu(X1)
        X1 = self.conv32(X1)
        X1 = self.conv_bn32(X1)

        X0_skip = X1
        X1 = self.conv33(X1)
        X1 = self.conv_bn33(X1)
        X1 = self.relu(X1)
        X1 = self.conv34(X1)
        X1 = self.conv_bn34(X1)
        X1 = self.relu(X1)
        X1 = self.conv35(X1)
        X1 = self.conv_bn35(X1)
        X1 = X1 + X0_skip

        X0_skip = X1
        X1 = self.conv36(X1)
        X1 = self.conv_bn36(X1)
        X1 = self.relu(X1)
        X1 = self.conv37(X1)
        X1 = self.conv_bn37(X1)
        X1 = self.relu(X1)
        X1 = self.conv38(X1)
        X1 = self.conv_bn38(X1)
        X1 = X1 + X0_skip

        # [6, 160, 3, 2],

        X1 = self.conv39(X1)
        X1 = self.conv_bn39(X1)
        X1 = self.relu(X1)
        X1 = self.conv40(X1)
        X1 = self.conv_bn40(X1)
        X1 = self.relu(X1)
        X1 = self.conv41(X1)
        X1 = self.conv_bn41(X1)

        X0_skip = X1
        X1 = self.conv42(X1)
        X1 = self.conv_bn42(X1)
        X1 = self.relu(X1)
        X1 = self.conv43(X1)
        X1 = self.conv_bn43(X1)
        X1 = self.relu(X1)
        X1 = self.conv44(X1)
        X1 = self.conv_bn44(X1)
        X1 = X1 + X0_skip

        X0_skip = X1
        X1 = self.conv45(X1)
        X1 = self.conv_bn45(X1)
        X1 = self.relu(X1)
        X1 = self.conv46(X1)
        X1 = self.conv_bn46(X1)
        X1 = self.relu(X1)
        X1 = self.conv47(X1)
        X1 = self.conv_bn47(X1)
        X1 = X1 + X0_skip

        # [6, 320, 1, 1],

        X1 = self.conv48(X1)
        X1 = self.conv_bn48(X1)
        X1 = self.relu(X1)
        X1 = self.conv49(X1)
        X1 = self.conv_bn49(X1)
        X1 = self.relu(X1)
        X1 = self.conv50(X1)
        X1 = self.conv_bn50(X1)

        X1 = self.conv51(X1)
        X1 = self.conv_bn51(X1)
        X1 = self.relu(X1)

        X1 = nn.functional.adaptive_avg_pool2d(X1, 1)
        X1 = X1.view(-1, 1280)
        X1 = self.classifier(X1)
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