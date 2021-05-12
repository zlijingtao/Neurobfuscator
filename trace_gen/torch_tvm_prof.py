import tvm
from tvm import relay
import logging
# logging.getLogger('autotvm').setLevel(logging.FATAL)
import numpy as np
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import tvm.contrib.graph_executor as runtime
from torch.jit import TracerWarning
import warnings
import shutil
import GPUtil
# warnings.filterwarnings("ignore", category=TracerWarning)

def prune_old_tasks(tasks, log_file):
    if os.path.isfile(log_file):
        new_tasks = []
        history = autotvm.record.ApplyHistoryBest(log_file)
        for task in tasks:
            # print(task.name)
            if history._query_inside(task.target, task.workload) is None:
                # if "dense_small_batch.cuda" in str(task.name) and "10" not in str(task.workload):
                if "dense_small_batch.cuda" in str(task.name):
                    continue
                if "depthwise_conv2d_nchw.cuda" in str(task.name) or "conv2d_nchw.cuda" in str(task.name) :
                    continue
                new_tasks.append(task)
        return new_tasks
    else:
        return tasks

def do_tune(tasks, old_tasks, log_filename, n_trial = 20, tuner = 'xgb'):
    tmp_log_file = log_filename + ".tmp"
    tuner = 'xgb'
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        # print(tasks.name)
        '''Template Path: tvm/python/tvm/topi/cuda/con2d_winograd.py and con2d_direct.py'''
        #search: Modify (3 places) in con2d_winograd.py

        # create tuner
        # ModelBasedTuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, feature_type='knob', loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)
        # tuner_obj = tvm.autotvm.tuner.XGBTuner(tsk, loss_type='rank')
        if os.path.isfile(tmp_log_file):
            tuner_obj.load_history(tvm.autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))

        tuner_obj.tune(
            n_trial=n_trial,
            early_stopping=600,
            measure_option=tvm.autotvm.measure_option(
                builder=tvm.autotvm.LocalBuilder(timeout=10),
                runner=tvm.autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150)),
            callbacks=[
                tvm.autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                tvm.autotvm.callback.log_to_file(tmp_log_file)
            ])

    if len(tasks) > 0:
        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)
    else:
        print("No tuning Task Found! Print Old Tasks")
        # for i, tsk in enumerate(reversed(old_tasks)):
            # print(tsk.name)s
            # print(tsk.config_space.space_map)

class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1):
    super(ResNetBasicblock, self).__init__()

    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.conv_bn1 = nn.BatchNorm2d(planes)

    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.conv_bn2 = nn.BatchNorm2d(planes)

    self.downsample = None

  def forward(self, x):
    residual = x
    basicblock = self.conv1(x)
    basicblock = self.conv_bn1(basicblock)
    basicblock = F.relu(basicblock, inplace=True)

    basicblock = self.conv2(basicblock)
    basicblock = self.conv_bn2(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    return residual + basicblock

class DepthwiseConv(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(DepthwiseConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride, padding=1, groups=inplanes, bias=False)
        self.conv_bn1 = nn.BatchNorm2d(inplanes)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv_bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv_bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.conv_bn2(out)
        return out

# Model starts here

class cnn_32p3p1p1p0_depth688p1_16p1p1p0p0_depth912p1_912p1p1p0p0_depth1264p1_272p1p1p0p0_depth736p1_240p1p1p0p0_depth1200p1_800p1p1p0p0_depth608p1_544p1p1p0p0_depth192p1_656p1p1p0p0_depth32p2_1184p1p1p0p0_depth1200p2_384p1p1p0p0_depth848p1_1104p1p1p0p0_depth1008p1_240p1p1p0p0_depth1232p2_912p1p1p0p0_depth1024p1_48p1p1p0p0_depth528p1_256p1p1p0p0_depth256p1_512p1p1p0p0_depth160p1_1104p1p1p0p0_depth96p1_656p1p1p0p0_depth416p1_128p1p1p0p0_depth784p1_128p1p1p0p0_depth1024p2_176p1p1p0p0_depth1200p2_960p1p1p0p0_depth1104p1_144p1p1p0p0_depth848p1_640p1p1p0p0_depth320p1_672p1p1p0p0_depth496p1_1120p1p1p0p0_depth176p1_384p1p1p0p0_depth784p1_816p1p1p0p0_depth528p1_320p1p1p0p0_depth736p1_576p1p1p0p0_depth752p1_352p1p1p0p0_depth1168p1_880p1p1p0p0_depth1152p1_48p1p1p0p0_depth688p1_32p1p1p0p0_depth1168p1_64p1p1p0p0_depth208p1_1104p1p1p0p0_bn1_mlp_bn0(torch.nn.Module):
    def __init__(self, input_features, reshape = True, decompo_list = None, dummy_list = None):
        super(cnn_32p3p1p1p0_depth688p1_16p1p1p0p0_depth912p1_912p1p1p0p0_depth1264p1_272p1p1p0p0_depth736p1_240p1p1p0p0_depth1200p1_800p1p1p0p0_depth608p1_544p1p1p0p0_depth192p1_656p1p1p0p0_depth32p2_1184p1p1p0p0_depth1200p2_384p1p1p0p0_depth848p1_1104p1p1p0p0_depth1008p1_240p1p1p0p0_depth1232p2_912p1p1p0p0_depth1024p1_48p1p1p0p0_depth528p1_256p1p1p0p0_depth256p1_512p1p1p0p0_depth160p1_1104p1p1p0p0_depth96p1_656p1p1p0p0_depth416p1_128p1p1p0p0_depth784p1_128p1p1p0p0_depth1024p2_176p1p1p0p0_depth1200p2_960p1p1p0p0_depth1104p1_144p1p1p0p0_depth848p1_640p1p1p0p0_depth320p1_672p1p1p0p0_depth496p1_1120p1p1p0p0_depth176p1_384p1p1p0p0_depth784p1_816p1p1p0p0_depth528p1_320p1p1p0p0_depth736p1_576p1p1p0p0_depth752p1_352p1p1p0p0_depth1168p1_880p1p1p0p0_depth1152p1_48p1p1p0p0_depth688p1_32p1p1p0p0_depth1168p1_64p1p1p0p0_depth208p1_1104p1p1p0p0_bn1_mlp_bn0,self).__init__()
        self.reshape = reshape
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.relu = torch.nn.ReLU(inplace=True)
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool2x2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv0 = torch.nn.Conv2d(3, 32, (3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv_bn0 = torch.nn.BatchNorm2d(32)
        self.conv1 = DepthwiseConv(32, 688, stride= 1)
        self.conv2 = torch.nn.Conv2d(688, 16, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = DepthwiseConv(16, 912, stride= 1)
        self.conv4 = torch.nn.Conv2d(912, 912, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn4 = torch.nn.BatchNorm2d(912)
        self.conv5 = DepthwiseConv(912, 1264, stride= 1)
        self.conv6 = torch.nn.Conv2d(1264, 272, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn6 = torch.nn.BatchNorm2d(272)
        self.conv7 = DepthwiseConv(272, 736, stride= 1)
        self.conv8 = torch.nn.Conv2d(736, 240, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn8 = torch.nn.BatchNorm2d(240)
        self.conv9 = DepthwiseConv(240, 1200, stride= 1)
        self.conv10 = torch.nn.Conv2d(1200, 800, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn10 = torch.nn.BatchNorm2d(800)
        self.conv11 = DepthwiseConv(800, 608, stride= 1)
        self.conv12 = torch.nn.Conv2d(608, 544, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn12 = torch.nn.BatchNorm2d(544)
        self.conv13 = DepthwiseConv(544, 192, stride= 1)
        self.conv14 = torch.nn.Conv2d(192, 656, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn14 = torch.nn.BatchNorm2d(656)
        self.conv15 = DepthwiseConv(656, 32, stride= 2)
        self.conv16 = torch.nn.Conv2d(32, 1184, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn16 = torch.nn.BatchNorm2d(1184)
        self.conv17 = DepthwiseConv(1184, 1200, stride= 2)
        self.conv18 = torch.nn.Conv2d(1200, 384, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn18 = torch.nn.BatchNorm2d(384)
        self.conv19 = DepthwiseConv(384, 848, stride= 1)
        self.conv20 = torch.nn.Conv2d(848, 1104, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn20 = torch.nn.BatchNorm2d(1104)
        self.conv21 = DepthwiseConv(1104, 1008, stride= 1)
        self.conv22 = torch.nn.Conv2d(1008, 240, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn22 = torch.nn.BatchNorm2d(240)
        self.conv23 = DepthwiseConv(240, 1232, stride= 2)
        self.conv24 = torch.nn.Conv2d(1232, 912, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn24 = torch.nn.BatchNorm2d(912)
        self.conv25 = DepthwiseConv(912, 1024, stride= 1)
        self.conv26 = torch.nn.Conv2d(1024, 48, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn26 = torch.nn.BatchNorm2d(48)
        self.conv27 = DepthwiseConv(48, 528, stride= 1)
        self.conv28 = torch.nn.Conv2d(528, 256, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn28 = torch.nn.BatchNorm2d(256)
        self.conv29 = DepthwiseConv(256, 256, stride= 1)
        self.conv30 = torch.nn.Conv2d(256, 512, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn30 = torch.nn.BatchNorm2d(512)
        self.conv31 = DepthwiseConv(512, 160, stride= 1)
        self.conv32 = torch.nn.Conv2d(160, 1104, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn32 = torch.nn.BatchNorm2d(1104)
        self.conv33 = DepthwiseConv(1104, 96, stride= 1)
        self.conv34 = torch.nn.Conv2d(96, 656, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn34 = torch.nn.BatchNorm2d(656)
        self.conv35 = DepthwiseConv(656, 416, stride= 1)
        self.conv36 = torch.nn.Conv2d(416, 128, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn36 = torch.nn.BatchNorm2d(128)
        self.conv37 = DepthwiseConv(128, 784, stride= 1)
        self.conv38 = torch.nn.Conv2d(784, 128, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn38 = torch.nn.BatchNorm2d(128)
        self.conv39 = DepthwiseConv(128, 1024, stride= 2)
        self.conv40 = torch.nn.Conv2d(1024, 176, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn40 = torch.nn.BatchNorm2d(176)
        self.conv41 = DepthwiseConv(176, 1200, stride= 2)
        self.conv42 = torch.nn.Conv2d(1200, 960, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn42 = torch.nn.BatchNorm2d(960)
        self.conv43 = DepthwiseConv(960, 1104, stride= 1)
        self.conv44 = torch.nn.Conv2d(1104, 144, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn44 = torch.nn.BatchNorm2d(144)
        self.conv45 = DepthwiseConv(144, 848, stride= 1)
        self.conv46 = torch.nn.Conv2d(848, 640, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn46 = torch.nn.BatchNorm2d(640)
        self.conv47 = DepthwiseConv(640, 320, stride= 1)
        self.conv48 = torch.nn.Conv2d(320, 672, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn48 = torch.nn.BatchNorm2d(672)
        self.conv49 = DepthwiseConv(672, 496, stride= 1)
        self.conv50 = torch.nn.Conv2d(496, 1120, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn50 = torch.nn.BatchNorm2d(1120)
        self.conv51 = DepthwiseConv(1120, 176, stride= 1)
        self.conv52 = torch.nn.Conv2d(176, 384, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn52 = torch.nn.BatchNorm2d(384)
        self.conv53 = DepthwiseConv(384, 784, stride= 1)
        self.conv54 = torch.nn.Conv2d(784, 816, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn54 = torch.nn.BatchNorm2d(816)
        self.conv55 = DepthwiseConv(816, 528, stride= 1)
        self.conv56 = torch.nn.Conv2d(528, 320, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn56 = torch.nn.BatchNorm2d(320)
        self.conv57 = DepthwiseConv(320, 736, stride= 1)
        self.conv58 = torch.nn.Conv2d(736, 576, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn58 = torch.nn.BatchNorm2d(576)
        self.conv59 = DepthwiseConv(576, 752, stride= 1)
        self.conv60 = torch.nn.Conv2d(752, 352, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn60 = torch.nn.BatchNorm2d(352)
        self.conv61 = DepthwiseConv(352, 1168, stride= 1)
        self.conv62 = torch.nn.Conv2d(1168, 880, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn62 = torch.nn.BatchNorm2d(880)
        self.conv63 = DepthwiseConv(880, 1152, stride= 1)
        self.conv64 = torch.nn.Conv2d(1152, 48, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn64 = torch.nn.BatchNorm2d(48)
        self.conv65 = DepthwiseConv(48, 688, stride= 1)
        self.conv66 = torch.nn.Conv2d(688, 32, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn66 = torch.nn.BatchNorm2d(32)
        self.conv67 = DepthwiseConv(32, 1168, stride= 1)
        self.conv68 = torch.nn.Conv2d(1168, 64, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn68 = torch.nn.BatchNorm2d(64)
        self.conv69 = DepthwiseConv(64, 208, stride= 1)
        self.conv70 = torch.nn.Conv2d(208, 1104, (1, 1), stride=(1, 1), padding=(0, 0))
        self.conv_bn70 = torch.nn.BatchNorm2d(1104)
        self.classifier = torch.nn.Linear(1104, 1000)
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
        X1 = self.conv_bn0(X1)
        X1 = self.conv1(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn1(X1)
        X1 = self.conv2(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn2(X1)
        X1 = self.conv3(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn3(X1)
        X1 = self.conv4(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn4(X1)
        X1 = self.conv5(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn5(X1)
        X1 = self.conv6(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn6(X1)
        X1 = self.conv7(X1)
        X1 = self.avgpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn7(X1)
        X1 = self.conv8(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn8(X1)
        X1 = self.conv9(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn9(X1)
        X1 = X1.view(-1, 7168)
        X1 = self.fc0(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn10(X1)
        X1 = self.conv11(X1)
        X1 = self.relu(X1)
        X1 = self.conv12(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn12(X1)
        X1 = self.conv13(X1)
        X1 = self.relu(X1)
        X1 = self.conv14(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn14(X1)
        X1 = self.conv15(X1)
        X1 = self.relu(X1)
        X1 = self.conv16(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn16(X1)
        X1 = self.conv17(X1)
        X1 = self.relu(X1)
        X1 = self.conv18(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn18(X1)
        X1 = self.conv19(X1)
        X1 = self.relu(X1)
        X1 = self.conv20(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn20(X1)
        X1 = self.conv21(X1)
        X1 = self.relu(X1)
        X1 = self.conv22(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn22(X1)
        X1 = self.conv23(X1)
        X1 = self.relu(X1)
        X1 = self.conv24(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn24(X1)
        X1 = self.conv25(X1)
        X1 = self.relu(X1)
        X1 = self.conv26(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn26(X1)
        X1 = self.conv27(X1)
        X1 = self.relu(X1)
        X1 = self.conv28(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn28(X1)
        X1 = self.conv29(X1)
        X1 = self.relu(X1)
        X1 = self.conv30(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn30(X1)
        X1 = self.conv31(X1)
        X1 = self.relu(X1)
        X1 = self.conv32(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn32(X1)
        X1 = self.conv33(X1)
        X1 = self.relu(X1)
        X1 = self.conv34(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn34(X1)
        X1 = self.conv35(X1)
        X1 = self.relu(X1)
        X1 = self.conv36(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn36(X1)
        X1 = self.conv37(X1)
        X1 = self.relu(X1)
        X1 = self.conv38(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn38(X1)
        X1 = self.conv39(X1)
        X1 = self.relu(X1)
        X1 = self.conv40(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn40(X1)
        X1 = self.conv41(X1)
        X1 = self.relu(X1)
        X1 = self.conv42(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn42(X1)
        X1 = self.conv43(X1)
        X1 = self.relu(X1)
        X1 = self.conv44(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn44(X1)
        X1 = self.conv45(X1)
        X1 = self.relu(X1)
        X1 = self.conv46(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn46(X1)
        X1 = self.conv47(X1)
        X1 = self.relu(X1)
        X1 = self.conv48(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn48(X1)
        X1 = self.conv49(X1)
        X1 = self.relu(X1)
        X1 = self.conv50(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn50(X1)
        X1 = self.conv51(X1)
        X1 = self.relu(X1)
        X1 = self.conv52(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn52(X1)
        X1 = self.conv53(X1)
        X1 = self.relu(X1)
        X1 = self.conv54(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn54(X1)
        X1 = self.conv55(X1)
        X1 = self.relu(X1)
        X1 = self.conv56(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn56(X1)
        X1 = self.conv57(X1)
        X1 = self.relu(X1)
        X1 = self.conv58(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn58(X1)
        X1 = self.conv59(X1)
        X1 = self.relu(X1)
        X1 = self.conv60(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn60(X1)
        X1 = self.conv61(X1)
        X1 = self.relu(X1)
        X1 = self.conv62(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn62(X1)
        X1 = self.conv63(X1)
        X1 = self.relu(X1)
        X1 = self.conv64(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn64(X1)
        X1 = self.conv65(X1)
        X1 = self.relu(X1)
        X1 = self.conv66(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn66(X1)
        X1 = self.conv67(X1)
        X1 = self.relu(X1)
        X1 = self.conv68(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn68(X1)
        X1 = self.conv69(X1)
        X1 = self.relu(X1)
        X1 = self.conv70(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn70(X1)
        X1 = nn.functional.adaptive_avg_pool2d(X1, 1)
        X1 = X1.view(-1, 1104)
        X1 = self.classifier(X1)
        X1 = self.logsoftmax(X1)
        return X1

# Model ends here
def run_tvm_torch(n_trial = 200):

    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = 1
    input_features = 3072
    print("num of input features is:", input_features)
    X = torch.randn(batch_size, input_features, device=cuda_device)

    # Start Call Model

    model = cnn_32p3p1p1p0_depth688p1_16p1p1p0p0_depth912p1_912p1p1p0p0_depth1264p1_272p1p1p0p0_depth736p1_240p1p1p0p0_depth1200p1_800p1p1p0p0_depth608p1_544p1p1p0p0_depth192p1_656p1p1p0p0_depth32p2_1184p1p1p0p0_depth1200p2_384p1p1p0p0_depth848p1_1104p1p1p0p0_depth1008p1_240p1p1p0p0_depth1232p2_912p1p1p0p0_depth1024p1_48p1p1p0p0_depth528p1_256p1p1p0p0_depth256p1_512p1p1p0p0_depth160p1_1104p1p1p0p0_depth96p1_656p1p1p0p0_depth416p1_128p1p1p0p0_depth784p1_128p1p1p0p0_depth1024p2_176p1p1p0p0_depth1200p2_960p1p1p0p0_depth1104p1_144p1p1p0p0_depth848p1_640p1p1p0p0_depth320p1_672p1p1p0p0_depth496p1_1120p1p1p0p0_depth176p1_384p1p1p0p0_depth784p1_816p1p1p0p0_depth528p1_320p1p1p0p0_depth736p1_576p1p1p0p0_depth752p1_352p1p1p0p0_depth1168p1_880p1p1p0p0_depth1152p1_48p1p1p0p0_depth688p1_32p1p1p0p0_depth1168p1_64p1p1p0p0_depth208p1_1104p1p1p0p0_bn1_mlp_bn0(input_features).to(cuda_device)

    # End Call Model
    model.eval()

    # output = model(X)
    # print(output)
    # return
    scripted_model = torch.jit.trace(model, X).eval()

    input_name = "data"
    shape_list = [(input_name, X.size())]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    # print(mod.astext(show_meta_data=False))

    # Start Set Option

    opt_level = 3

    # End Set Option

    target = tvm.target.cuda()

    '''Tuning Option'''
    autotvm_on = True
    tuner_option = "xgb"
    tvm_log_file = "autotvm_tracegen.log"

    if not os.path.exists(tvm_log_file):
        #make a copy of the invoice to work with
        if "GTX 1660 SUPER" in GPUtil.getGPUs()[0].name:
            pretuned_file="./gtx1660super/autotvm_tracegen.log"
            shutil.copy(pretuned_file, tvm_log_file)
        elif "RTX 3090" in GPUtil.getGPUs()[0].name:
            pretuned_file="./rtx3090/autotvm_tracegen.log"
            shutil.copy(pretuned_file, tvm_log_file)
        else:
            print("Not pretune on your GPU type, generate new")
            open(tvm_log_file, 'a').close()

    if not autotvm_on:
        tvm_log_file = "faketvm_tracegen.log"
        if not os.path.exists(tvm_log_file):
            open(tvm_log_file, 'a').close()

    # extract workloads from relay program
    if autotvm_on:
        print("Extract tasks...")
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)
        print("Prune old tasks...")
        old_tasks = tasks
        tasks = prune_old_tasks(tasks, tvm_log_file)
        # run tuning tasks
        print("Tuning...")

        do_tune(tasks, old_tasks, tvm_log_file, n_trial, tuner_option)

    with autotvm.apply_history_best(tvm_log_file):
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target, params=params)

    lib.export_library("./temp_lib/temp_runtime.tar")

# run_tvm_torch()
    # input_data = tvm.nd.array(np.random.uniform(size=[batch_size, input_features]).astype("float32"))
    #
    # ctx = tvm.gpu()
    #
    # module = runtime.GraphModule(lib["default"](ctx))
    #
    # module.run(data=input_data)

    #     if model_name == "mlp":
    #         mlp = MLP4(input_features).to(cuda_device)
    #         new_out = mlp(X)
    #     elif model_name == "cnn":
    #         cnn = CNN4(input_features).to(cuda_device)
    #         new_out = cnn(X)
    # else:
    #     X = torch.randn(batch_size, input_features)
    #     if model_name == "mlp":
    #         mlp = MLP4(input_features)
    #         new_out = mlp(X)
    #     elif model_name == "cnn":
    #         cnn = CNN4(input_features)
    #         new_out = cnn(X)
