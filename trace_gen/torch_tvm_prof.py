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

class cnn_256p3p1p1p1_512p3p1p1p1_16p3p1p1p2_256p3p1p1p2_64p3p1p1p0_bn1_mlp_128_128_bn1(torch.nn.Module):
    def __init__(self, input_features, reshape = True, decompo_list = None, dummy_list = None):
        super(cnn_256p3p1p1p1_512p3p1p1p1_16p3p1p1p2_256p3p1p1p2_64p3p1p1p0_bn1_mlp_128_128_bn1,self).__init__()
        self.reshape = reshape
        self.decompo_list = decompo_list
        self.dummy_list = dummy_list
        self.relu = torch.nn.ReLU(inplace=True)
        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)
        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool2x2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv0 = torch.nn.Conv2d(3, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn0 = torch.nn.BatchNorm2d(256)
        self.conv1 = torch.nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn1 = torch.nn.BatchNorm2d(512)
        self.conv2 = torch.nn.Conv2d(512, 16, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 256, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, 64, (3, 3), stride=(1, 1), padding=(1, 1))
        self.conv_bn4 = torch.nn.BatchNorm2d(64)
        self.fc0 = torch.nn.Linear(256, 128)
        self.fc_bn0 = torch.nn.BatchNorm1d(128)
        self.fc1 = torch.nn.Linear(128, 128)
        self.fc_bn1 = torch.nn.BatchNorm1d(128)
        self.classifier = torch.nn.Linear(128, 10)
        self.classifier_bn = torch.nn.BatchNorm1d(10, affine=False)
        self.reset_parameters(input_features)
    def reset_parameters(self, input_features):
        stdv = 1.0 / math.sqrt(input_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)
    def forward(self, X1):
        if self.reshape:
            X1 = X1.reshape(-1, 3, 32, 32)
        X1 = self.conv0(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn0(X1)
        X1 = self.conv1(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn1(X1)
        X1 = self.conv2(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn2(X1)
        X1 = self.conv3(X1)
        X1 = self.maxpool2x2(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn3(X1)
        X1 = self.conv4(X1)
        X1 = self.relu(X1)
        X1 = self.conv_bn4(X1)
        X1 = X1.view(-1, 256)
        X1 = self.fc0(X1)
        X1 = self.fc_bn0(X1)
        X1 = self.relu(X1)
        X1 = self.fc1(X1)
        X1 = self.fc_bn1(X1)
        X1 = self.relu(X1)
        X1 = self.classifier(X1)
        X1 = self.classifier_bn(X1)
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

    model = cnn_256p3p1p1p1_512p3p1p1p1_16p3p1p1p2_256p3p1p1p2_64p3p1p1p0_bn1_mlp_128_128_bn1(input_features).to(cuda_device)

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
