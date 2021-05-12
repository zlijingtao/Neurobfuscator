import torch
import time
import torch.nn.functional as F
import tvm
from tvm import relay
import numpy as np
import tvm.contrib.graph_executor as runtime
import argparse
import shutil
import os
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='input batch size.')
parser.add_argument('--input_features', type=int, default=3072, help='flattened input dimension.')
args = parser.parse_args()

batch_size = args.batch_size
input_features = args.input_features
ctx = tvm.gpu()

'''Single Test'''

path_lib = "./temp_lib/temp_runtime.tar"

if os.path.isdir("./temp_lib/temp_runtime"):
    shutil.rmtree("./temp_lib/temp_runtime")

loaded_lib = tvm.runtime.load_module(path_lib)

input_data = tvm.nd.array(np.random.uniform(size=[batch_size, input_features]).astype("float32"))

module = runtime.GraphModule(loaded_lib["default"](ctx))

module.run(data=input_data)
# out_deploy = module.get_output(0).asnumpy()
