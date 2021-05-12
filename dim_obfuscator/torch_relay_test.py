import torch
import time
import torch.nn.functional as F
import tvm
from tvm import relay
import numpy as np
import tvm.contrib.graph_executor as runtime
import argparse


def run_lib(args):

    batch_size = args.batch_size
    input_features = args.input_features
    ctx = tvm.gpu()

    '''Single Test'''

    lib_name = args.lib_name

    path_lib = "./deploy_lib/{}.tar".format(lib_name)

    loaded_lib = tvm.runtime.load_module(path_lib)
    np.random.seed(1234)
    input_data = tvm.nd.array(np.random.uniform(size=[batch_size, input_features]).astype("float32"))

    module = runtime.GraphModule(loaded_lib["default"](ctx))
    module.run(data=input_data)
    out_deploy = module.get_output(0).asnumpy()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lib_name', type=str, default="lib_model_3_opt_3", help='library name in ./deploy_lib/')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size.')
    parser.add_argument('--input_features', type=int, default=3072, help='flattened input dimension.')
    args = parser.parse_args()
    
    run_lib(args)