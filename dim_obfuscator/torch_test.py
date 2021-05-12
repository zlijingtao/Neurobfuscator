import torch
import time
import torch.nn.functional as F
import numpy as np
import argparse


def run_lib(args):
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU
    
    batch_size = args.batch_size
    input_features = args.input_features
    
    lib_name = args.lib_name

    path_lib = "./deploy_lib/torch_{}.tar".format(lib_name)

    '''Single Test'''
    torch.manual_seed(1234)
    X = torch.randn(batch_size, input_features, device=cuda_device)
    model = torch.load(path_lib)
    model.eval()
    out = model(X)
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--lib_name', type=str, default="lib_model_3_opt_3", help='library name in ./deploy_lib/')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size.')
    parser.add_argument('--input_features', type=int, default=3072, help='flattened input dimension.')
    args = parser.parse_args()
    
    run_lib(args)