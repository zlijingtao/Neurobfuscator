import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import argparse
import os
import sys
import logging
import importlib
from torch_relay_build import torch_relay_func
from torch_func_modifier import func_modifier
sys.path.append('../seq_predictor')
from torch_utils import setup_logger, summary, get_extra_entries, identify_model, modify_state_dict, csv_to_time_overhead, reverse_map
from predict import predictor, layer_int_to_name_map
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

'''Randomly Generate Obfuscating operators'''
'''wident_list: type: float, range: [1.0, inf), help: the factor to the output channel/dim of current layer'''
'''decompo_list: type: int, range: [0, 4], help: 1: branch the oc by 2, 2: branch the oc by 4, 3: branch the ic by 2, 4: branch the ic by 4, 0: Do nothing'''
'''dummy_list: type: int, range: [0, inf), help: adding zero vector (same shape as current layer's activation) by N times'''
'''deepen_list: type: bin, help: add (or not) a deepen layer to the current layer'''
'''skipcon_list: type: bin, help: add (or not) a skipconnection layer to the current layer'''
'''kerneladd_list: type: int, range: [0, 1], help: padding zero to filter and input feature map, to fake a large filter size, 0: nothing, 1: padding 1 (filter size 3->5)'''

class Obfuscator(object):
    def __init__(self, model_id, batch_size, input_features, autotvm_on = True, n_trial = 200, tuner = 'xgb'):
        assert torch.cuda.is_available()
        self.cuda_device = torch.device("cuda")  # device object representing GPU
        self.batch_size = batch_size
        self.input_features = input_features
        self.model_id = model_id
        self.autotvm_on = autotvm_on
        self.n_trial = n_trial
        self.tuner = tuner
        self.opt_level = 3
        self.run_style = "normal"
        self.model_name = "model_{}".format(self.model_id+1)
        mod = importlib.import_module("." + self.model_name, package="model_file")
        self.cnn = getattr(mod, "custom_cnn_{}".format(self.model_id))
        self.out_name = "env"


        # from model_3 import custom_cnn_2 (load original model before obfuscation)
        self.model = self.cnn(input_features, False).to(self.cuda_device)

        #print out the model to check the range of obfuscation
        model_log_file = './obf_tmp_file/model_info.log'
        logger = setup_logger('first_logger', model_log_file, level = logging.DEBUG)
        if self.input_features == 3072:
            logger.debug(summary(self.model, (3,32,32)))
        elif self.input_features == 150528:
            logger.debug(summary(self.model, (3,224,224)))
        elif self.input_features == 784:
            logger.debug(summary(self.model, (1,28,28)))

        #Get the length of the model, initialize three lists specifying the obfuscation
        self.modify_list, self.decompo_list, self.kerneladd_list = identify_model(model_log_file)
        self.widen_list = [1.0] * len(self.decompo_list)
        self.dummy_list = [0] * len(self.decompo_list)
        self.deepen_list = [0] * len(self.decompo_list)
        self.skipcon_list = [0] * len(self.decompo_list)
        # self.fuse_list = [9] * len(self.modify_list)
        self.fuse_list = [9] * 400
        self.prune_list = [0] * 14
        self.model_file = self.model_name + ".py"
        self.model_file = func_modifier(self.model_file, self.modify_list)
        new_model_name = self.model_file.split(".py")[0]
        mod = importlib.import_module("." + new_model_name, package="model_file")
        self.cnn = getattr(mod, "custom_cnn_{}".format(model_id))
        
    def get_current_dict(self):
        dict = {}
        dict['widen_list'] = self.widen_list
        dict['decompo_list'] = self.decompo_list
        dict['dummy_list'] = self.dummy_list
        dict['deepen_list'] = self.deepen_list
        dict['skipcon_list'] = self.skipcon_list
        dict['kerneladd_list'] = self.kerneladd_list
        dict['fuse_list'] = self.fuse_list
        dict['prune_list'] = self.prune_list
        return dict

    def print_dict_string(self):
        out_str = "\n"
        current_dict = self.get_current_dict()
        for key in current_dict.keys():
            if key == 'widen_list':
                default_value = 1.0
            elif key == 'decompo_list':
                default_value = 0
            elif key == 'fuse_list':
                default_value = 9
            else:
                default_value = 0
            out_str += "{}: ".format(key)
            for i in range(len(current_dict[key])):
                if current_dict[key][i] != default_value:
                    out_str += "{}[{}] = {} | ".format(key, i, current_dict[key][i])
            out_str += "\n"
        return out_str

    def get_full_length(self):
        return len(self.decompo_list)

    def get_conv_length(self):
        return len(self.kerneladd_list)

    def get_fuse_length(self):
        # obf_model = self.cnn(self.input_features, False, self.decompo_list, self.dummy_list).to(self.cuda_device)
        # model_log_file = "./obf_tmp_file/model_obf_info.log"
        # if os.path.isfile(model_log_file):
        #     os.remove(model_log_file)
        #     open(model_log_file, 'a').close()
        # obf_logger = setup_logger('obf_logger', model_log_file, level = logging.DEBUG)
        # if self.input_features == 3072:
        #     obf_logger.debug(summary(obf_model, (3,32,32)))
        # elif self.input_features == 150528:
        #     obf_logger.debug(summary(obf_model, (3,224,224)))
        # elif self.input_features == 784:
        #     obf_logger.debug(summary(obf_model, (1,28,28)))
        # modify_list, _, _ = identify_model(model_log_file)

        # self.fuse_list = [9] * (len(modify_list) + get_extra_entries(self.decompo_list, self.dummy_list, self.deepen_list, self.skipcon_list))
        return len(self.fuse_list)

    def apply_dd(self, widen_list, decompo_list, dummy_list, deepen_list, skipcon_list, kerneladd_list, prune_list):
        if len(widen_list) == len(self.widen_list):
            self.widen_list = widen_list
        else:
            print("widen_list apply failed. the length needs to match")

        if len(decompo_list) == len(self.decompo_list):
            self.decompo_list = decompo_list
        else:
            print("decompo_list apply failed. the length needs to match")

        if len(deepen_list) == len(self.deepen_list):
            self.deepen_list = deepen_list
        else:
            print("deepen_list apply failed. the length needs to match")

        if len(skipcon_list) == len(self.skipcon_list):
            self.skipcon_list = skipcon_list
        else:
            print("skipcon_list apply failed. the length needs to match")

        if len(kerneladd_list) == len(self.kerneladd_list):
            self.kerneladd_list = kerneladd_list
        else:
            print("kerneladd_list apply failed. the length needs to match")

        if len(dummy_list) == len(self.dummy_list):
            self.dummy_list = dummy_list
        else:
            print("dummy_list apply failed. the length needs to match")
        
        if len(prune_list) == len(self.prune_list):
            self.prune_list = prune_list
        else:
            print("dummy_list apply failed. the length needs to match")
        
        if os.path.isfile("./model_file/{}.pickle".format(self.model_name)):
            # print("load state_dict...")
            state_dict = torch.load("./model_file/{}.pickle".format(self.model_name))
        else:
            state_dict = self.model.state_dict()
            # print("create new state_dict...")
            torch.save(state_dict, "./model_file/{}.pickle".format(self.model_name))

        state_dict = modify_state_dict(self.model_file, state_dict, self.modify_list, self.widen_list, self.decompo_list, self.deepen_list, self.skipcon_list, self.kerneladd_list)
        torch.save(state_dict, "./model_file/{}_obf.pickle".format(self.model_name))
        return 0

    def apply_fuse(self, fuse_list):
        if len(fuse_list) <= self.get_fuse_length():
            self.fuse_list[:len(fuse_list)] = fuse_list
        else:
            print("fuse_list apply failed. the length needs to be smaller than the actual one")
        return 0

    def trace_gen(self):
        torch_relay_func(self, False, self.autotvm_on)
        return 0


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=3, help='model_id, 0 ~ 2')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size.')
    parser.add_argument('--input_features', type=int, default=3072, help='flattened input dimension.')
    parser.add_argument('--out_name', type=str, default="0", help='lib_name=lib_${model_name}_${out_name}')
    parser.add_argument('--n_trial', type=int, default=200, help='number of iteration for auto-scheduler')
    parser.add_argument('--autotvm_on', action='store_true', default=True, help='Use TVM auto-scheduler or not')
    parser.add_argument('--tuner', type=str, default="xgb", choices = ['xgb', 'xgb-rank', 'ga', 'random', 'gridsearch'], help='type of tuner for autoTVM')
    parser.add_argument('--run_style', type=str, default="normal", choices = ['normal', 'test_tuner'], help='normal test or test tuner (for Fig.xx)')
    parser.add_argument('--widen_list', type=str, default="None", help='')
    parser.add_argument('--decompo_list', type=str, default="None", help='')
    parser.add_argument('--dummy_list', type=str, default="None", help='')
    parser.add_argument('--deepen_list', type=str, default="None", help='')
    parser.add_argument('--skipcon_list', type=str, default="None", help='')
    parser.add_argument('--kerneladd_list', type=str, default="None", help='')
    parser.add_argument('--fuse_list', type=str, default="None", help='')
    parser.add_argument('--predict_type', type=str, default="full", help='Pick dataset you want to predict on', choices=("reduced", "full", "time_only"))
    parser.add_argument('--restore_step', type=int, default=149, help='Global step to restore from checkpoint.')
    parser.add_argument('--normalize', type=str, default="smart", help='Pick normalization for the training data, need to match with the predictor', choices=("sb", "smart"))
    parser.add_argument('--num_hidden', type=int, default=128, help='number of hidden units for the LSTM')
    parser.add_argument('--predictor_name', type=str, default='deepsniffer_LSTM_both_autotvm_smart')
    args = parser.parse_args()

    predict_type = args.predict_type
    restore_step = args.restore_step
    normalize = args.normalize
    num_hidden = args.num_hidden
    predictor_name = "{}_{}_cifar10".format(args.predictor_name, num_hidden)
    print("Predictor: ({}) ".format(predict_type) + predictor_name)


    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = args.batch_size
    input_features = args.input_features
    opt_level = 3
    torch.manual_seed(1234)
    X = torch.randn(batch_size, input_features, device=cuda_device)
    model_id = args.model_id
    model_name = "model_{}".format(model_id+1)
    mod = importlib.import_module("." + model_name, package="model_file")
    cnn = getattr(mod, "custom_cnn_{}".format(model_id))

    # from model_3 import custom_cnn_2 (load original model before obfuscation)
    model = cnn(input_features, False).to(cuda_device)

    #print out the model to check the range of obfuscation
    model_log_file = './obf_tmp_file/model_info.log'
    logger = setup_logger('first_logger', model_log_file)
    if args.input_features == 3072:
        logger.info(summary(model, (3,32,32)))
    elif args.input_features == 150528:
        logger.info(summary(model, (3,224,224)))
    elif args.input_features == 784:
        logger.info(summary(model, (1,28,28)))

    #Get the length of the model, initialize three lists specifying the obfuscation
    modify_list, decompo_list, kerneladd_list = identify_model(model_log_file)
    widen_list = [1.0] * len(decompo_list)
    dummy_list = [0] * len(decompo_list)
    deepen_list = [0] * len(decompo_list)
    skipcon_list = [0] * len(decompo_list)
    prune_list =  [0] * 14

    
    # widen_list[1] = 1.25
    # decompo_list[2] = 1
    # decompo_list[0] = 0
    # deepen_list[1] = 1
    # skipcon_list[1] = 1
    # prune_list = [3] * 14
    # decompo_list[2] = 2
    # decompo_list[3] = 3
    # dummy_list[2] = 5
    # widen_list[1] = 1.00
    # widen_list[5] = 1.50
    # kerneladd_list[4] = 1
    
    # '''Set the obfuscating operators here (default value for each entry is 5 in decompo_list)'''
    # decompo_list[2] = 2
    # decompo_list[3] = 3

    # '''Set the obfuscating operators here (default value for each entry is 0 in dummy_list)'''
    # dummy_list[2] = 5

    # '''Set the obfuscating operators here (default value for each entry is 1.0 in widen_list)'''
    # widen_list[1] = 1.00
    
    # widen_list[5] = 1.50

    # '''Set the obfuscating operators here (default value for each entry is 0 in deepen_list)'''
    # deepen_list[1] = 0

    # '''Set the obfuscating operators here (default value for each entry is 0 in skipcon_list)'''
    # skipcon_list[1] = 0

    # '''Set the obfuscating operators here (default value for each entry is 0 in kerneladd_list)'''
    # kerneladd_list[4] = 1

    #obfuscate the model with the lists
    
    #step 1 add API to the function
    model_file = model_name + ".py"
    new_model_file = func_modifier(model_file, modify_list)
    new_model_name = new_model_file.split(".py")[0]
    mod = importlib.import_module("." + new_model_name, package="model_file")
    cnn_obf = getattr(mod, "custom_cnn_{}".format(model_id))


    #step 2 load and modify the model state_dict
    if os.path.isfile("./model_file/{}.pickle".format(model_name)):
        state_dict = torch.load("./model_file/{}.pickle".format(model_name))
    else:
        state_dict = model.state_dict()
        torch.save(state_dict, "./model_file/{}.pickle".format(model_name))

    state_dict = modify_state_dict(new_model_file, state_dict, modify_list, widen_list, decompo_list, deepen_list, skipcon_list, kerneladd_list)
    torch.save(state_dict, "./model_file/{}_obf.pickle".format(model_name))

    del model
    torch.cuda.empty_cache()


    model = cnn_obf(input_features, False, widen_list, decompo_list, dummy_list, deepen_list, skipcon_list, kerneladd_list).to(cuda_device)

    model_log_file = "./obf_tmp_file/model_obf_info.log"
    obf_logger = setup_logger('obf_logger', model_log_file)
    if args.input_features == 3072:
        obf_logger.info(summary(model, (3,32,32)))
    elif args.input_features == 150528:
        obf_logger.info(summary(model, (3,224,224)))
    elif args.input_features == 784:
        obf_logger.info(summary(model, (1,28,28)))

    del model
    torch.cuda.empty_cache()

    modify_list, _, _ = identify_model(model_log_file)
    fuse_list = [9] * (len(modify_list) + sum(dummy_list))
    fuse_list = [9] * (len(modify_list) + get_extra_entries(decompo_list, dummy_list, deepen_list, skipcon_list))
    
    '''Set the obfuscating operators here (default value for each entry is 99)'''
    fuse_list[1] = 9
    fuse_list[2] = 9
    
    
    args.widen_list = widen_list
    args.decompo_list = decompo_list
    args.dummy_list = dummy_list
    args.deepen_list = deepen_list
    args.skipcon_list = skipcon_list
    args.kerneladd_list = kerneladd_list
    args.fuse_list = fuse_list
    args.prune_list = prune_list
    #step 3 call the torch_relay_func() with all four lists
    '''Generate the Trace File'''
    torch_relay_func(args, True, args.autotvm_on)
    
    
    '''Calculate the total Cycle cost'''
    sample_file = "./env_file/lib_model_{}_obf_{}.csv".format(model_id+1, args.out_name)
    label_file = "./model_file/model_{}.npy".format(model_id+1)
    
    '''Get the prediction result'''
    if predict_type == "reduced":
        log_dir = "../seq_predictor/obfuscator/predictor/logs_{}".format(predictor_name)
    elif predict_type == "full":
        log_dir = "../seq_predictor/obfuscator/predictor/logs_full_{}".format(predictor_name)
    elif predict_type == "time_only":
        log_dir = "../seq_predictor/obfuscator/predictor/logs_timeonly_{}".format(predictor_name)
    trace_predictor = predictor(log_dir, restore_step, label_file, sample_file, predict_type, normalize, num_hidden, 1)
    _, predict = trace_predictor.get_reward()
    print("Prediction after obfuscation is:\n", predict)
    ground_truth = reverse_map(label_file, layer_int_to_name_map)
    print("Original Architecture Sequence is:\n", ground_truth)
    #finished the obfuscation

if __name__ == '__main__':
    main()
