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
import pandas as pd
import importlib
from datetime import datetime
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


def reward_function(current_param, clean_param):
    # clean_param = np.array([64, 128])
    return np.sum(np.abs(current_param - clean_param)/ clean_param)

def fitness_function(avg_DER, cycle, clean_cycle, budget = 0.02):
    offset = (budget / 2.) ** 2
    # offset = 0.02
    if cycle/clean_cycle - 1.0 - budget > 0:
        fitness = avg_DER * (1/ ((cycle/clean_cycle - 1.0 - budget)**2 + offset))
    else:
        fitness = avg_DER * (1/ offset)
    return fitness

def extract_param_from_str(param_str, operator_name = "conv"):
    param = []
    if operator_name == "conv":
        param.append(round(float(param_str.split(", ")[0].split("Conv2D(")[1])))
        param.append(round(float(param_str.split(", ")[1])))
    elif operator_name == "fc":
        param.append(round(float(param_str.split(", ")[0].split("Linear(")[-1])))
    elif operator_name == "depth":
        param.append(round(float(param_str.split(", ")[0].split("DepthwiseConv2D(")[1])))
        param.append(round(float(param_str.split(", ")[1])))
    return np.asarray(param)

def csv_to_time_overhead(csv_file):
    '''Sum up the Cycles in the DNN's profiling report (.csv)'''
    try:
        df = pd.read_csv(csv_file, skiprows=2)
    except:
        return 999999999
    trace_df = df[df['Metric Name'] == "Cycles"]
    trace_df= trace_df.replace(',','', regex=True)
    trace_df['Metric Value'] = pd.to_numeric(trace_df['Metric Value'])

    cost = trace_df['Metric Value'].sum()
    return cost

def cal_pop_fitness(env, parent_list, budget = 0.02):
    fitness_list = []
    cycle_list = []
    DER_list = []
    param_list = []
    clean_cycle = env.clean_cycles
    clean_param = env.clean_param
    for parent in parent_list:
        env.assign_dict(parent)
        current_param, cycle = env.get_3_avg_param()
        avg_DER = reward_function(current_param, clean_param)
        fitness = fitness_function(avg_DER, cycle, clean_cycle, budget)
        fitness_list.append(fitness)
        cycle_list.append(cycle)
        DER_list.append(avg_DER)
        param_list.append(current_param)
    return fitness_list, cycle_list, DER_list, param_list

def select_mating_pool(parent_list, fitness_list, n_mating):
    parents = []
    fitness_array = np.array(fitness_list)
    for i in range(n_mating):
        max_index = np.where(fitness_array == np.max(fitness_array))
        max_index = max_index[0][0]
        parents.append(parent_list[max_index])
        fitness_array[max_index] = -99
    return parents

def crossover(parents, offspring_size):
    offspring = []
    cross_point = np.random.randint(len(parents[0]["widen_list"]))
    cross_point1 = np.random.randint(len(parents[0]["kerneladd_list"]))
    cross_point2 = np.random.randint(14)
    for k in range(offspring_size):
        parent1_idx = k%len(parents)
        parent2_idx = (k+1)%len(parents)
        parent1_dict = parents[parent1_idx]
        parent2_dict = parents[parent2_idx]
        of_dict = {}
        of_dict["widen_list"] = parent1_dict["widen_list"][:cross_point]
        of_dict["widen_list"].extend(parent2_dict["widen_list"][cross_point:])
        of_dict["dummy_list"] = parent1_dict["dummy_list"][:cross_point]
        of_dict["dummy_list"].extend(parent2_dict["dummy_list"][cross_point:])
        of_dict["kerneladd_list"] = parent1_dict["kerneladd_list"][:cross_point1]
        of_dict["kerneladd_list"].extend(parent2_dict["kerneladd_list"][cross_point1:])
        of_dict["prune_list"] = parent1_dict["prune_list"][:cross_point2]
        of_dict["prune_list"].extend(parent2_dict["prune_list"][cross_point2:])

        dummy_arr = np.asarray(of_dict['dummy_list'])
        kerneladd_arr = np.asarray(of_dict['kerneladd_list'])
        dummy_arr[1:] = 0 #Force the second, third entry to be 0
        kerneladd_arr[1:] = 0 #Force the second, third entry to be 0
        of_dict['dummy_list'] = list(dummy_arr.flatten().astype(int))
        of_dict['kerneladd_list'] = list(kerneladd_arr.flatten().astype(int))
        offspring.append(of_dict)
    return offspring

def mutation(offspring_list, sigma, forbid_List):
    for i in range(len(offspring_list)):
        offspring_list[i] = apply_noise(offspring_list[i], sigma = sigma, forbid_List = forbid_List)
    return offspring_list

def apply_noise(dict, sigma, forbid_List):
    '''Apply noise to obfuscating operators'''
    '''All noise are Gaussian noise, generated by the noise-vector'''
    widen_arr = np.asarray(dict['widen_list'])
    dummy_arr = np.asarray(dict['dummy_list'])
    kerneladd_arr = np.asarray(dict['kerneladd_list'])
    prune_array = np.asarray(dict['prune_list'])

    noise_list = [1/4, 1/2, 1/2, 1/2, 1/2]

    if 'widen_list' in forbid_List:
        noise_list[0] = 0
    if 'dummy_list' in forbid_List:
        noise_list[1] = 0
    if 'kerneladd_list' in forbid_List:
        noise_list[2] = 0
    if 'prune_list' in forbid_List:
        noise_list[3] = 0
        noise_list[4] = 0

    noise = np.random.randn(1, widen_arr.shape[0])
    widen_arr = widen_arr + noise_list[0] * np.floor(1/2 * sigma * noise)
    # widen_arr[2:] = 0 #Force the third entry to be 0

    noise = np.random.randn(1, dummy_arr.shape[0])
    dummy_arr = dummy_arr + noise_list[1] * sigma * noise
    dummy_arr[1:] = 0 #Force the second, third entry to be 0

    noise = np.random.randn(1, kerneladd_arr.shape[0])
    kerneladd_arr = kerneladd_arr + noise_list[2] * sigma * noise
    kerneladd_arr[1:] = 0 #Force the second, third entry to be 0

    noise = np.random.randn(1, prune_array.shape[0])
    prune_bin_entry = [1, 2, 5, 6, 7, 8, 13]
    prune_4_entry = [0, 3, 4, 9, 10, 11, 12]
    prune_array[prune_bin_entry] = np.clip(np.floor(prune_array[prune_bin_entry] + noise_list[3] * sigma * noise[0][prune_bin_entry]), 0, 1)
    prune_array[prune_4_entry] = np.clip(np.floor(prune_array[prune_4_entry] + noise_list[4] * sigma * noise[0][prune_4_entry]), 0, 3)

    act_dict = {}
    act_dict['widen_list'] = list(np.clip(widen_arr, 1, 2).flatten().astype(float))
    act_dict['dummy_list'] = list(np.clip(np.floor(dummy_arr), 0, 4).flatten().astype(int))
    act_dict['kerneladd_list'] = list(np.clip(np.floor(kerneladd_arr), 0, 1).flatten().astype(int))
    act_dict['prune_list'] = list(prune_array.flatten().astype(int))
    return act_dict



class dim_obf_env(object):
    def __init__(self, batch_size, input_features, model_id, out_name, obf_logger, selected_entry = None):
        assert torch.cuda.is_available()
        self.cuda_device = torch.device("cuda")  # device object representing GPU
        self.batch_size = batch_size
        self.input_features = input_features
        self.model_id = model_id
        self.run_style = "normal"
        self.n_trial = 50
        self.tuner = "xgb"
        self.model_name = "model_{}".format(model_id)
        self.out_name = out_name
        mod = importlib.import_module("." + self.model_name, package="model_file")
        cnn = getattr(mod, "custom_cnn_{}".format(model_id))
        model = cnn(input_features, False).to(self.cuda_device)

        #print out the model to check the range of obfuscation
        model_log_file = './obf_tmp_file/model_info.log'
        logger = setup_logger('first_logger', model_log_file)

        #load its sequence (use the ground-truth one, we have no segmentable here.)
        model_npy_name = np.load("./model_file/" + self.model_name + ".npy")
        print(model_npy_name)

        #use the n-th entry of trace file in predictor for obfuscation objective
        self.selected_entry = selected_entry
        operator_name = layer_int_to_name_map[model_npy_name[1]] #TODO: will be dependent on selected entry.
        if not self.selected_entry:
            if self.input_features == 3072:
                operator_name = layer_int_to_name_map[model_npy_name[1]]
                if operator_name == "conv" or operator_name == "depth":
                    logger.info(summary(model, (3,32,32)))
                elif operator_name == "fc":
                    logger.info(summary(model, (3072,)))
                self.selected_entry = 3 #Very Important:
                
            elif self.input_features == 150528:
                operator_name = layer_int_to_name_map[model_npy_name[1]]
                if operator_name == "conv" or operator_name == "depth":
                    logger.info(summary(model, (3,224,224)))
                self.selected_entry = 10 #Very Important:

            elif self.input_features == 784:
                operator_name = layer_int_to_name_map[model_npy_name[1]]
                if operator_name == "conv" or operator_name == "depth":
                    logger.info(summary(model, (1,28,28)))
                elif operator_name == "fc":
                    logger.info(summary(model, (784,)))
                self.selected_entry = 1
                
        self.operator_name  = operator_name
        #Get the length of the model, initialize three lists specifying the obfuscation
        self.modify_list, self.decompo_list, self.kerneladd_list = identify_model(model_log_file)

        #Start Obfuscation
        self.obf_logger = obf_logger

        #step 1 add API to the function
        model_file = self.model_name + ".py"
        self.new_model_file = func_modifier(model_file, self.modify_list)
        new_model_name = self.new_model_file.split(".py")[0]
        mod = importlib.import_module("." + new_model_name, package="model_file")
        self.cnn_obf = getattr(mod, "custom_cnn_{}".format(model_id))

        #step 2 load and modify the model state_dict
        if os.path.isfile("./model_file/{}.pickle".format(self.model_name)):
            state_dict = torch.load("./model_file/{}.pickle".format(self.model_name))
        else:
            state_dict = model.state_dict()
            torch.save(state_dict, "./model_file/{}.pickle".format(self.model_name))

        del model

        self.widen_list = [1.0] * len(self.decompo_list)
        self.dummy_list = [0] * len(self.decompo_list)
        self.deepen_list = [0] * len(self.decompo_list)
        self.skipcon_list = [0] * len(self.decompo_list)
        self.prune_list =  [0] * 14
        self.fuse_list = [25] * 100

        self.obf_dict = {"widen_list": self.widen_list, "dummy_list": self.dummy_list,
                    "kerneladd_list": self.kerneladd_list, "prune_list": self.prune_list}
        self.cycles = 0
        if operator_name == "conv" or operator_name == "depth":
            self.clean_param, self.clean_cycles = self.get_3_avg_param(1, 3, 3)
        elif operator_name == "fc":
            self.clean_param, self.clean_cycles = self.get_3_avg_param(1, 0, 3)
    def get_obf_dict(self):
        return self.obf_dict

    def assign_dict(self, input_dict):
        self.widen_list = input_dict['widen_list']
        self.kerneladd_list = input_dict['kerneladd_list']
        self.dummy_list = input_dict['dummy_list']
        self.prune_list = input_dict['prune_list']
        if "fuse_list" in input_dict.keys():
            self.fuse_list = input_dict['fuse_list']
        if "decompo_list" in input_dict.keys():
            self.decompo_list = input_dict['decompo_list']
        self.obf_dict = input_dict

    def apply_dict(self):

        state_dict = torch.load("./model_file/{}.pickle".format(self.model_name))
        state_dict = modify_state_dict(self.new_model_file, state_dict, self.modify_list, self.widen_list, self.decompo_list, self.deepen_list, self.skipcon_list, self.kerneladd_list)
        torch.save(state_dict, "./model_file/{}_obf.pickle".format(self.model_name))
        torch.cuda.empty_cache()

        model = self.cnn_obf(self.input_features, False, self.widen_list, self.decompo_list, self.dummy_list, self.deepen_list, self.skipcon_list, self.kerneladd_list).to(self.cuda_device)

        del model
        torch.cuda.empty_cache()
        #step 3 call the torch_relay_func() with all four lists
        '''Generate the Trace File'''
        torch_relay_func(self, True, True)

        '''Calculate the total Cycle cost'''
        sample_file = "./env_file/lib_model_{}_obf_{}.csv".format(self.model_id, self.out_name)

        # try:
        df = pd.read_csv(sample_file, skiprows=2)
        trace_df = df[['ID', 'Metric Name', 'Metric Value']]
        reduced_trace_array = np.zeros((1, trace_df['ID'].nunique(), 5))
        full_trace_array = np.zeros((1, trace_df['ID'].nunique(), trace_df['Metric Name'].nunique()))

        old_row_id = -1
        count = 0
        for index, row in trace_df.iterrows():
            if row['ID'] == old_row_id:
                count += 1
            else:
                old_row_id += 1
                count = 0
            full_trace_array[0, old_row_id, count] = row['Metric Value']
            if row['Metric Name'] == 'Cycles':
                reduced_trace_array[0, old_row_id, 0] = row['Metric Value']
            elif row['Metric Name'] == 'Mem Read':
                reduced_trace_array[0, old_row_id, 1] = row['Metric Value']
            elif row['Metric Name'] == 'Mem Write':
                reduced_trace_array[0, old_row_id, 2] = row['Metric Value']
        reduced_trace_array[0, 0, 4] = 1.0
        for i in range(trace_df['ID'].nunique()):
            if reduced_trace_array[0, i, 2] != 0:
                reduced_trace_array[0, i, 3] = reduced_trace_array[0, i, 1]/reduced_trace_array[0, i, 2]
            if i > 0:
                if reduced_trace_array[0, i-1, 2] != 0:
                    reduced_trace_array[0, i, 4] = reduced_trace_array[0, i, 1]/reduced_trace_array[0, i-1, 2]
                else:
                    reduced_trace_array[0, i, 4] = 1.0

        full_trace_array = np.nan_to_num(full_trace_array)
        full_trace_list = list(full_trace_array[0, self.selected_entry, :])

        reduced_trace_array = np.nan_to_num(reduced_trace_array)
        reduced_trace_list = list(reduced_trace_array[0, self.selected_entry, :])
        timeonly_trace_list = [reduced_trace_list[0]]

        #Append input image height/width
        if self.operator_name == "conv" or self.operator_name == "depth":
            if self.input_features == 3072:
                full_trace_list.append(32)
                reduced_trace_list.append(32)
                timeonly_trace_list.append(32)
            elif self.input_features == 150528:
                full_trace_list.append(224)
                reduced_trace_list.append(224)
                timeonly_trace_list.append(224)
            elif self.input_features == 784:
                full_trace_list.append(28)
                reduced_trace_list.append(28)
                timeonly_trace_list.append(28)
        elif self.operator_name == "fc":
            full_trace_list.append(self.input_features)
            reduced_trace_list.append(self.input_features)
            timeonly_trace_list.append(self.input_features)
        overall_list = [timeonly_trace_list, reduced_trace_list, full_trace_list]

        self.cycles = overall_list[0][0]
        self.cycles = csv_to_time_overhead(sample_file)
        sys.path.append('../dim_predictor')
        from dim_predict import Dim_predictor

        dim_predictor = Dim_predictor(path_to_model = "../dim_predictor/saved_models", n_estimators = 100, min_samples_split = 30, dataset_type = "all", operator_name = self.operator_name)
        final_str = ""
        final_str += "Average Prediction on three cases: \n"
        final_str += (dim_predictor.predict(overall_list) + "\n")
        print(final_str)
        # except:
        #     print("Exeception! Output GroundTruth as prediction.")
        #     final_str = "Average Prediction on three cases: \nPrediction is:\n Conv2D(64, 128, kernel_size = 3, stride = 1, padding = 1)\n"
        return final_str
    def random_act_dict_gen(self, range_list):
        '''Randomly Generate Obfuscating operators'''
        '''widen_list: type: float, range: [1.0, inf), help: the factor to the output channel/dim of current layer'''
        '''dummy_list: type: int, range: [0, inf), help: adding zero vector (same shape as current layer's activation) by N times'''
        '''kerneladd_list: type: int, range: [0, 1], help: padding zero to filter and input feature map, to fake a large filter size, 0: nothing, 1: padding 1 (filter size 3->5)'''
        '''prune_list: type int, range: [0, 1] for prune_bin_entry, [0, 3] for prune_4_entry'''


        '''widen_list follow 1 + 0.125 * randint[1,5) * bernouli (controlled by range_list[0])'''
        self.widen_list = list(1 + 0.125 * (np.random.randint(low = 1, high = 7, size=(1, len(self.widen_list))) * np.random.binomial(size=(1, len(self.widen_list)), n=1, p = range_list[0])).flatten())
        self.widen_list[2] = 0
        '''dummy_list follow randint[0, 5)  * bernouli (controlled by range_list[0])'''
        self.dummy_list = list((np.random.randint(5, size=(1, len(self.dummy_list))) * np.random.binomial(size=(1, len(self.dummy_list)), n=1, p = range_list[1])).flatten())
        for i in range(len(self.dummy_list)):
            self.dummy_list[i] = 0
        '''lists below follow bernouli (controlled by range_list[2], [3], [4])'''
        self.kerneladd_list = list(np.random.binomial(size=(1, len(self.kerneladd_list)), n=1, p = range_list[2]).flatten())
        for i in range(len(self.kerneladd_list)):
            self.kerneladd_list[i] = 0
        '''prune_list prune_bin_entry follows bernouli (controlled by range_list[5]), prune_4_entry follows randint[0, 4)'''
        prune_bin_entry = [1, 2, 5, 6, 7, 8, 13]
        prune_4_entry = [0, 3, 4, 9, 10, 11, 12]
        prune_array = np.zeros((14,))
        prune_array[prune_bin_entry] = np.random.binomial(size=(1, len(prune_bin_entry)), n=1, p = range_list[3])
        prune_array[prune_4_entry] = np.random.randint(4, size=(1, len(prune_4_entry))) * np.random.binomial(size=(1, len(prune_4_entry)), n=1, p = range_list[3])
        self.prune_list = list(prune_array.flatten())
        self.obf_dict = {"widen_list": self.widen_list, "dummy_list": self.dummy_list,
                    "kerneladd_list": self.kerneladd_list, "prune_list": self.prune_list}

    def avg_param_from_dependency(self, l1_id = 1, l2_id =3):
        total_cycles = 0
        self.selected_entry = l2_id
        current_prediction = self.apply_dict()
        current_param = extract_param_from_str(current_prediction, self.operator_name)
        total_cycles += self.cycles
        self.selected_entry = l1_id
        avg_param = current_param
        current_prediction = self.apply_dict()
        previous_param = extract_param_from_str(current_prediction, self.operator_name)
        total_cycles += self.cycles
        if self.operator_name != "fc":
            avg_param[0] = (current_param[0] + previous_param[1])/2
        return avg_param, total_cycles
    
    def get_3_avg_param(self, l1_id = 1, l2_id =3, average_time = 1):
        current_param, current_cycle = self.avg_param_from_dependency(l1_id, l2_id)
        for i in range(average_time - 1):
            current_param_i, current_cycle_i = self.avg_param_from_dependency(l1_id, l2_id)
            current_param += current_param_i
            current_cycle += current_cycle_i
        return current_param/average_time, current_cycle/average_time


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int, default=2, help='model_id, 1 ~ 2')
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size.')
    parser.add_argument('--input_features', type=int, default=3072, help='flattened input dimension.')
    parser.add_argument('--out_name', type=str, default="dim", help='lib_name for save')
    parser.add_argument('--widen_list', type=str, default="None", help='')
    parser.add_argument('--decompo_list', type=str, default="None", help='')
    parser.add_argument('--dummy_list', type=str, default="None", help='')
    parser.add_argument('--deepen_list', type=str, default="None", help='')
    parser.add_argument('--skipcon_list', type=str, default="None", help='')
    parser.add_argument('--kerneladd_list', type=str, default="None", help='')
    parser.add_argument('--fuse_list', type=str, default="None", help='')
    parser.add_argument('--budget', type=float, default=0.02, help='Time budget to do the obfuscation')
    parser.add_argument('--n_generation', type=int, default=20, help='number of geenration for GA')
    parser.add_argument('--n_pop', type=int, default=16, help='number of population for GA')
    parser.add_argument('--run_option', type=str, default='random')
    args = parser.parse_args()

    torch.manual_seed(1234)
    log_file = "./obfuscate_dim_model{}_{}.log".format(args.model_id, datetime.today().strftime('%m%d%H%M'))
    obf_logger = setup_logger('obf_logger', log_file, level = logging.DEBUG, console_out = True)


    env1 = dim_obf_env(args.batch_size, args.input_features, args.model_id, args.out_name, obf_logger)

    clean_param, clean_cycles = env1.get_3_avg_param()


    best_reward = 0.0
    best_dict = {}
    best_prediction = []
    best_cycle = 0
    run_style = args.run_option
    num_iter = 200
    obf_logger.debug(f'Running Style is {run_style}, Clean Cycle is: {clean_cycles}, Clean param is: {str(clean_param)}. ')
    if run_style == "random":
        for i in range(num_iter):
            env1.random_act_dict_gen([0.6, 0.6, 0.6, 0.6])
            # current_prediction = env1.apply_dict()
            current_param, current_cycle = env1.get_3_avg_param()
            reward = reward_function(current_param, clean_param)
            if reward >= best_reward:
                best_reward = reward
                best_dict = env1.get_obf_dict()
                best_prediction = current_param
                best_cycle = current_cycle
            obf_logger.debug(f"Iteration {i} - Current Reward: {reward}, Cycle: {current_cycle}, Action: {str(env1.get_obf_dict())}, Param: {str(current_param)}")
        obf_logger.debug(f"Iteration {i} - Best Reward: {best_reward}, Cycle: {best_cycle}, Action: {str(best_dict)}, Param: {str(best_prediction)}")
    elif run_style == "grid_search":
        act_dict = {}
        # Grid search on widen_list
        act_dict = {'widen_list': [1.0, 1.0, 1.0], 'dummy_list': [0, 0, 0], 'kerneladd_list': [0, 0, 0], 'prune_list': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        widen_list = [1.0, 1.125, 1.25, 1.375, 1.5]
        for i in range(len(widen_list)):
            for j in range(len(widen_list)):
                act_dict['widen_list'][0] = widen_list[i]
                act_dict['widen_list'][1] = widen_list[j]
                env1.assign_dict(act_dict)
                current_param, current_cycle = env1.get_3_avg_param()
                reward = reward_function(current_param, clean_param)
                obf_logger.debug(f"Iteration 0 - Current Reward: {reward}, Cycle: {current_cycle}, Action: {str(env1.get_obf_dict())}, Param: {str(current_param)}")
        # Grid search on dummy_list
        act_dict = {'widen_list': [1.0, 1.0, 1.0], 'dummy_list': [0, 0, 0], 'kerneladd_list': [0, 0, 0], 'prune_list': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        dummy_list = [1, 2, 3, 5, 10, 20]
        for i in range(len(dummy_list)):
            act_dict['dummy_list'][0] = dummy_list[i]
            env1.assign_dict(act_dict)
            current_param, current_cycle = env1.get_3_avg_param()
            reward = reward_function(current_param, clean_param)
            obf_logger.debug(f"Iteration 0 - Current Reward: {reward}, Cycle: {current_cycle}, Action: {str(env1.get_obf_dict())}, Param: {str(current_param)}")
        # Grid search on kerneladd_list
        act_dict = {'widen_list': [1.0, 1.0, 1.0], 'dummy_list': [0, 0, 0], 'kerneladd_list': [0, 0, 0], 'prune_list': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        kerneladd_list = [0, 1, 2, 3]
        for i in range(len(kerneladd_list)):
            for j in range(len(kerneladd_list)):
                act_dict['kerneladd_list'][0] = kerneladd_list[i]
                act_dict['kerneladd_list'][1] = kerneladd_list[j]
                env1.assign_dict(act_dict)
                if act_dict['kerneladd_list'][1] != 0:
                    current_param, current_cycle = env1.get_3_avg_param(1, 2)
                else:
                    current_param, current_cycle = env1.get_3_avg_param()
                reward = reward_function(current_param, clean_param)
                obf_logger.debug(f"Iteration 0 - Current Reward: {reward}, Cycle: {current_cycle}, Action: {str(env1.get_obf_dict())}, Param: {str(current_param)}")

        # Random search on prune_list (grid-search space is too large)
        for i in range(100):
            env1.random_act_dict_gen([0.0, 0.0, 0.0, 0.6])
            current_param, current_cycle = env1.get_3_avg_param()
            reward = reward_function(current_param, clean_param)
            obf_logger.debug(f"Iteration 0 - Current Reward: {reward}, Cycle: {current_cycle}, Action: {str(env1.get_obf_dict())}, Param: {str(current_param)}")

    elif run_style == "provide":
        for i in range(1):
            act_dict = {'widen_list': [1.0, 1.125, 5.5], 'dummy_list': [3, 0, 2], 'kerneladd_list': [0, 0, 0], 'prune_list': [3, 1, 0, 2, 3, 0, 0, 0, 1, 3, 1, 0, 2, 0]}
            env1.selected_entry = 2
            env1.assign_dict(act_dict)
            current_param, current_cycle = env1.get_3_avg_param()
            reward = reward_function(current_param, clean_param)
            obf_logger.debug(f"Iteration 0 - Current Reward: {reward}, Cycle: {current_cycle}, Action: {str(env1.get_obf_dict())}, Param: {str(current_param)}")

    elif run_style == "genetic_algorithm":
        '''Settings'''
        npop = args.n_pop
        n_generation = args.n_generation
        budget = args.budget
        n_mating = int(npop/2)
        initialize_list = [0.6, 0.6, 0.6, 0.6]
        if_fixed_fuse = True
        mutation_sigma = 4.0
        mutation_forbid_list = []
        parent_list = []
        fitness_list = []
        fitness = np.zeros(npop)
        obf_logger.debug("New Run: Genetic Algorithm of population {}, generation {} and mating {}".format(npop, n_generation, n_mating))
        obf_logger.debug("Time Budget is {}".format(budget))
        obf_logger.debug("Initialize with {}".format(str(initialize_list)))

        for j in range(npop-2):
            env1.random_act_dict_gen(initialize_list)
            w = env1.get_obf_dict()
            parent_list.append(w)
        best_dict = {'widen_list': [1.0, 1.0, 1.0], 'dummy_list': [1, 0, 0], 'kerneladd_list': [0, 0, 0], 'prune_list': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]}
        parent_list = mutation(parent_list, mutation_sigma, mutation_forbid_list)
        parent_list.append(best_dict)
        parent_list.append(best_dict)
        for generation in range(n_generation):
            obf_logger.debug("\nGeneration: " + str(generation))
            if generation % 4 == 0 and generation != 0:
                mutation_sigma = mutation_sigma/2
                obf_logger.debug("Mutation Sigma decay to: " + str(mutation_sigma))
            fitness_list, cycle_list, DER_list, param_list = cal_pop_fitness(env1, parent_list, budget)
            fitness_array = np.asarray(fitness_list)
            max_index = np.where(fitness_array == np.max(fitness_array))
            max_index = max_index[0][0]
            best_cycle = cycle_list[max_index]
            best_score = fitness_list[max_index]
            best_DER = DER_list[max_index]
            best_param = param_list[max_index]
            obf_logger.debug("\nBest Fitness Score so far is: " + str(best_score) + ", its Latency is: " + str(best_cycle) + ", its avg_DER is: " + str(best_DER) + ", its prediction is: " + str(best_param))

            parent_string = ""
            for i in range(npop):
                parent_string += ("\n[{}] Fitness Score: ".format(i) + str(fitness_list[i]) + "; avgDER: {}; Cycle: {}".format(DER_list[i], cycle_list[i]) + "; Action: " + str(parent_list[i]) + "; Prediction: " + str(param_list[i]) + "\n")
            obf_logger.debug(parent_string)
            parents = select_mating_pool(parent_list, fitness_list, n_mating)
            offspring_crossover = crossover(parents, npop - n_mating)
            offspring_mutation = mutation(offspring_crossover, mutation_sigma, mutation_forbid_list)
            parent_list[:len(parents)] = parents
            parent_list[len(parents):] = offspring_mutation
    else:
        pass



if __name__ == '__main__':
    main()
