import numpy as np
import logging
import tvm
from tvm import relay
logging.getLogger('autotvm').setLevel(logging.FATAL)
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import argparse
from torch.jit import TracerWarning
import warnings
warnings.filterwarnings("ignore", category=TracerWarning)
import os
import pandas as pd
import shutil
import GPUtil
import importlib
import json
np.random.seed(1234)

def csv_to_time_overhead(csv_file, torch_trace = False):
    df = pd.read_csv(csv_file, skiprows=2)
    # df = df.drop_duplicates(subset=['ID'])
    # print(df)
    trace_df = df[df['Metric Name'] == "Cycles"]
    trace_df= trace_df.replace(',','', regex=True)
    trace_df['Metric Value'] = pd.to_numeric(trace_df['Metric Value'])
    cost = trace_df['Metric Value'].sum()
    if torch_trace:
        cost = cost/2
    return cost

def prune_old_tasks(tasks, log_file):
    if os.path.isfile(log_file):
        new_tasks = []
        history = autotvm.record.ApplyHistoryBest(log_file)
        for task in tasks:
            if history._query_inside(task.target, task.workload) is None:
                if "dense_small_batch.cuda" in str(task.name) and "10" not in str(task.workload):
                    continue
                new_tasks.append(task)
            # else:
                # print(task.target)
                # print(task.workload)
        return new_tasks
    else:
        return tasks

def do_tune(tasks, old_tasks, log_filename, n_trial = 20, tuner = 'xgb'):
    tmp_log_file = log_filename + ".tmp"
    tuner = 'xgb'
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        '''Template Path: tvm/python/tvm/topi/cuda/con2d_winograd.py and con2d_direct.py'''
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
        print("No tuning Task Found!")

def process_task_string(task_string):
    task_string = task_string.replace("'", '"')
    task_string = task_string.replace('(', '[')
    task_string = task_string.replace(')', ']')
    task_string = task_string.replace('None', 'null')
    task_string = task_string.replace('cuda", [', 'cuda", [[')
    task_string = task_string[1:]
    return task_string

def is_square(apositiveint):
  x = apositiveint // 2
  seen = set([x])
  while x * x != apositiveint:
    x = (x + (apositiveint // x)) // 2
    if x in seen: return False
    seen.add(x)
  return True

def prune_tvm_dict(dict, prune_list):

    if len(dict["entity"]) == 1:
        choice_list = [64, 32, 16]
        if prune_list[0] > 0 and prune_list[0] <= len(choice_list): # prune_list[0] is to select one of the tiling of Dense, min: 0, do nothing, max: 4, set threshold = 64
            if dict["entity"][0][2][1] > choice_list[prune_list[0] - 1]:
                dict["entity"][0][2][1] = choice_list[prune_list[0] - 1]

    elif len(dict["entity"]) == 6:

        dim_y = np.prod(dict["entity"][1][2][1:])
        dividable = True
        if is_square(dim_y):
            prod_a = int(np.sqrt(dim_y))
            prod_b = prod_a
        elif is_square(2*dim_y):
            prod_a = int(np.sqrt(2*dim_y))
            prod_b = int(np.sqrt(2*dim_y)/2)
        else:
            dividable = False
        if prune_list[1] != 0 and dividable: # prune_list[1] is to select one of the tile_y dim to be 1, make others to be the approximate square root of dim.
            dict["entity"][1][2] = [-1, 1, prod_a, prod_b]


        dim_x = np.prod(dict["entity"][2][2][1:])
        dividable = True
        if is_square(dim_x):
            prod_a = int(np.sqrt(dim_x))
            prod_b = prod_a
        elif is_square(2*dim_x):
            prod_a = int(np.sqrt(2*dim_x))
            prod_b = int(np.sqrt(2*dim_x)/2)
        else:
            dividable = False
        if prune_list[2] != 0 and dividable: # prune_list[2] is to select one of the tile_x dim to be 1, make others to be the approximate square root of dim.
            dict["entity"][2][2] = [-1, 1, prod_a, prod_b]


        choice_list = [32, 16, 8]
        if prune_list[3] > 0 and prune_list[3] <= len(choice_list): # prune_list[3] is to select one of the tiling, and assign to the last tile_rc dim, min: 0, do nothing, max: 4, set threshold = 16
            if dict["entity"][3][2][1] > choice_list[prune_list[3] - 1]:
                dict["entity"][3][2][1] = choice_list[prune_list[3] - 1]

        unroll_max_step = [0, 128, 1500]
        if prune_list[4] > 0 and prune_list[4] <= len(unroll_max_step): # prune_list[4] is to select one of the tiling of Dense, min: 0, do nothing; max: 3.
            dict["entity"][4][2] = unroll_max_step[prune_list[4]-1]

        unroll_explicit = [0, 1]
        if prune_list[5] > 0 and prune_list[5] <= len(unroll_explicit): # prune_list[5] is to select one of the tiling of Dense, min: 0, do nothing; max: 2.
            dict["entity"][5][2] = unroll_explicit[prune_list[5]-1]

    elif len(dict["entity"]) == 8:

        dim_f = np.prod(dict["entity"][0][2][1:])
        dividable = True
        if is_square(dim_f):
            prod_a = int(np.sqrt(dim_f))
            prod_b = prod_a
        elif is_square(2*dim_f):
            prod_a = int(np.sqrt(2*dim_f))
            prod_b = int(np.sqrt(2*dim_f)/2)
        else:
            dividable = False
        if prune_list[6] != 0 and dividable: # prune_list[6] is to select one of the tile_y dim to be 1, make others to be the approximate square root of dim.
            dict["entity"][0][2] = [-1, 1, prod_a, prod_b]


        dim_y = np.prod(dict["entity"][1][2][1:])
        dividable = True
        if is_square(dim_y):
            prod_a = int(np.sqrt(dim_y))
            prod_b = prod_a
        elif is_square(2*dim_y):
            prod_a = int(np.sqrt(2*dim_y))
            prod_b = int(np.sqrt(2*dim_y)/2)
        else:
            dividable = False
        if prune_list[7] != 0 and dividable: # prune_list[7] is to select one of the tile_y dim to be 1, make others to be the approximate square root of dim.
            dict["entity"][1][2] = [-1, 1, prod_a, prod_b]


        dim_x = np.prod(dict["entity"][2][2][1:])
        dividable = True
        if is_square(dim_x):
            prod_a = int(np.sqrt(dim_x))
            prod_b = prod_a
        elif is_square(2*dim_x):
            prod_a = int(np.sqrt(2*dim_x))
            prod_b = int(np.sqrt(2*dim_x)/2)
        else:
            dividable = False
        if prune_list[8] != 0 and dividable: # prune_list[8] is to select one of the tile_x dim to be 1, make others to be the approximate square root of dim.
            dict["entity"][2][2] = [-1, 1, prod_a, prod_b]


        choice_list = [32, 16, 8]
        if prune_list[9] > 0 and prune_list[9] <= len(choice_list): # prune_list[9] is to select one of the tiling, and assign to the last tile_rc dim, min: 0, do nothing, max: 4, set threshold = 16
            if dict["entity"][3][2][1] > choice_list[prune_list[9] - 1]:
                dict["entity"][3][2][1] = choice_list[prune_list[9] - 1]


        choice_list = [32, 16, 8]
        if prune_list[10] > 0 and prune_list[10] <= len(choice_list): # prune_list[10] is to select one of the tiling, and assign to the last tile_ry dim, min: 0, do nothing, max: 4, set threshold = 16
            if dict["entity"][4][2][1] > choice_list[prune_list[10] - 1]:
                dict["entity"][4][2][1] = choice_list[prune_list[10] - 1]


        choice_list = [32, 16, 8]
        if prune_list[11] > 0 and prune_list[11] <= len(choice_list): # prune_list[11] is to select one of the tiling, and assign to the last tile_rx dim, min: 0, do nothing, max: 4, set threshold = 16
            if dict["entity"][5][2][1] > choice_list[prune_list[11] - 1]:
                dict["entity"][5][2][1] = choice_list[prune_list[11] - 1]


        unroll_max_step = [0, 128, 1500]
        if prune_list[12] > 0 and prune_list[12] <= len(unroll_max_step): # prune_list[12] is to select one of the tiling of Dense, min: 0, do nothing; max: 3.
            dict["entity"][6][2] = unroll_max_step[prune_list[12]-1]

        unroll_explicit = [0, 1]
        if prune_list[13] > 0 and prune_list[13] <= len(unroll_explicit): # prune_list[13] is to select one of the tiling of Dense, min: 0, do nothing; max: 2.
            dict["entity"][7][2] = unroll_explicit[prune_list[13]-1]


    return dict

def prune_tvm(tasks, tvm_log_file, prune_list):

    prune_list = [int(i) for i in prune_list]

    with open(tvm_log_file, "r") as in_file:
        buf = in_file.readlines()


    # for task in tasks:
        # print(task.name)
    new_file = tvm_log_file.split(".log")[0] + "_pruned.log"
    with open(new_file, "w") as out_file:
        if all(v == 0 for v in prune_list):
            print("Nothing to Apply!")
            for line in buf:
                out_file.write(line)
        else:
            for line in buf:
                if next((True for task in tasks if process_task_string(str(task.workload)) in line), False):
                    # print("line_id is", line_id)
                    splited_set1 = line.split(', {}], "config": ')
                    splited_set2 = splited_set1[1].split(', "result":')
                    # splited_set3 = splited_set1[0].split('.cuda", ')
                    # print(splited_set2[0])
                    # param_list = json.loads(splited_set3[1])
                    dict = json.loads(splited_set2[0])
                    # dict = prune_tvm_dict(dict, param_list, prune_list)
                    dict = prune_tvm_dict(dict, prune_list)
                    '''reform line'''
                    # line = splited_set3[0] + '.cuda", ' + str(param_list).replace("None", "null").replace("'", '"') + ', {}], "config": ' + str(dict).replace("None", "null").replace("'", '"') + ', "result":' +splited_set2[1]
                    line = splited_set1[0] + ', {}], "config": ' + str(dict).replace("None", "null").replace("'", '"') + ', "result":' +splited_set2[1]
                    # print(line)
                out_file.write(line)
    return new_file


def torch_relay_func(args, compare_torch = False, autotvm_on = True):
    assert torch.cuda.is_available()
    cuda_device = torch.device("cuda")  # device object representing GPU

    batch_size = args.batch_size
    input_features = args.input_features
    widen_list = args.widen_list
    decompo_list = args.decompo_list
    dummy_list = args.dummy_list
    deepen_list = args.deepen_list
    skipcon_list = args.skipcon_list
    kerneladd_list = args.kerneladd_list
    fuse_list = args.fuse_list
    print("Fuse List is" + str(fuse_list))
    prune_list = args.prune_list


    # This is the opt_level for the vanilla trace
    opt_level = 3

    X = torch.randn(batch_size, input_features, device=cuda_device)
    model_id = args.model_id
    model_name = "model_{}_obf".format(model_id+1)
    mod = importlib.import_module("." + model_name, package="model_file")
    cnn = getattr(mod, "custom_cnn_{}".format(model_id))

    #step 1 receive the two lists and apply them to the model function
    print("Apply High-level Graph Obfuscation:")
    model = cnn(input_features, True, widen_list, decompo_list, dummy_list, deepen_list, skipcon_list, kerneladd_list).to(cuda_device)



    #step 2 load the obfuscated model parameters
    state_dict = torch.load("./model_file/{}.pickle".format(model_name))
    model.load_state_dict(state_dict)

    model.eval()
    if compare_torch:
        torch.save(model, "./deploy_lib/torch_lib_{}_{}.tar".format(model_name, args.out_name))

    scripted_model = torch.jit.trace(model, X).eval()

    del model
    torch.cuda.empty_cache()

    input_name = "data"
    shape_list = [(input_name, X.size())]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

    target = tvm.target.cuda()

    '''Tuning Option'''

    if args.run_style == "test_tuner":
        tvm_log_file = "./obf_tmp_file/autotvm_{}_{}_{}.log".format(model_name, args.tuner, args.n_trial)
        if not os.path.exists(tvm_log_file):
            open(tvm_log_file, 'a').close()
    else:
        tvm_log_file = "./obf_tmp_file/autotvm_%s.log" % model_name

        if not os.path.exists(tvm_log_file):
            # make a copy of the pretuned-file to work with
            if "GTX 1660 SUPER" in GPUtil.getGPUs()[0].name:
                pretuned_file="../trace_gen/gtx1660super/autotvm_tracegen.log"
                shutil.copy(pretuned_file, tvm_log_file)
            elif "RTX 3090" in GPUtil.getGPUs()[0].name:
                pretuned_file="../trace_gen/rtx3090/autotvm_tracegen.log"
                shutil.copy(pretuned_file, tvm_log_file)
            else:
                print("Not pretune on your GPU type, generate new")
                open(tvm_log_file, 'a').close()




    if not autotvm_on:
        tvm_log_file = "./obf_tmp_file/faketvm_%s.log" % model_name
        if not os.path.exists(tvm_log_file):
            open(tvm_log_file, 'a').close()
    else:

        print("Extract tasks...")
        tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

        print("Prune old tasks...")
        old_tasks = tasks
        tasks = prune_old_tasks(tasks, tvm_log_file)

        print("Tuning...")
        do_tune(tasks, old_tasks, tvm_log_file, args.n_trial, args.tuner)

        print("Apply Backend Obfuscator...")
        '''Prune the autotvm results [as an obfuscation]'''
        tvm_log_pruned = prune_tvm(old_tasks, tvm_log_file, prune_list)

    #step 3 apply the fuse through relay backend
    max_fuse_depth = 99

    with autotvm.apply_history_best(tvm_log_pruned):
        print("Apply Defusion:")
        with tvm.transform.PassContext(opt_level=opt_level, config={"relay.FuseOps.max_depth": max_fuse_depth, "relay.FuseOps.list_length": len(fuse_list), "relay.FuseOps.allow_fuse" : fuse_list}):
            lib = relay.build(mod, target, params=params)
    lib.export_library("./deploy_lib/lib_{}_{}.tar".format(model_name, args.out_name))
    lib_name = "lib_{}_{}".format(model_name, args.out_name)

    os.system('ncu --target-processes application-only --print-kernel-base function --log-file ./env_file/{}.csv --csv --kernel-regex-base function --launch-skip-before-match 0  --profile-from-start 1 --clock-control base --print-fp --apply-rules yes --section ImportantTraceAnalysis python torch_relay_test.py --batch_size {} --input_features {} --lib_name {}'.format(lib_name, args.batch_size, args.input_features, lib_name))
    os.system('nvprof -f -o obf_tmp_file/{}.sql --profile-from-start on -- python torch_relay_test.py --batch_size {} --input_features {} --lib_name {}'.format(lib_name, args.batch_size, args.input_features, lib_name))
    # nvprof -f -o net.sql --profile-from-start off -- python xxx.py
    if compare_torch:

        os.system('ncu --target-processes application-only --print-kernel-base function --log-file ./env_file/torch_{}.csv --csv --kernel-regex-base function --launch-skip-before-match 0  --profile-from-start 1 --clock-control base --print-fp --apply-rules yes --section ImportantTraceAnalysis python torch_test.py --batch_size {} --input_features {} --lib_name {}'.format(lib_name, args.batch_size, args.input_features, lib_name))

        time_cost_torch = csv_to_time_overhead("./env_file/torch_{}.csv".format(lib_name), True)

        time_cost_tvm = csv_to_time_overhead("./env_file/{}.csv".format(lib_name))

        print("torch time cycle =",  time_cost_torch)

        print("tvm time cycle =",  time_cost_tvm)
