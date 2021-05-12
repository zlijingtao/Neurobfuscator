import torch
import time
import torch.nn.functional as F
import numpy as np
import argparse
import math
import csv
import pandas as pd
import os
import pickle

from func_generator import layer_name_to_int_map
#Sequential Model Generator. Search space is limited.

def transform_dd(train_inputs, train_targets,seg_table):
    length = train_inputs.shape[1]
    dd = np.zeros(length)
    # for i in range(0,len(train_targets)):
    #     index = int(seg_table[i])
    #     if train_targets[i] == incept_op:
    #         dd[index]=incept_dd
    #     elif train_targets[i] == add_op:
    #         dd[index]=add_dd
    temp = train_inputs[0]
    train_inputs_t = np.column_stack((temp,dd))
    train_inputs = np.array([train_inputs_t])

    return train_inputs

def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

def all_select_seg(train_inputs, train_targets, seg_table):
    train_inputs = transform_dd(train_inputs, train_targets, seg_table)
    roi_inputs = train_inputs
    roi_targets = train_targets
    roi_targets_sparse = sparse_tuple_from([roi_targets])
    return roi_inputs, roi_targets_sparse, roi_targets

# Some configs. (latency, r, w, r/w, i/o)
def trace_csv_numpy(trace_file = "None"):
    full_trace_array = None
    reduced_trace_array = None

    try:
        file1 = open(trace_file, 'r')
        #read the third line, check if the model file is executed without error.
        Lines = file1.readlines()
        file1.close()
    except FileNotFoundError:
        return full_trace_array, reduced_trace_array
    if len(Lines) <= 3:
        trace_file = "None"
    elif "ERROR" in Lines[2]:
        #Then remove that file.
        print(trace_file)
        os.remove(trace_file)
        #Then remove that label.
        trace_prefix = trace_file.split('.')[0]
        label_name = trace_prefix + '.npy'
        # os.remove(label_name)
        #Set trace_file to None
        trace_file = "None"


    if trace_file != "None":
        df = pd.read_csv(trace_file, skiprows=2)
        trace_df = df[['ID', 'Metric Name', 'Metric Value']]
        reduced_trace_array = np.zeros((1, trace_df['ID'].nunique(), 5))
        full_trace_array = np.zeros((1, trace_df['ID'].nunique(), trace_df['Metric Name'].nunique()))
        old_row_id = -1
        count = 0
        # float("123,456.908".replace(',',''))
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
        reduced_trace_array = np.nan_to_num(reduced_trace_array)
    return full_trace_array, reduced_trace_array

def trace_csv_seg(trace_file = "None", fuse = True, trace_type = "tvm", no_repeat = False):
    seg_array = None
    seg_list = []
    try:
        file1 = open(trace_file, 'r')
        #read the third line, check if the model file is executed without error.
        Lines = file1.readlines()
        file1.close()
    except FileNotFoundError:
        return None
    if len(Lines) <= 4:
        trace_file = "None"
    elif "ERROR" in Lines[2] or "ERROR" in Lines[3]:
        #Then remove that file.
        print(trace_file)
        # os.remove(trace_file)
        #Then remove that label.
        trace_prefix = trace_file.split('.')[0]
        label_name = trace_prefix + '.npy'
        # os.remove(label_name)
        #Set trace_file to None
        trace_file = "None"

    # layer_name_to_int_map = {'conv':0, 'fc':1, 'pooling':2, 'bn':3, 'depthConv':4, 'relu':5, 'pointConv':6, 'add':7, 'softmax': 8}
    if trace_file != "None":
        trace_prefix = trace_file.split('.')[0]
        label_name = trace_prefix + '.npy'
        label_array = np.load(label_name)
        # print(label_array)
        if fuse:
            label_array = fuse_label(label_array, no_repeat = no_repeat)
            # np.save(label_name, label_array)
        # print(label_array)
        number_layer = label_array.shape[0]
        # print(layer_name_to_int_map)
        # print(number_layer)
        if trace_type == "tvm":
            layer_indicator = ['fused_nn_conv2d', 'fused_nn_contrib_conv2d', 'fused_nn_max_pool2d', 'fused_nn_avg_pool2d', 'fused_nn_dense', 'fused_nn_log_softmax_1_kernel1', 'fused_nn_log_softmax_kernel1']
            must_include = "kernel0"
        elif trace_type == "pytorch":
            layer_indicator = ['vectorized_elementwise_kernel', 'generateWinogradTilesKernel', 'max_pool_forward_nchw', 'bn_fw_inf_1C11_kernel_NCHW', 'bn_pointwise', 'batch_norm_transform_input_kernel', 'gemv2T_kernel_val', 'softmax_warp_forward']
            must_include = ""
        else:
            layer_indicator = []
            must_include = ""
        if trace_file != "None":
            df = pd.read_csv(trace_file, skiprows=2)
            try:
                df = df.drop_duplicates(subset=['ID'])
            except:
                print("Error File is", trace_file)
            # print(df)
            trace_df = df[['ID', 'Kernel Name']]
            # pd.set_option("display.max_rows", 100)
            # print(trace_df)
            old_row_id = -1
            count = 0
            previous_seg = None
            for index, row in trace_df.iterrows():
                layer_type, if_seg = find_match(row['Kernel Name'], layer_indicator, must_include)
                if if_seg:
                    if previous_seg != None:
                        if no_repeat == True:
                            if layer_type != previous_seg:
                                seg_list.append(str(row['ID']))
                        else:
                            seg_list.append(str(row['ID']))
                    else:
                        seg_list.append(str(row['ID']))
                    previous_seg = layer_type
        if len(seg_list) != number_layer:
            pass
            # print(trace_file)
            # print(seg_list)
            # print(label_array)
        if len(seg_list) == number_layer:
            seg_array = np.asarray(seg_list)
            # print(trace_file)
            # print(seg_list)
            # print(label_array)
            # print(len(seg_list))
            # print(seg_array)
        # print(label_array)
            # np.save(trace_prefix + '.seg', seg_array)
        return (seg_array, label_array)
def fuse_label(label_array, no_repeat = False):
    # layer_name_to_int_map = {'conv':0, 'fc':1, 'pooling':2, 'bn':3, 'depthConv':4, 'relu':5, 'pointConv':6, 'add':7, 'softmax': 8}
    new_label_array = []
    fuse_list = [3, 5]
    previous_layer = None
    for i in range(label_array.shape[0]):
        if label_array[i] not in fuse_list:
            if previous_layer != None:
                if no_repeat == True:
                    if label_array[i] != previous_layer:
                        new_label_array.append(label_array[i])
                else:
                    new_label_array.append(label_array[i])
            else:
                new_label_array.append(label_array[i])
            previous_layer = label_array[i]
    new_label_array = np.asarray(new_label_array)
    return new_label_array

def find_match(string, string_list, must_include):
    must_include_t = must_include
    if "conv" in string:
        layer_type = 0
    elif "dense" in string:
        layer_type = 1
    elif "max_pool" in string:
        layer_type = 2
    elif "softmax" in string:
        layer_type = 8
    else:
        layer_type = 8
    # Make a exception for winograd kernel, mark the second kernel
    if "winograd" in string:
        must_include_t = "kernel1"
    # Otherwise, must_include is "kernel0"
    for item in string_list:
        if item == string:
            return layer_type, True
        elif (item in string) and (must_include_t in string):
            return layer_type, True
    return layer_type, False
def dump_to_pickel(dir, reduced = True, time_only = False, include_name = "batch", prefix_output = "train_data", seed_lower_range = 0 , seed_upper_range = 9999, blend_complex = 0):
    train_inputs_list = []
    train_targets_sparse_list = []
    train_seq_len_list = []
    index_list = []
    original_list = []
    index = 0
    for filename in os.listdir(dir):
        # if seed in filename:
        #     seed_ID = int(filename.split("_")[-1].split(".")[0])
        if blend_complex != 0:
            complex_seed_lower_range = 4000
            complex_seed_upper_range = complex_seed_lower_range + blend_complex
            if "complex_batch" in filename:
                file_prefix = filename.split('.')[0]
                seed_1 = int(file_prefix.split("_")[-1])
                if seed_1 < complex_seed_lower_range or seed_1 > complex_seed_upper_range:
                    continue
                file_type = filename.split('.')[1]
                # print("adding complex batch")
                # for sample in self.sample_list:
                if file_type == 'csv':
                    seg_tuble = trace_csv_seg(dir + filename)
                    if seg_tuble is None:
                        print("Error, no seg_table")
                        continue
                    else:
                        seg_table, train_targets = seg_tuble
                        if seg_table is None:
                            print("Error, no seg_table")
                            continue
                    if reduced:
                        _, train_inputs = trace_csv_numpy(dir + filename)
                        if time_only:
                            train_inputs = train_inputs[:,:,0].reshape(1, train_inputs.shape[1], 1)
                    else:
                        train_inputs, _ = trace_csv_numpy(dir + filename)
                    # train_targets = np.load(dir + file_prefix + '.npy')
                    train_inputs, train_targets_sparse, original = all_select_seg(train_inputs, train_targets, seg_table)  # 0 for scheduler
                    # print(train_inputs.shape[2])
                    train_seq_len = [train_inputs.shape[1]]
                    train_inputs_list.append(train_inputs)
                    train_targets_sparse_list.append(train_targets_sparse)
                    train_seq_len_list.append(train_seq_len)
                    index_list.append(index)
                    index += 1
                    original_list.append(original)
        if include_name in filename:
            file_prefix = filename.split('.')[0]
            seed_0 = int(file_prefix.split("_")[-1])
            if seed_0 < seed_lower_range or seed_0 > seed_upper_range:
                continue
            file_type = filename.split('.')[1]
            # for sample in self.sample_list:
            if file_type == 'csv':
                seg_tuble = trace_csv_seg(dir + filename)
                if seg_tuble is None:
                    print("Error, no seg_table")
                    continue
                else:
                    seg_table, train_targets = seg_tuble
                    if seg_table is None:
                        print("Error, no seg_table")
                        continue
                if reduced:
                    _, train_inputs = trace_csv_numpy(dir + filename)
                    if time_only:
                        train_inputs = train_inputs[:,:,0].reshape(1, train_inputs.shape[1], 1)
                else:
                    train_inputs, _ = trace_csv_numpy(dir + filename)
                # train_targets = np.load(dir + file_prefix + '.npy')
                train_inputs, train_targets_sparse, original = all_select_seg(train_inputs, train_targets, seg_table)  # 0 for scheduler
                # print(train_inputs.shape[2])
                train_seq_len = [train_inputs.shape[1]]
                train_inputs_list.append(train_inputs)
                train_targets_sparse_list.append(train_targets_sparse)
                train_seq_len_list.append(train_seq_len)
                index_list.append(index)
                index += 1
                original_list.append(original)
            # print(file_prefix)
    train_dict = {}
    train_dict['train_inputs_list'] = train_inputs_list
    train_dict['train_targets_sparse_list'] = train_targets_sparse_list
    train_dict['train_seq_len_list'] = train_seq_len_list
    train_dict['index_list'] = index_list
    train_dict['original_list'] = original_list
    # print(train_targets_sparse_list)
    # print(train_inputs_list)
    # print(train_inputs_list[0].shape)
    # print(index_list)
    # print(original_list)
    if reduced:
        if time_only:
            with open('../train_predictor/obfuscator/dataset/{}_dict_time_only.pickle'.format(prefix_output), 'wb') as handle:
                pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open('../train_predictor/obfuscator/dataset/{}_dict.pickle'.format(prefix_output), 'wb') as handle:
                pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('../train_predictor/obfuscator/dataset/{}_dict_full.pickle'.format(prefix_output), 'wb') as handle:
            pickle.dump(train_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="time_only", help='Pick dataset you want to generate', choices=("reduced", "full", "time_only"))
    parser.add_argument("--selection", type=str, default="batch", help='Type a string, only contains which will be generated')
    parser.add_argument("--prefix_output", type=str, default="train_data", help='prefix for the dataset output')
    parser.add_argument("--seed_lower_range", type=int, default=0, help='seed_lower_range')
    parser.add_argument("--seed_upper_range", type=int, default=9999, help='seed_upper_range')
    parser.add_argument("--blend_complex", type=int, default=0, help='complex sample to blend')
    args = parser.parse_args()
    if args.dataset_type == "reduced":
        dump_to_pickel('trace/', reduced = True, time_only = False, include_name = args.selection, prefix_output = args.prefix_output, seed_lower_range = args.seed_lower_range, seed_upper_range = args.seed_upper_range, blend_complex = args.blend_complex)
    elif args.dataset_type == "full":
        dump_to_pickel('trace/', reduced = False, time_only = False, include_name = args.selection, prefix_output = args.prefix_output, seed_lower_range = args.seed_lower_range, seed_upper_range = args.seed_upper_range, blend_complex = args.blend_complex)
    elif args.dataset_type == "time_only":
        dump_to_pickel('trace/', reduced = True, time_only = True, include_name = args.selection, prefix_output = args.prefix_output, seed_lower_range = args.seed_lower_range, seed_upper_range = args.seed_upper_range, blend_complex = args.blend_complex)
