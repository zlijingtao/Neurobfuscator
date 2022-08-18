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

from func_gen_cifar import layer_name_to_int_map
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
        print(label_array)
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


def extrac_oc_from_conv(conv_name):
    if "res" in conv_name:
        return int(conv_name.split("p")[0].replace("res", ""))
    elif "depth" in conv_name:
        return int(conv_name.split("p")[1].replace("th", ""))
    elif "bn" in conv_name:
        return 0
    else:
        return int(conv_name.split("p")[0])

def extrac_others_from_conv(conv_name):
    if "res" in conv_name:
        kernel_size = 3
        stride_size = 1
        padding_size = 1
    elif "depth" in conv_name:
        kernel_size = 3
        stride_size = 1
        padding_size = 1
    else:
        kernel_size = int(conv_name.split("p")[1])
        stride_size = int(conv_name.split("p")[2])
        padding_size = int(conv_name.split("p")[3])

    return kernel_size, stride_size, padding_size

def change_img_dim(conv_name, img_dimension):
    if "res" not in conv_name:
        stride_size = int(conv_name.split("p")[2])
        if stride_size == 2:
            img_dimension = int(img_dimension/2)
        if_pool = True if int(conv_name.split("p")[4]) > 0 else False
        if if_pool:
            img_dimension = int(img_dimension/2)
    return img_dimension

def panda_add_entry(dir, seg_table, train_inputs, file_prefix, panda_dict, orig_img_dim, operator_name = 'conv', n_class = 10):
    mstring_path = dir + file_prefix + ".mstring"
    npy_path = dir + file_prefix + ".npy"
    label_array = np.load(npy_path)
    label_array = fuse_label(label_array, no_repeat = False)
    my_file = open(mstring_path, "r")
    content_list = my_file.readlines()
    model_name = content_list[0]

    if operator_name == "conv":
        model_name_split = model_name.split("_mlp")[0].split("_")
        count = 1
        pre_conv = "3p"
        img_dimension = orig_img_dim
        for i in range(len(seg_table)):
            if label_array[i] != 0 and label_array[i] != 4:
                continue
            #Deal with the conv layer
            conv_name = model_name_split[int(np.floor(count))]
            input_channels = extrac_oc_from_conv(pre_conv)
            pre_conv = conv_name
            if label_array[i] == 4:
                count += 1
                continue
            output_channels = extrac_oc_from_conv(conv_name)
            kernel, stride, pad = extrac_others_from_conv(conv_name)

            # print("count {} - conv name is: {}, Image_Dim = {}, IC = {}, OC = {}".format(count, conv_name, img_dimension, input_channels, output_channels))


            # If current layer is a reslayer, then next one is still this layer.
            if "res" in model_name_split[int(np.floor(count))]:
                count += 0.5
            # If current layer is a normal conv layer, count += 1
            else:
                count += 1
            
            if train_inputs.shape[2] == 1:
                panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])

            elif train_inputs.shape[2] == 5:
                panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])
                panda_dict['FeatureMemRead'].append(train_inputs[0, int(seg_table[i]), 1])
                panda_dict['FeatureMemWrite'].append(train_inputs[0, int(seg_table[i]), 2])
                panda_dict['FeatureMemRWratio'].append(train_inputs[0, int(seg_table[i]), 3])
                panda_dict['FeatureMemRAWratio'].append(train_inputs[0, int(seg_table[i]), 4])
            elif train_inputs.shape[2] == 11:
                panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])
                panda_dict['FeatureMemRead'].append(train_inputs[0, int(seg_table[i]), 1])
                panda_dict['FeatureMemWrite'].append(train_inputs[0, int(seg_table[i]), 2])
                panda_dict['FeatureL1Util'].append(train_inputs[0, int(seg_table[i]), 3])
                panda_dict['FeatureL1Hit'].append(train_inputs[0, int(seg_table[i]), 4])
                panda_dict['FeatureL1Read'].append(train_inputs[0, int(seg_table[i]), 5])
                panda_dict['FeatureL1Write'].append(train_inputs[0, int(seg_table[i]), 6])
                panda_dict['FeatureL2Hit'].append(train_inputs[0, int(seg_table[i]), 7])
                panda_dict['FeatureL2Util'].append(train_inputs[0, int(seg_table[i]), 8])
                panda_dict['FeatureL2Read'].append(train_inputs[0, int(seg_table[i]), 9])
                panda_dict['FeatureL2Write'].append(train_inputs[0, int(seg_table[i]), 10])
            panda_dict['FeatureImageDim'].append(img_dimension)
            panda_dict['TargetIC'].append(input_channels)
            panda_dict['TargetOC'].append(output_channels)
            panda_dict['TargetKernel'].append(kernel)
            panda_dict['TargetStride'].append(stride)
            panda_dict['TargetPad'].append(pad)

            img_dimension = change_img_dim(conv_name, img_dimension)
    
    elif operator_name == "fc":
        model_name_split = model_name.split("_mlp")[0].split("_")
        img_dimension = orig_img_dim
        count = 1
        for i in range(len(seg_table)):
            if label_array[i] != 0 and label_array[i] != 4:
                continue
            #Deal with the conv layer
            
            conv_name = model_name_split[int(np.floor(count))]
            if label_array[i] == 4:
                count += 1
                continue
            output_channels = extrac_oc_from_conv(conv_name)
            if "res" in model_name_split[int(np.floor(count))]:
                count += 0.5
            # If current layer is a normal conv layer, count += 1
            else:
                count += 1
            
            img_dimension = change_img_dim(conv_name, img_dimension)
            
        model_name_split = model_name.split("_mlp_")[1].split("_bn")[0].split("_")
        model_name_split.append(f'{n_class}')
        feature_dim = img_dimension ** 2 * output_channels
        
        count = 0
        for i in range(len(seg_table)):
            if label_array[i] != 1:
                continue

            if train_inputs.shape[2] == 1:
                panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])

            elif train_inputs.shape[2] == 5:
                panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])
                panda_dict['FeatureMemRead'].append(train_inputs[0, int(seg_table[i]), 1])
                panda_dict['FeatureMemWrite'].append(train_inputs[0, int(seg_table[i]), 2])
                panda_dict['FeatureMemRWratio'].append(train_inputs[0, int(seg_table[i]), 3])
                panda_dict['FeatureMemRAWratio'].append(train_inputs[0, int(seg_table[i]), 4])
            elif train_inputs.shape[2] == 11:
                panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])
                panda_dict['FeatureMemRead'].append(train_inputs[0, int(seg_table[i]), 1])
                panda_dict['FeatureMemWrite'].append(train_inputs[0, int(seg_table[i]), 2])
                panda_dict['FeatureL1Util'].append(train_inputs[0, int(seg_table[i]), 3])
                panda_dict['FeatureL1Hit'].append(train_inputs[0, int(seg_table[i]), 4])
                panda_dict['FeatureL1Read'].append(train_inputs[0, int(seg_table[i]), 5])
                panda_dict['FeatureL1Write'].append(train_inputs[0, int(seg_table[i]), 6])
                panda_dict['FeatureL2Hit'].append(train_inputs[0, int(seg_table[i]), 7])
                panda_dict['FeatureL2Util'].append(train_inputs[0, int(seg_table[i]), 8])
                panda_dict['FeatureL2Read'].append(train_inputs[0, int(seg_table[i]), 9])
                panda_dict['FeatureL2Write'].append(train_inputs[0, int(seg_table[i]), 10])

            panda_dict['FeatureDim'].append(feature_dim)
            panda_dict['TargetDim'].append(int(model_name_split[count]))
            feature_dim = int(model_name_split[count])
            count += 1
    
    elif operator_name == "depth":
        model_name_split = model_name.split("_mlp")[0].split("_")
        count = 1
        pre_conv = "3p"
        img_dimension = orig_img_dim
        for i in range(len(seg_table)):
            if label_array[i] != 0 and label_array[i] != 4:
                continue
            #Deal with the conv layer
            conv_name = model_name_split[int(np.floor(count))]
            input_channels = extrac_oc_from_conv(pre_conv)
            pre_conv = conv_name
            # if label_array[i] == 4:
            output_channels = extrac_oc_from_conv(conv_name)
            kernel, stride, pad = extrac_others_from_conv(conv_name)

            # print("count {} - conv name is: {}, Image_Dim = {}, IC = {}, OC = {}".format(count, conv_name, img_dimension, input_channels, output_channels))


            # If current layer is a reslayer, then next one is still this layer.
            if "res" in model_name_split[int(np.floor(count))]:
                count += 0.5
            # If current layer is a normal conv layer, count += 1
            else:
                count += 1
            
            if label_array[i] == 4:
                if train_inputs.shape[2] == 1:
                    panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])

                elif train_inputs.shape[2] == 5:
                    panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])
                    panda_dict['FeatureMemRead'].append(train_inputs[0, int(seg_table[i]), 1])
                    panda_dict['FeatureMemWrite'].append(train_inputs[0, int(seg_table[i]), 2])
                    panda_dict['FeatureMemRWratio'].append(train_inputs[0, int(seg_table[i]), 3])
                    panda_dict['FeatureMemRAWratio'].append(train_inputs[0, int(seg_table[i]), 4])
                elif train_inputs.shape[2] == 11:
                    panda_dict['FeatureCycles'].append(train_inputs[0, int(seg_table[i]), 0])
                    panda_dict['FeatureMemRead'].append(train_inputs[0, int(seg_table[i]), 1])
                    panda_dict['FeatureMemWrite'].append(train_inputs[0, int(seg_table[i]), 2])
                    panda_dict['FeatureL1Util'].append(train_inputs[0, int(seg_table[i]), 3])
                    panda_dict['FeatureL1Hit'].append(train_inputs[0, int(seg_table[i]), 4])
                    panda_dict['FeatureL1Read'].append(train_inputs[0, int(seg_table[i]), 5])
                    panda_dict['FeatureL1Write'].append(train_inputs[0, int(seg_table[i]), 6])
                    panda_dict['FeatureL2Hit'].append(train_inputs[0, int(seg_table[i]), 7])
                    panda_dict['FeatureL2Util'].append(train_inputs[0, int(seg_table[i]), 8])
                    panda_dict['FeatureL2Read'].append(train_inputs[0, int(seg_table[i]), 9])
                    panda_dict['FeatureL2Write'].append(train_inputs[0, int(seg_table[i]), 10])
                panda_dict['FeatureImageDim'].append(img_dimension)
                panda_dict['TargetIC'].append(input_channels)
                panda_dict['TargetOC'].append(output_channels)
                panda_dict['TargetStride'].append(stride)

            else:
                img_dimension = change_img_dim(conv_name, img_dimension)
    
    
    return panda_dict


def collect_feature_target(dir, operator_name = 'conv', reduced = True, time_only = False, include_name = "complex_batch_1_nclass_10_infeat_3072", prefix_output = "train_data", orig_img_dim = 32):
    train_inputs_list = []
    train_targets_sparse_list = []
    train_seq_len_list = []
    index_list = []
    original_list = []
    index = 0
    panda_dict = {}
    n_class = int(include_name.split("nclass_")[-1].split("_infeat")[0])
    if reduced:
        if time_only:
            panda_dict['FeatureCycles'] = []
        else:
            panda_dict['FeatureCycles'] = []
            panda_dict['FeatureMemRead'] = []
            panda_dict['FeatureMemWrite'] = []
            panda_dict['FeatureMemRWratio'] = []
            panda_dict['FeatureMemRAWratio'] = []
    else:
        panda_dict['FeatureCycles'] = []
        panda_dict['FeatureMemRead'] = []
        panda_dict['FeatureMemWrite'] = []
        panda_dict['FeatureL1Util'] = []
        panda_dict['FeatureL1Hit'] = []
        panda_dict['FeatureL1Read'] = []
        panda_dict['FeatureL1Write'] = []
        panda_dict['FeatureL2Hit'] = []
        panda_dict['FeatureL2Util'] = []
        panda_dict['FeatureL2Read'] = []
        panda_dict['FeatureL2Write'] = []

    if operator_name == "conv":
        panda_dict['FeatureImageDim'] = []
        panda_dict['TargetIC'] = []
        panda_dict['TargetOC'] = []
        panda_dict['TargetKernel'] = []
        panda_dict['TargetStride'] = []
        panda_dict['TargetPad'] = []
    elif operator_name == "fc":
        panda_dict['FeatureDim'] = []
        panda_dict['TargetDim'] = []
    elif operator_name == "depth":
        panda_dict['FeatureImageDim'] = []
        panda_dict['TargetIC'] = []
        panda_dict['TargetOC'] = []
        panda_dict['TargetStride'] = []
    for filename in os.listdir(dir):
        if include_name in filename:
            file_prefix = filename.split('.')[0]
            file_type = filename.split('.')[1]
            if file_type == 'csv':
                print(filename)
                seg_tuble = trace_csv_seg(dir + filename)
                if seg_tuble is None:
                    continue
                else:
                    seg_table, _ = seg_tuble
                    print(seg_table)
                    if seg_table is None:
                        continue
                if reduced:
                    _, train_inputs = trace_csv_numpy(dir + filename)
                    if time_only:
                        train_inputs = train_inputs[:,:,0].reshape(1, train_inputs.shape[1], 1)
                else:
                    train_inputs, _ = trace_csv_numpy(dir + filename)
                panda_dict = panda_add_entry(dir, seg_table, train_inputs, file_prefix, panda_dict, orig_img_dim, operator_name, n_class)
    df = pd.DataFrame(panda_dict)
    if reduced:
        if time_only:
            df.to_csv(f'../seq_predictor/obfuscator/dataset/timeonly_{operator_name}_classification.csv', index=False) 
        else:
            df.to_csv(f'../seq_predictor/obfuscator/dataset/reduced_{operator_name}_classification.csv', index=False) 
    else:
        df.to_csv(f'../seq_predictor/obfuscator/dataset/full_{operator_name}_classification.csv', index=False) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="full", help='Pick dataset you want to generate', choices=("reduced", "full", "time_only"))
    parser.add_argument("--selection", type=str, default="complex_batch_1_nclass_1000_infeat_150528", help='Type a string, only contains which will be generated')
    parser.add_argument("--prefix_output", type=str, default="cifar10_train_data", help='prefix for the dataset output')
    parser.add_argument("--orig_img_dim", type=int, default=224, help='original image dimension')
    args = parser.parse_args()
    if args.dataset_type == "reduced":
        collect_feature_target('trace/', 'conv', reduced = True, time_only = False, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
        collect_feature_target('trace/', 'fc', reduced = True, time_only = False, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
        collect_feature_target('trace/', 'depth', reduced = True, time_only = False, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
    elif args.dataset_type == "full":
        collect_feature_target('trace/', 'conv', reduced = False, time_only = False, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
        collect_feature_target('trace/', 'fc', reduced = False, time_only = False, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
        collect_feature_target('trace/', 'depth', reduced = False, time_only = False, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
    elif args.dataset_type == "time_only":
        collect_feature_target('trace/', 'conv', reduced = True, time_only = True, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
        collect_feature_target('trace/', 'fc', reduced = True, time_only = True, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
        collect_feature_target('trace/', 'depth', reduced = True, time_only = True, include_name = args.selection, prefix_output = args.prefix_output, orig_img_dim = args.orig_img_dim)
