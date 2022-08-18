import numpy as np
import argparse
import math
import logging
import os
#Sequential Model Generator. Search space is limited.


#$0: Number of filter
#$1: Filter size
#$2: Stride size
#$3: padding size
#$4: 0/1: if maxpool
# cnn_16p3p1p1p0_16p3p1p1p1_32p3p1p1p0_bn1_mlp_512_256_128_bn1

layer_name_to_int_map = {'conv':0, 'fc':1, 'pooling':2, 'bn':3, 'depthConv':4, 'relu':5, 'pointConv':6, 'add':7, 'softmax': 8}

def model_name_from_seed(seed):
    np.random.seed(seed)
    min_conv_layer = 4
    min_fc_layer = 1
    max_conv_layer = 12
    max_fc_layer = 4
    min_conv_degree = 4 #Max allowed channel is 2^6
    max_conv_degree = 10 #Max allowed channel is 2^6
    min_fc_degree = 4 #Max allowed neuron is 2^10
    max_fc_degree = 9 #Max allowed neuron is 2^10
    max_num_pool = 5 #Set to 5 does not work for mnist (<=2)!
    num_conv = np.random.randint(low = min_conv_layer, high = max_conv_layer)
    num_fc = np.random.randint(low = min_fc_layer, high = max_fc_layer)
    model_name = "cnn"

    pool_counter = 0
    for i in range(num_conv):
        if i == 0 or i == num_conv - 1:
            #first/last conv layer use normal
            conv_choice = 0
        else:
            #[normal: 0~3, res:4, depth:5]
            #66% chance get normal layer, 16%/16% chance get depth and resblock
            conv_choice = np.random.randint(low = 0, high = 6)
        if conv_choice <= 3:
            #gen_normal_conv2d
            model_name += "_"
            # Channel Size
            channel_size = 2**np.random.randint(low = min_conv_degree, high = max_conv_degree)
            model_name += "{}p".format(channel_size)
            # Filter Size
            model_name += "{}p".format(3)
            # Stride Size
            if (np.random.binomial(size=(1,), n=1, p = 0.2))[0] == 1:
                pool_counter += 1
                if pool_counter >= max_num_pool:
                    model_name += "{}p".format(1)
                else:
                    model_name += "{}p".format(2)
            else:
                model_name += "{}p".format(1)
            # Padding Size
            model_name += "{}p".format(1)
            # Pooling or not [maxpool: 1 & 2, not:0, avgpool:3]
            if_pool = (np.random.randint(4) * np.random.binomial(size=(1,), n=1, p = 0.5))[0]
            if if_pool >= 1:
                pool_counter += 1
                if pool_counter >= max_num_pool:
                    if_pool = 0
            model_name += "{}".format(if_pool)
        elif conv_choice == 4:
            #gen_resblock
            model_name += "_"
            # Channel Size
            model_name += "res{}p".format(channel_size)
            # Stride Size
            model_name += "{}".format(1)
        elif conv_choice == 5:
            #gen_resblock
            model_name += "_"
            # Channel Size
            channel_size = 2**np.random.randint(low = min_conv_degree, high = max_conv_degree)
            model_name += "depth{}p".format(channel_size)
            # Stride Size
            model_name += "{}".format(1)
    model_name += "_bn{}_mlp".format(np.random.randint(2))

    for i in range(num_fc):
        model_name += "_"
        # Neuron Size
        model_name += "{}".format(2**np.random.randint(low = min_fc_degree, high = max_fc_degree))

    model_name += "_bn{}".format(np.random.randint(2))
    return model_name


def mlp_list_from_string(model_name):
    size_list = []
    layer_list = model_name.split("_")
    start = 0
    fc_bn = False
    for i in range(len(layer_list)):
        if "mlp" in layer_list[i]:
            start = 1
        elif start == 1:
            if "bn" in layer_list[i]:
                if "bn1" in layer_list[i]:
                    fc_bn = True
                else:
                    fc_bn = False
            else:
                size_list.append(layer_list[i])

    return size_list, fc_bn

def cnn_list_from_string(model_name):
    size_conv = []
    filter_conv = []
    stride_conv = []
    padding_conv = []
    pool_conv = []
    layer_list = model_name.split("_")
    start = 0
    conv_bn = False
    for i in range(len(layer_list)):
        if "cnn" in layer_list[i]:
            start = 1
        elif "mlp" in layer_list[i]:
            start = 0
        elif start == 1:
            if "res" in layer_list[i]:
                res_list = layer_list[i].split("p")
                size_conv.append(res_list[0])
                filter_conv.append('3')
                stride_conv.append(res_list[1])
                padding_conv.append('1')
                pool_conv.append('0')
            elif "depth" in layer_list[i]:
                res_list = layer_list[i].split("p")
                size_conv.append(res_list[1])
                filter_conv.append('3')
                stride_conv.append(res_list[2])
                padding_conv.append('1')
                pool_conv.append('0')
            else:
                if "bn" in layer_list[i]:
                    if "bn1" in layer_list[i]:
                        conv_bn = True
                    else:
                        conv_bn = False
                else:
                    res_list = layer_list[i].split("p")
                    size_conv.append(res_list[0])
                    filter_conv.append(res_list[1])
                    stride_conv.append(res_list[2])
                    padding_conv.append(res_list[3])
                    pool_conv.append(res_list[4])
    return size_conv, filter_conv, stride_conv, padding_conv, pool_conv, conv_bn

def fc_list_gen(size_fc, transition_size, fc_bn):
    fc_list = []
    for i in range(len(size_fc)):
        if i == 0:
            fc_list.append(["self.fc{}".format(i), " = torch.nn.Linear({}, {})".format(transition_size, size_fc[i])])
            if fc_bn:
                fc_list.append(["self.fc_bn" + "{}".format(i), " = torch.nn.BatchNorm1d({})".format(size_fc[i])])
        elif i != len(size_fc) - 1:
            fc_list.append(["self.fc{}".format(i), " = torch.nn.Linear({}, {})".format(size_fc[i-1], size_fc[i])])
            if fc_bn:
                fc_list.append(["self.fc_bn{}".format(i), " = torch.nn.BatchNorm1d({})".format(size_fc[i])])
        else:
            fc_list.append(["self.classifier", " = torch.nn.Linear({}, {})".format(size_fc[i-1], size_fc[i])])
            if fc_bn:
                fc_list.append(["self.classifier_bn", " = torch.nn.BatchNorm1d({}, affine=False)".format(size_fc[i])])
    return fc_list

def conv_list_gen(size_conv, filter_conv, stride_conv, padding_conv, input_channel, conv_bn):
    conv_list = []
    for i in range(len(size_conv)):
        if i == 0:
            conv_list.append(["self.conv{}".format(i), " = torch.nn.Conv2d({}, {}".format(input_channel, size_conv[i]) + ", ({}, {}), ".format(filter_conv[i], filter_conv[i]) + "stride=({}, {}), ".format(stride_conv[i], stride_conv[i]) + "padding=({}, {}))".format(padding_conv[i], padding_conv[i])])
            if conv_bn:
                conv_list.append(["self.conv_bn{}".format(i), " = torch.nn.BatchNorm2d({})".format(size_conv[i])])
        else:
            if "res" in size_conv[i]:
                conv_list.append(["self.conv{}".format(i), " = ResNetBasicblock({}, {}, ".format(size_conv[i-1], size_conv[i-1]) + "stride= {})".format(stride_conv[i])])
                size_conv[i] = size_conv[i-1]
            elif "th" in size_conv[i]:
                size_conv_num = int(size_conv[i].split("h")[1])
                conv_list.append(["self.conv{}".format(i), " = DepthwiseConv({}, {}, ".format(size_conv[i-1], size_conv_num) + "stride= {})".format(stride_conv[i])])
                size_conv[i] = '{}'.format(size_conv_num)
            else:
                conv_list.append(["self.conv{}".format(i), " = torch.nn.Conv2d({}, {}".format(size_conv[i-1], size_conv[i]) + ", ({}, {}), ".format(filter_conv[i], filter_conv[i]) + "stride=({}, {}), ".format(stride_conv[i], stride_conv[i]) + "padding=({}, {}))".format(padding_conv[i], padding_conv[i])])
                if conv_bn:
                    conv_list.append(["self.conv_bn{}".format(i), " = torch.nn.BatchNorm2d({})".format(size_conv[i])])
    return conv_list

def seq_from_string(model_name):
    label_sequence = []
    layer_list = model_name.split("_")
    start = 0
    for i in range(len(layer_list)):
        if "cnn" in layer_list[i]:
            start = 1
        elif "bn" in layer_list[i]:
            start = 0
        elif start == 1:
            if "res" in layer_list[i]:
                label_sequence.append(layer_name_to_int_map['conv'])
                label_sequence.append(layer_name_to_int_map['bn'])
                label_sequence.append(layer_name_to_int_map['relu'])
                label_sequence.append(layer_name_to_int_map['conv'])
                label_sequence.append(layer_name_to_int_map['bn'])
                # label_sequence.append(layer_name_to_int_map['add'])
            elif "depth" in layer_list[i]:
                label_sequence.append(layer_name_to_int_map['depthConv'])
                label_sequence.append(layer_name_to_int_map['bn'])
                label_sequence.append(layer_name_to_int_map['relu'])
                label_sequence.append(layer_name_to_int_map['pointConv'])
                label_sequence.append(layer_name_to_int_map['bn'])
            else:
                layer_list_step_in = layer_list[i].split("p")
                label_sequence.append(layer_name_to_int_map['conv'])
                if layer_list_step_in[4] != '0':
                    label_sequence.append(layer_name_to_int_map['pooling'])
                label_sequence.append(layer_name_to_int_map['relu'])
                if "bn1_mlp" in model_name:
                    label_sequence.append(layer_name_to_int_map['bn'])
    mlp_name = model_name.split("mlp")
    layer_list = mlp_name[1].split("_")
    start = 1
    for i in range(len(layer_list)):
        if "bn" in layer_list[i]:
            start = 0
        elif start == 1:
            label_sequence.append(layer_name_to_int_map['fc'])
            if "bn1" in mlp_name[1]:
                label_sequence.append(layer_name_to_int_map['bn'])
            if (i != len(layer_list) - 2):
                label_sequence.append(layer_name_to_int_map['relu'])
    label_sequence.append(layer_name_to_int_map['softmax'])
    return label_sequence

def save_label_from_string(dir_name, output_file, model_name):
    file_name = output_file.replace(".csv", "")
    label_fname = dir_name + "{}.npy".format(file_name)
    mstring_fname = dir_name + "{}.mstring".format(file_name)
    label_sequence = seq_from_string(model_name)
    np.save(label_fname, np.asarray(label_sequence))
    text_file = open(mstring_fname, "w")
    text_file.write(model_name)
    text_file.close()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_features", type=int, default=3072, help="flattened input size")
    parser.add_argument("--input_channel", type=int, default=3, help="input channel")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    # parser.add_argument("--model_name", type=str, default="mlp_512_256_128_bn1", help="model name")
    parser.add_argument("--random_seed", type=int, default="1234", help="random seed to generate the model")
    parser.add_argument("--file_name", type=str, default="torch_tvm_prof", help="this script directly modifies torch_tvm_prof.py")
    parser.add_argument("--output_file", type=str, default="complex_batch_1_nclass_1000_infeat_150528_seed_0.csv", help="this script directly modifies torch_tvm_prof.py")
    parser.add_argument("--opt_level", type=int, default="3", help="opt_level set to 3")
    args = parser.parse_args()
    '''format: $0p$1p$2p$3p$4'''
    dir_name = "./trace/"

    model_name = model_name_from_seed(args.random_seed)
    # for test only
    # model_name = 'mlp_512_256_128_bn1'

    logging.basicConfig(level=logging.DEBUG, filename=dir_name + "trace_gen.log", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info(model_name)

    print("Seed: {}, model name:{}".format(args.random_seed, model_name))

    save_label_from_string(dir_name, args.output_file, model_name)

    size_fc, fc_bn = mlp_list_from_string(model_name)
    size_fc.append(args.num_classes)
    num_fc = len(size_fc)

    size_conv, filter_conv, stride_conv, padding_conv, pool_conv, conv_bn = cnn_list_from_string(model_name)
    num_conv = len(size_conv)

    transition_size = args.input_features
    input_channel = args.input_channel
    input_features_sqrt = int(math.sqrt(args.input_features/args.input_channel))

    feature_sqrt = input_features_sqrt
    #derive the transition_size
    for i in range(len(pool_conv)):
        feature_sqrt = int((feature_sqrt + 2 * int(padding_conv[i]) + 1 - int(filter_conv[i]))/ int(stride_conv[i]))
        if (int(pool_conv[i]) != 0):
            feature_sqrt = int(feature_sqrt/2)


    if num_conv == 0:
        model_type = "mlp"

    else:
        model_type = "cnn"
        #calculate the flattened output size of cnn
        transition_size = int(size_conv[-1]) * feature_sqrt * feature_sqrt

    conv_list = conv_list_gen(size_conv, filter_conv, stride_conv, padding_conv, input_channel, conv_bn)
    fc_list = fc_list_gen(size_fc, transition_size, fc_bn)

    with open("./{}.py".format(args.file_name), "r") as in_file:
        buf = in_file.readlines()

    start_delete = 0
    with open("./{}.py".format(args.file_name), "w") as out_file:
        for line in buf:
            if ("# Model starts here" in line):
                out_file.write(line)
                start_delete = 1
            elif ("# Model ends here" in line):
                out_file.write(line)
                start_delete = 0
            elif ("# Start Call Model" in line):
                out_file.write(line)
                start_delete = 1
            elif ("# End Call Model" in line):
                out_file.write(line)
                start_delete = 0
            elif ("# Start Set Option" in line):
                out_file.write(line)
                start_delete = 1
            elif ("# End Set Option" in line):
                out_file.write(line)
                start_delete = 0
            elif start_delete == 0:
                out_file.write(line)

    with open("./{}.py".format(args.file_name), "r") as in_file:
        buf = in_file.readlines()

    with open("./{}.py".format(args.file_name), "w") as out_file:
        for line in buf:
            try:
                if "# Model starts here" in line:
                    if "mlp" in model_name:
                        line = line + "\nclass " + model_name + "(torch.nn.Module):"
                        line = line + "\n    def __init__(self, input_features, reshape = True, decompo_list = None, dummy_list = None):"
                        line = line + "\n        super(" + model_name + ",self).__init__()"
                        line = line + "\n        self.reshape = reshape"
                        line = line + "\n        self.decompo_list = decompo_list"
                        line = line + "\n        self.dummy_list = dummy_list"
                        line = line + "\n        self.relu = torch.nn.ReLU(inplace=True)"
                        line = line + "\n        self.logsoftmax = torch.nn.LogSoftmax(dim = 1)"
                        line = line + "\n        self.maxpool2x2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)"
                        line = line + "\n        self.avgpool2x2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)"
                        for i in range(len(conv_list)):
                            line = line + "\n        " + conv_list[i][0] + conv_list[i][1]
                        for i in range(len(fc_list)):
                            line = line + "\n        " + fc_list[i][0] + fc_list[i][1]
                        line = line + "\n        self.reset_parameters(input_features)"
                        line = line + "\n    def reset_parameters(self, input_features):"
                        line = line + "\n        stdv = 1.0 / math.sqrt(input_features)"
                        line = line + "\n        for weight in self.parameters():"
                        line = line + "\n            weight.data.uniform_(-stdv, +stdv)"
                        line = line + "\n    def forward(self, X1):"
                        j = 0
                        for i in range(len(conv_list)):
                            if i == 0:
                                line = line + "\n        if self.reshape:"
                                line = line + "\n            X1 = X1.reshape(-1, {}, {}, {})".format(input_channel, input_features_sqrt, input_features_sqrt)
                                line = line + "\n        X1 = " + conv_list[i][0] + "(X1)"
                            else:
                                line = line + "\n        X1 = " + conv_list[i][0] + "(X1)"
                            if ("conv" in conv_list[i][0]) and ("conv_bn" not in conv_list[i][0]):
                                if (int(pool_conv[j]) == 1 or int(pool_conv[j]) == 2):
                                    line = line + "\n        X1 = self.maxpool2x2(X1)"
                                elif int(pool_conv[j]) == 3:
                                    line = line + "\n        X1 = self.avgpool2x2(X1)"
                                line = line + "\n        X1 = self.relu(X1)"
                                j += 1
                            if i == len(conv_list) - 1:
                                line = line + "\n        X1 = X1.view(-1, {})".format(transition_size)
                        for i in range(len(fc_list)):
                            if i == 0 and num_conv == 0:
                                line = line + "\n        X1 = " + fc_list[i][0] + "(X1)"
                            else:
                                line = line + "\n        X1 = " + fc_list[i][0] + "(X1)"
                            if i < len(fc_list) - 1:
                                if "Linear" in fc_list[i+1][1]:
                                    line = line + "\n        X1 = self.relu(X1)"
                            else:
                                line = line + "\n        X1 = self.logsoftmax(X1)"
                        line = line + "\n        return X1\n\n"
                    else:
                        line = line
                elif "# Start Call Model" in line:
                    line = line + "\n    model = " + model_name + "(input_features).to(cuda_device)\n\n"
                elif "# Start Set Option" in line:
                    line = line + "\n    opt_level = {}\n\n".format(args.opt_level)
                elif "batch_size = " in line:
                    line = "    batch_size = " + "{}\n".format(args.batch_size)
                elif "input_features = " in line:
                    line = "    input_features = " + "{}\n".format(args.input_features)
                out_file.write(line)
            except:
                out_file.write(line)
    
    if args.file_name == "torch_tvm_prof":
        from torch_tvm_prof import run_tvm_torch
        run_tvm_torch()
        os.system('ncu --target-processes application-only --print-kernel-base function --log-file trace/{} --csv --kernel-regex-base function --launch-skip-before-match 0  --profile-from-start 1 --clock-control base --print-fp --apply-rules yes --section ImportantTraceAnalysis python torch_tvm_execute.py --batch_size {} --input_features {}'.format(args.output_file, args.batch_size, args.input_features))
    elif args.file_name == "torch_prof":
        os.system('ncu --target-processes application-only --print-kernel-base function --log-file trace/{} --csv --kernel-regex-base function --launch-skip-before-match 0  --profile-from-start 1 --clock-control base --print-fp --apply-rules yes --section ImportantTraceAnalysis python torch_execute_infer.py'.format(args.output_file))
    
if __name__ == '__main__':
    main()
