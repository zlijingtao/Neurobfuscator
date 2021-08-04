import torch 
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import logging
import importlib
import sys
from torch_relay_build import torch_relay_func
from torch_func_modifier import func_modifier
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

'''Calculate Latency Overhead from csv file'''
def csv_to_time_overhead(csv_file):
    df = pd.read_csv(csv_file, skiprows=2)
    # df = df.drop_duplicates(subset=['ID'])
    # print(df)
    trace_df = df[df['Metric Name'] == "Cycles"]
    trace_df= trace_df.replace(',','', regex=True)
    trace_df['Metric Value'] = pd.to_numeric(trace_df['Metric Value'])
    cost = trace_df['Metric Value'].sum()
    return cost

'''Decode a sequence prediction'''
def reverse_map(label_file, layer_int_to_name_map):
    label_array = np.load(label_file)

    str_decoded = ''
    for x in label_array:
        if x in layer_int_to_name_map:
            str_decoded = str_decoded + layer_int_to_name_map[x] + ' '
        else:
            print("x=%d MAJOR ERROR? OUT OF PREDICTION SCOPE" % x)
    return str_decoded

'''Modify the saved model parameters, this is necessary for a trained model in practice'''
def modify_state_dict(model_file, state_dict, modify_list, widen_list, decompo_list, deepen_list, skipcon_list, kerneladd_list):
    j = -1
    # print("modify the state_dict to apply decomposition")
    with open("./model_file/" + model_file, "r") as in_file:
        buf = in_file.readlines()
    for i in range(len(modify_list)):
        if "conv" in modify_list[i]: # assume all conv layer use "conv + digit" as layer name. such as conv1, conv8
            j += 1
            conv_name = "conv" + modify_list[i].split("v")[1]
            if_bias = next((False for line in buf if ("self.{} = ".format(conv_name) in line and "False" in line and "torch.nn." in line)), True)
            '''Modify kerneladd_list (conv)'''
            if kerneladd_list[j] > 0: #kernel_size
                orig_weight = state_dict["{}.weight".format(conv_name)]
                state_dict['{}.weight'.format(conv_name)] = F.pad(input=orig_weight, pad=(kerneladd_list[j], kerneladd_list[j], kerneladd_list[j], kerneladd_list[j], 0, 0, 0, 0), mode='constant', value=0.0)
            '''Modify widen_list (conv)'''
            if (widen_list[j] > 1) and (j < len(kerneladd_list) - 1): #layer_widening, do not do to the last conv layer
                orig_weight = state_dict["{}.weight".format(conv_name)]
                orig_shape = orig_weight.size()
                if if_bias:
                    orig_bias = state_dict["{}.bias".format(conv_name)]
                bn_name = "conv_bn" + modify_list[i].split("v")[1]
                next_conv_name = "conv" + str(int(modify_list[i].split("v")[1]) + 1)
                orig_next_weight = state_dict['{}.weight'.format(next_conv_name)]

                extra_c = int(np.floor(orig_shape[0] *(widen_list[j] - np.floor(widen_list[j]))/4)*4) # get float part
                mul_factor = int(np.floor(widen_list[j])) # get int
                state_dict['{}.weight'.format(conv_name)] = torch.cat([orig_weight[:extra_c, :, :, :].repeat_interleave(mul_factor+1,0), orig_weight[extra_c:, :, :, :].repeat_interleave(mul_factor,0)], 0)
                if if_bias:
                    state_dict['{}.bias'.format(conv_name)] = torch.cat([orig_bias[:extra_c].repeat_interleave(mul_factor+1,0), orig_bias[extra_c:].repeat_interleave(mul_factor,0)], 0)
                result = next((True for line in buf if bn_name in line), False)
                if result:
                    orig_bn_weight = state_dict['{}.weight'.format(bn_name)]
                    orig_bn_bias = state_dict['{}.bias'.format(bn_name)]
                    orig_bn_running_mean = state_dict['{}.running_mean'.format(bn_name)]
                    orig_bn_running_var = state_dict['{}.running_var'.format(bn_name)]
                    state_dict['{}.weight'.format(bn_name)] = torch.cat([orig_bn_weight[:extra_c].repeat_interleave(mul_factor+1,0), orig_bn_weight[extra_c:].repeat_interleave(mul_factor,0)], 0)
                    state_dict['{}.bias'.format(bn_name)] = torch.cat([orig_bn_bias[:extra_c].repeat_interleave(mul_factor+1,0), orig_bn_bias[extra_c:].repeat_interleave(mul_factor,0)], 0)
                    state_dict['{}.running_mean'.format(bn_name)] = torch.cat([orig_bn_running_mean[:extra_c].repeat_interleave(mul_factor+1,0), orig_bn_running_mean[extra_c:].repeat_interleave(mul_factor,0)], 0)
                    state_dict['{}.running_var'.format(bn_name)] = torch.cat([orig_bn_running_var[:extra_c].repeat_interleave(mul_factor+1,0), orig_bn_running_var[extra_c:].repeat_interleave(mul_factor,0)], 0)
                state_dict['{}.weight'.format(next_conv_name)] = torch.cat([orig_next_weight[:, :extra_c, :, :].repeat_interleave(mul_factor+1,1)/(mul_factor+1), orig_next_weight[:, extra_c:, :, :].repeat_interleave(mul_factor,1)/mul_factor], 1)

            '''Modify deepen_list (conv)'''
            if deepen_list[j] == 1: #layer_deepening
                state_dict['{}_dp.weight'.format(conv_name)] = torch.eye(state_dict['{}.weight'.format(conv_name)].size()[0]).unsqueeze(2).unsqueeze(3)
                if if_bias:
                    state_dict['{}_dp.bias'.format(conv_name)] = torch.zeros_like(state_dict['{}.bias'.format(conv_name)])
                else:
                    state_dict['{}_dp.bias'.format(conv_name)] = torch.zeros([state_dict['{}_dp.weight'.format(conv_name)].size()[1]])

            '''Modify skipcon_list (conv)'''
            if skipcon_list[j] == 1: #skip_connection
                orig_weight_shape = state_dict["{}.weight".format(conv_name)].size()
                state_dict['{}_sk.weight'.format(conv_name)] = torch.zeros([orig_weight_shape[0], orig_weight_shape[0], orig_weight_shape[2], orig_weight_shape[3]])
                if if_bias:
                    state_dict['{}_sk.bias'.format(conv_name)] = torch.zeros_like(state_dict['{}.bias'.format(conv_name)])
                else:
                    state_dict['{}_sk.bias'.format(conv_name)] = torch.zeros([orig_weight_shape[0]])
            '''Modify decompo_list (conv)'''
            orig_weight = state_dict["{}.weight".format(conv_name)]
            orig_shape = orig_weight.size()
            if if_bias:
                orig_bias = state_dict["{}.bias".format(conv_name)]
            state_dict.pop("{}.weight".format(conv_name), None)
            state_dict.pop("{}.bias".format(conv_name), None)
            disable_decompo = next((True for line in buf if "self.decompo_list[{}] = 0".format(j) in line), False)
            if decompo_list[j] == 0 or decompo_list[j] > 4 or disable_decompo:
                state_dict["{}.weight".format(conv_name)] = orig_weight
                if if_bias:
                    state_dict["{}.bias".format(conv_name)] = orig_bias
            elif decompo_list[j] == 1:
                result = next((True for line in buf if "decompo_list[{}] == 1".format(j) in line), False)
                if result:
                    state_dict["{}_0.weight".format(conv_name)] = orig_weight[:int(orig_shape[0]/2), :, :, :]
                    state_dict["{}_1.weight".format(conv_name)] = orig_weight[int(orig_shape[0]/2):, :, :, :]
                    if if_bias:
                        state_dict["{}_0.bias".format(conv_name)] = orig_bias[:int(orig_shape[0]/2)]
                        state_dict["{}_1.bias".format(conv_name)] = orig_bias[int(orig_shape[0]/2):]
                else:
                    state_dict["{}.weight".format(conv_name)] = orig_weight
                    if if_bias:
                        state_dict["{}.bias".format(conv_name)] = orig_bias
            elif decompo_list[j] == 2:
                result = next((True for line in buf if "decompo_list[{}] == 2".format(j) in line), False)
                if result:
                    state_dict["{}_0.weight".format(conv_name)] = orig_weight[:int(orig_shape[0]/4), :, :, :]
                    state_dict["{}_1.weight".format(conv_name)] = orig_weight[int(orig_shape[0]/4):int(orig_shape[0]/2), :, :, :]
                    state_dict["{}_2.weight".format(conv_name)] = orig_weight[int(orig_shape[0]/2):int(3*orig_shape[0]/4), :, :, :]
                    state_dict["{}_3.weight".format(conv_name)] = orig_weight[int(3*orig_shape[0]/4):, :, :, :]
                    if if_bias:
                        state_dict["{}_0.bias".format(conv_name)] = orig_bias[:int(orig_shape[0]/4)]
                        state_dict["{}_1.bias".format(conv_name)] = orig_bias[int(orig_shape[0]/4):int(orig_shape[0]/2)]
                        state_dict["{}_2.bias".format(conv_name)] = orig_bias[int(orig_shape[0]/2):int(3*orig_shape[0]/4)]
                        state_dict["{}_3.bias".format(conv_name)] = orig_bias[int(3*orig_shape[0]/4):]
                else:
                    state_dict["{}.weight".format(conv_name)] = orig_weight
                    if if_bias:
                        state_dict["{}.bias".format(conv_name)] = orig_bias
            elif decompo_list[j] == 3:
                result = next((True for line in buf if "decompo_list[{}] == 3".format(j) in line), False)
                if result:
                    state_dict["{}_0.weight".format(conv_name)] = orig_weight[:, :int(orig_shape[1]/2), :, :]
                    state_dict["{}_1.weight".format(conv_name)] = orig_weight[:, int(orig_shape[1]/2):, :, :]
                    if if_bias:
                        state_dict["{}_0.bias".format(conv_name)] = orig_bias/2
                        state_dict["{}_1.bias".format(conv_name)] = orig_bias/2
                else:
                    state_dict["{}.weight".format(conv_name)] = orig_weight
                    if if_bias:
                        state_dict["{}.bias".format(conv_name)] = orig_bias
            elif decompo_list[j] == 4:
                result = next((True for line in buf if "decompo_list[{}] == 4".format(j) in line), False)
                if result:
                    state_dict["{}_0.weight".format(conv_name)] = orig_weight[:, :int(orig_shape[1]/4), :, :]
                    state_dict["{}_1.weight".format(conv_name)] = orig_weight[:, int(orig_shape[1]/4):int(orig_shape[1]/2), :, :]
                    state_dict["{}_2.weight".format(conv_name)] = orig_weight[:, int(orig_shape[1]/2):int(3*orig_shape[1]/4), :, :]
                    state_dict["{}_3.weight".format(conv_name)] = orig_weight[:, int(3*orig_shape[1]/4):, :, :]
                    if if_bias:
                        state_dict["{}_0.bias".format(conv_name)] = orig_bias/4
                        state_dict["{}_1.bias".format(conv_name)] = orig_bias/4
                        state_dict["{}_2.bias".format(conv_name)] = orig_bias/4
                        state_dict["{}_3.bias".format(conv_name)] = orig_bias/4
                else:
                    state_dict["{}.weight".format(conv_name)] = orig_weight
                    if if_bias:
                        state_dict["{}.bias".format(conv_name)] = orig_bias


        elif "fc" in modify_list[i] or "classifier" in modify_list[i]:
            j += 1
            '''Modify widen_list (fc)'''
            if (widen_list[j] > 1) and (j < len(widen_list) - 1): #layer_widening, do not do to the last fc layer
                orig_weight = state_dict["{}.weight".format(modify_list[i])]
                orig_shape = orig_weight.size()
                orig_bias = state_dict["{}.bias".format(modify_list[i])]
                bn_name = "fc_bn" + modify_list[i].split("c")[1]
                if (j < len(widen_list) - 2):
                    next_fc_name = "fc" + str(int(modify_list[i].split("c")[1]) + 1)
                else:
                    next_fc_name = "classifier"
                orig_next_weight = state_dict['{}.weight'.format(next_fc_name)]

                extra_c = int(np.floor(orig_shape[0] *(widen_list[j] - np.floor(widen_list[j]))/4)*4) # get float part
                mul_factor = int(np.floor(widen_list[j])) # get int

                state_dict['{}.weight'.format(modify_list[i])] = torch.cat([orig_weight[:extra_c, :].repeat_interleave(mul_factor+1,0), orig_weight[extra_c:, :].repeat_interleave(mul_factor,0)], 0)
                state_dict['{}.bias'.format(modify_list[i])] = torch.cat([orig_bias[:extra_c].repeat_interleave(mul_factor+1,0), orig_bias[extra_c:].repeat_interleave(mul_factor,0)], 0)
                result = next((True for line in buf if bn_name in line), False)
                result2 = next((True for line in buf if (bn_name in line) and ("affine=False" in line)), False)
                if result:
                    orig_bn_weight = state_dict['{}.weight'.format(bn_name)]
                    orig_bn_bias = state_dict['{}.bias'.format(bn_name)]
                    orig_bn_running_mean = state_dict['{}.running_mean'.format(bn_name)]
                    orig_bn_running_var = state_dict['{}.running_var'.format(bn_name)]
                    if not result2:
                        state_dict['{}.weight'.format(bn_name)] = torch.cat([orig_bn_weight[:extra_c].repeat_interleave(mul_factor+1,0), orig_bn_weight[extra_c:].repeat_interleave(mul_factor,0)], 0)
                        state_dict['{}.bias'.format(bn_name)] = torch.cat([orig_bn_bias[:extra_c].repeat_interleave(mul_factor+1,0), orig_bn_bias[extra_c:].repeat_interleave(mul_factor,0)], 0)
                    state_dict['{}.running_mean'.format(bn_name)] = torch.cat([orig_bn_running_mean[:extra_c].repeat_interleave(mul_factor+1,0), orig_bn_running_mean[extra_c:].repeat_interleave(mul_factor,0)], 0)
                    state_dict['{}.running_var'.format(bn_name)] = torch.cat([orig_bn_running_var[:extra_c].repeat_interleave(mul_factor+1,0), orig_bn_running_var[extra_c:].repeat_interleave(mul_factor,0)], 0)
                state_dict['{}.weight'.format(next_fc_name)] = torch.cat([orig_next_weight[:, :extra_c].repeat_interleave(mul_factor+1,1)/(mul_factor+1), orig_next_weight[:, extra_c:].repeat_interleave(mul_factor,1)/mul_factor], 1)


            '''Modify deepen_list (fc)'''
            if deepen_list[j] == 1: #layer_deepening
                state_dict['{}_dp.weight'.format(modify_list[i])] = torch.eye(state_dict['{}.weight'.format(modify_list[i])].size()[0])
                state_dict['{}_dp.bias'.format(modify_list[i])] = torch.zeros_like(state_dict['{}.bias'.format(modify_list[i])])

            '''Modify skipcon_list (fc)'''
            if skipcon_list[j] == 1: #skip_connection
                orig_weight_shape = state_dict["{}.weight".format(modify_list[i])].size()
                state_dict['{}_sk.weight'.format(modify_list[i])] = torch.zeros([orig_weight_shape[0], orig_weight_shape[0]])
                state_dict['{}_sk.bias'.format(modify_list[i])] = torch.zeros_like(state_dict['{}.bias'.format(modify_list[i])])

            '''Modify decompo_list (fc)'''
            orig_weight = state_dict["{}.weight".format(modify_list[i])]
            orig_shape = orig_weight.size()
            orig_bias = state_dict["{}.bias".format(modify_list[i])]
            state_dict.pop("{}.weight".format(modify_list[i]), None)
            state_dict.pop("{}.bias".format(modify_list[i]), None)
            if decompo_list[j] == 1:
                result = next((True for line in buf if "decompo_list[{}] == 1".format(j) in line), False)
                if result:
                    state_dict["{}_0.weight".format(modify_list[i])] = orig_weight[:int(orig_shape[0]/2), :]
                    state_dict["{}_0.bias".format(modify_list[i])] = orig_bias[:int(orig_shape[0]/2)]
                    state_dict["{}_1.weight".format(modify_list[i])] = orig_weight[int(orig_shape[0]/2):, :]
                    state_dict["{}_1.bias".format(modify_list[i])] = orig_bias[int(orig_shape[0]/2):]
                else:
                    state_dict["{}.weight".format(modify_list[i])] = orig_weight
                    state_dict["{}.bias".format(modify_list[i])] = orig_bias
            elif decompo_list[j] == 2:
                result = next((True for line in buf if "decompo_list[{}] == 2".format(j) in line), False)
                if result:
                    state_dict["{}_0.weight".format(modify_list[i])] = orig_weight[:int(orig_shape[0]/4), :]
                    state_dict["{}_0.bias".format(modify_list[i])] = orig_bias[:int(orig_shape[0]/4)]
                    state_dict["{}_1.weight".format(modify_list[i])] = orig_weight[int(orig_shape[0]/4):int(orig_shape[0]/2), :]
                    state_dict["{}_1.bias".format(modify_list[i])] = orig_bias[int(orig_shape[0]/4):int(orig_shape[0]/2)]
                    state_dict["{}_2.weight".format(modify_list[i])] = orig_weight[int(orig_shape[0]/2):int(3*orig_shape[0]/4), :]
                    state_dict["{}_2.bias".format(modify_list[i])] = orig_bias[int(orig_shape[0]/2):int(3*orig_shape[0]/4)]
                    state_dict["{}_3.weight".format(modify_list[i])] = orig_weight[int(3*orig_shape[0]/4):, :]
                    state_dict["{}_3.bias".format(modify_list[i])] = orig_bias[int(3*orig_shape[0]/4):]
                else:
                    state_dict["{}.weight".format(modify_list[i])] = orig_weight
                    state_dict["{}.bias".format(modify_list[i])] = orig_bias
            elif decompo_list[j] == 3:
                result = next((True for line in buf if "decompo_list[{}] == 3".format(j) in line), False)
                if result:
                    state_dict["{}_0.weight".format(modify_list[i])] = orig_weight[: , :int(orig_shape[1]/2)]
                    state_dict["{}_0.bias".format(modify_list[i])] = orig_bias/2
                    state_dict["{}_1.weight".format(modify_list[i])] = orig_weight[: , int(orig_shape[1]/2):]
                    state_dict["{}_1.bias".format(modify_list[i])] = orig_bias/2
                else:
                    state_dict["{}.weight".format(modify_list[i])] = orig_weight
                    state_dict["{}.bias".format(modify_list[i])] = orig_bias
            elif decompo_list[j] == 4:
                result = next((True for line in buf if "decompo_list[{}] == 4".format(j) in line), False)
                if result:
                    state_dict["{}_0.weight".format(modify_list[i])] = orig_weight[: , :int(orig_shape[1]/4)]
                    state_dict["{}_0.bias".format(modify_list[i])] = orig_bias/4
                    state_dict["{}_1.weight".format(modify_list[i])] = orig_weight[: , int(orig_shape[1]/4):int(orig_shape[1]/2)]
                    state_dict["{}_1.bias".format(modify_list[i])] = orig_bias/4
                    state_dict["{}_2.weight".format(modify_list[i])] = orig_weight[: , int(orig_shape[1]/2):int(3 * orig_shape[1]/4)]
                    state_dict["{}_2.bias".format(modify_list[i])] = orig_bias/4
                    state_dict["{}_3.weight".format(modify_list[i])] = orig_weight[: , int(3 * orig_shape[1]/4):]
                    state_dict["{}_3.bias".format(modify_list[i])] = orig_bias/4
                else:
                    state_dict["{}.weight".format(modify_list[i])] = orig_weight
                    state_dict["{}.bias".format(modify_list[i])] = orig_bias
            else:
                state_dict["{}.weight".format(modify_list[i])] = orig_weight
                state_dict["{}.bias".format(modify_list[i])] = orig_bias
    return state_dict

'''We are identifying a _obf model file to derive the search space. i.e. len(decompo_list)'''
def identify_model(model_log_file, forbid1x1 = False):
    num_conv = 0
    num_linear = 0
    modify_list = ["reshape"]
    kerneladd_list = []
    decompo_list = []
    with open(model_log_file, "r") as in_file:
        buf = in_file.readlines()
        for line in buf:
            if "Conv2d" in line:
                if forbid1x1:
                    if "(1x1)" not in line:
                        modify_list.append("conv{}".format(num_conv))
                        decompo_list.append(0)
                        kerneladd_list.append(0)
                else:
                    modify_list.append("conv{}".format(num_conv))
                    decompo_list.append(0)
                    kerneladd_list.append(0)
                num_conv += 1
            elif "Linear" in line:
                if num_linear == 0:
                    modify_list.append("reshape")
                if "[-1, 10]" in line or "[-1, 100]" in line or "[-1, 1000]" in line:
                    modify_list.append("classifier")
                    decompo_list.append(0)
                    num_linear += 1
                else:
                    modify_list.append("fc{}".format(num_linear))
                    decompo_list.append(0)
                    num_linear += 1
            elif "Pool" in line:
                modify_list.append("maxpool")
            elif "LogSoftmax" in line:
                modify_list.append("softmax")
    return modify_list, decompo_list, kerneladd_list

'''This funciton adds more entry to allow more fusable operation (after obfuscation, the number of fusion node increases accordingly, see misc/copy2tvm/tvm/src/transforms/fuse_ops.cc)'''
def get_extra_entries(decompo_list, dummy_list, deepen_list, skipcon_list):
    #other list could also bring more entries. number of entry decompo list could bring is been offseted by +3.
    result = sum(dummy_list)
    result += 4 * (sum(deepen_list) + sum(skipcon_list))
    for i in range(len(decompo_list)):
        if decompo_list[i] == 1:
            result += 4
        elif decompo_list[i] == 3:
            result += 10
        elif decompo_list[i] == 2:
            result += 6
        elif decompo_list[i] == 4:
            result += 17
    return result


def setup_logger(name, log_file, level=logging.INFO, console_out = False):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    if console_out:
        stdout_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stdout_handler)
    return logger

def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info = summary_string(model, input_size, batch_size, device, dtypes)
    return result

def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]["weight_shape"] = list(module.weight.size())
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        # line_new = "{:>20}  {:>25} {:>15}".format(
        #     layer,
        #     str(summary[layer]["output_shape"]),
        #     "{0:,}".format(summary[layer]["nb_params"]),
        # )
        if "weight_shape" in summary[layer]:
            if len(summary[layer]["weight_shape"]) == 4:
                kernel_size = summary[layer]["weight_shape"][-1]
            else:
                kernel_size = "x"
        else:
            kernel_size = "x"

        line_new = "{:>20}  {:>25} {:>15}".format(
            layer + " ({}x{})".format(kernel_size, kernel_size),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)