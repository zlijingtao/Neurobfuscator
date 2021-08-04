import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
import re
'''Functions to add obfuscating knobs to the model template'''
'''We use a brute-force way to extract the dimension information from the model and re-code with the added knobs'''
'''(Can only work on models matched the template)'''


'''Replace nn module in __init__() with added obfuscating knobs'''
def replace_func(line, reduced_list):
    broken_list = (line.split("=")[0]).split(".")
    line_type = broken_list[1].replace(" ", "")
    num_conv = 0
    dim1 = 0
    dim2 = 0
    if "_bn" in line_type:
        broken_list = line_type.split("_bn")
        master_type = broken_list[0] + broken_list[1]
        place = None
        for i in range(len(reduced_list)):
            if master_type == reduced_list[i]:
                if "conv" in line_type:
                    conv_id = int(master_type.replace("conv", ""))
                place = i
        if place == None:
            return line, dim1, dim2
    else:
        place = None
        for i in range(len(reduced_list)):
            if line_type == reduced_list[i]:
                if "conv" in line_type:
                    conv_id = int(line_type.replace("conv", ""))
                place = i
        if place == None:
            return line, dim1, dim2
            
    for item in reduced_list:
        if "conv" in item:
            num_conv += 1
    if "conv" in line_type and "_bn" not in line_type:
        lins_list = re.split('\(|,|\)',line)
        lins_list = line.split()
        dim1 = int(lins_list[2].split('(')[1].replace(",", ""))
        dim2 = int(lins_list[3].replace(",", ""))
        filt1 = int(lins_list[4].replace(",", "").replace("(", ""))
        filt2 = int(lins_list[5].replace("),", ""))
        stride1 = int(lins_list[6].split('(')[1].replace(",", ""))
        stride2 = int(lins_list[7].replace("),", ""))
        pad1 = int(lins_list[8].split('(')[1].replace(",", ""))
        num_group = 1
        if_bias = True
        if ("bias" in line) or ("group" in line):
            pad2 = int(lins_list[9].replace("),", ""))
            if ("group" in line):
                num_group = int(lins_list[10].split('=')[1].replace(",", "").replace(" ", ""))
            if ("bias" in line):
                if (lins_list[-1].split('=')[1].replace(")", "").replace(" ", "")) == "False":
                    if_bias = False
        else:
            pad2 = int(lins_list[9].replace("))", ""))
        
        if (dim2 %2 == 0):
            line = "\n        params = [{}, {}, {}, {}, {}, {}, {}, {}]".format(dim1, dim2, filt1, filt2, stride1, stride2, pad1, pad2)
            rest_stuff_conv = ", (params[2], params[3]), stride=(params[4], params[5]), padding=(params[6], params[7])"
            
            if num_group != 1:
                line = line + "\n        params.append({})".format(num_group)
                rest_stuff_conv += ", groups=params[8]"
            if not if_bias:
                rest_stuff_conv += ", bias=False"
            rest_stuff_conv += ")" 
            line = line + "\n        if self.widen_list != None:"
            if (place > 0):
                line = line + "\n            if self.widen_list[{}] > 1.0:".format(place - 1)
                line = line + "\n                params[0] = int(np.floor(params[0] * self.widen_list[{}] / 4) * 4)".format(place - 1)
            if (place < num_conv -1):
                line = line + "\n            if self.widen_list[{}] > 1.0:".format(place)
                line = line + "\n                params[1] = int(np.floor(params[1] * self.widen_list[{}] / 4) * 4)".format(place)
            line = line + "\n        if self.kerneladd_list != None:"
            line = line + "\n            if self.kerneladd_list[{}] > 0:".format(place)
            line = line + "\n                params[2] = params[2] + 2* int(self.kerneladd_list[{}])".format(place)
            line = line + "\n                params[3] = params[3] + 2* int(self.kerneladd_list[{}])".format(place)
            line = line + "\n                params[6] = params[6] + int(self.kerneladd_list[{}])".format(place)
            line = line + "\n                params[7] = params[7] + int(self.kerneladd_list[{}])".format(place)
            line = line + "\n        if self.decompo_list != None:"
            if num_group != 1: # If this layer is depth-wise conv, decompo cannot be applied.
                line = line + "\n            self.decompo_list[{}] = 0".format(place)
            line = line + "\n            if self.decompo_list[{}] == 1:".format(place)
            line = line + "\n                self.conv{}_0 = torch.nn.Conv2d(params[0], int(params[1]/2)".format(conv_id) + rest_stuff_conv
            line = line + "\n                self.conv{}_1 = torch.nn.Conv2d(params[0], int(params[1]/2)".format(conv_id) + rest_stuff_conv
            if (dim2 %4 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 2:".format(place)
                line = line + "\n                self.conv{}_0 = torch.nn.Conv2d(params[0], int(params[1]/4)".format(conv_id) + rest_stuff_conv
                line = line + "\n                self.conv{}_1 = torch.nn.Conv2d(params[0], int(params[1]/4)".format(conv_id) + rest_stuff_conv
                line = line + "\n                self.conv{}_2 = torch.nn.Conv2d(params[0], int(params[1]/4)".format(conv_id) + rest_stuff_conv
                line = line + "\n                self.conv{}_3 = torch.nn.Conv2d(params[0], int(params[1]/4)".format(conv_id) + rest_stuff_conv
            if (dim1 %2 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 3:".format(place)
                line = line + "\n                self.conv{}_0 = torch.nn.Conv2d(int(params[0]/2), params[1]".format(conv_id) + rest_stuff_conv
                line = line + "\n                self.conv{}_1 = torch.nn.Conv2d(int(params[0]/2), params[1]".format(conv_id) + rest_stuff_conv
            if (dim1 %4 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 4:".format(place)
                line = line + "\n                self.conv{}_0 = torch.nn.Conv2d(int(params[0]/4), params[1]".format(conv_id) + rest_stuff_conv
                line = line + "\n                self.conv{}_1 = torch.nn.Conv2d(int(params[0]/4), params[1]".format(conv_id) + rest_stuff_conv
                line = line + "\n                self.conv{}_2 = torch.nn.Conv2d(int(params[0]/4), params[1]".format(conv_id) + rest_stuff_conv
                line = line + "\n                self.conv{}_3 = torch.nn.Conv2d(int(params[0]/4), params[1]".format(conv_id) + rest_stuff_conv
            line = line + "\n            else:"
            line = line + "\n                self.conv{} = torch.nn.Conv2d(params[0], params[1]".format(conv_id) + rest_stuff_conv
            line = line + "\n        else:"
            line = line + "\n            self.conv{} = torch.nn.Conv2d(params[0], params[1]".format(conv_id) + rest_stuff_conv
            line = line + "\n        if self.deepen_list != None:"
            line = line + "\n            if self.deepen_list[{}] == 1:".format(place)
            line = line + "\n                self.conv{}_dp = torch.nn.Conv2d(params[1], params[1], (1, 1), stride=(1, 1), padding=(0, 0))".format(conv_id)
            line = line + "\n        if self.skipcon_list != None:"
            line = line + "\n            if self.skipcon_list[{}] == 1:".format(place)
            line = line + "\n                self.conv{}_sk = torch.nn.Conv2d(params[1], params[1], (params[2], params[3]), stride=(1, 1), padding=(int((params[2] - 1)/2), int((params[3] - 1)/2)))\n\n".format(conv_id)
    elif ("conv_bn" in line_type):
        line = "\n        self.conv_bn{} = torch.nn.BatchNorm2d(params[1])\n\n".format(conv_id)
    elif ("fc" in line_type or "classifier" in line_type) and "_bn" not in line_type:
        lins_list = re.split('\(|,|\)',line)
        lins_list = line.split()
        dim1 = int(lins_list[2].split('(')[1].replace(",", ""))
        dim2 = int(lins_list[3].replace(")", ""))
        if (dim2 %2 == 0):
            line = "\n        params = [{}, {}]".format(dim1, dim2)
            line = line + "\n        if self.widen_list != None:"
            if (place > num_conv):
                line = line + "\n            if self.widen_list[{}] > 1.0:".format(place - 1)
                line = line + "\n                params[0] = int(np.floor(params[0] * self.widen_list[{}] / 4) * 4)".format(place - 1)
            if "classifier" not in line_type:
                line = line + "\n            if self.widen_list[{}] > 1.0:".format(place)
                line = line + "\n                params[1] = int(np.floor(params[1] * self.widen_list[{}] / 4) * 4)".format(place)
            else:
                line = line + "\n            pass"
            line = line + "\n        if self.decompo_list != None:"
            line = line + "\n            if self.decompo_list[{}] == 1:".format(place)
            line = line + "\n                self.{}_0 = torch.nn.Linear(params[0], int(params[1]/2))".format(line_type)
            line = line + "\n                self.{}_1 = torch.nn.Linear(params[0], int(params[1]/2))".format(line_type)
            if (dim2 %4 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 2:".format(place)
                line = line + "\n                self.{}_0 = torch.nn.Linear(params[0], int(params[1]/4))".format(line_type)
                line = line + "\n                self.{}_1 = torch.nn.Linear(params[0], int(params[1]/4))".format(line_type)
                line = line + "\n                self.{}_2 = torch.nn.Linear(params[0], int(params[1]/4))".format(line_type)
                line = line + "\n                self.{}_3 = torch.nn.Linear(params[0], int(params[1]/4))".format(line_type)
            if (dim1 %2 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 3:".format(place)
                line = line + "\n                self.{}_0 = torch.nn.Linear(int(params[0]/2), params[1])".format(line_type)
                line = line + "\n                self.{}_1 = torch.nn.Linear(int(params[0]/2), params[1])".format(line_type)
            if (dim1 %4 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 4:".format(place)
                line = line + "\n                self.{}_0 = torch.nn.Linear(int(params[0]/4), params[1])".format(line_type)
                line = line + "\n                self.{}_1 = torch.nn.Linear(int(params[0]/4), params[1])".format(line_type)
                line = line + "\n                self.{}_2 = torch.nn.Linear(int(params[0]/4), params[1])".format(line_type)
                line = line + "\n                self.{}_3 = torch.nn.Linear(int(params[0]/4), params[1])".format(line_type)
            line = line + "\n            else:"
            line = line + "\n                self.{} = torch.nn.Linear(params[0], params[1])".format(line_type)
            line = line + "\n        else:"
            line = line + "\n            self.{} = torch.nn.Linear(params[0], params[1])\n\n".format(line_type)
            line = line + "\n        if self.deepen_list != None:"
            line = line + "\n            if self.deepen_list[{}] == 1:".format(place)
            line = line + "\n                self.{}_dp = torch.nn.Linear(params[1], params[1])".format(line_type)
            line = line + "\n        if self.skipcon_list != None:"
            line = line + "\n            if self.skipcon_list[{}] == 1:".format(place)
            line = line + "\n                self.{}_sk = torch.nn.Linear(params[1], params[1])\n\n".format(line_type)
    elif "fc_bn" in line_type:
        line = "\n        self.{} = torch.nn.BatchNorm1d(params[1])\n\n".format(line_type)
    elif "classifier_bn" in line_type:
        line = "\n        self.{} = torch.nn.BatchNorm1d(params[1], affine=False)\n\n".format(line_type)
    return line, dim1, dim2


'''Replace nn module in forward() with added obfuscating knobs'''
def replace_graph(line, reduced_list, dim_list, master_line_type = ""):
    line_type = line.split(".")[1].split("(")[0]
   
    if "relu" in line_type:
        place = None
        for i in range(len(reduced_list)):
            if master_line_type == reduced_list[i]:
                place = i

        if place  == None or master_line_type == "":
            return line, []
        master_type = master_line_type
    else:
        place = None
        for i in range(len(reduced_list)):
            if line_type == reduced_list[i]:
                if "conv" in line_type:
                    conv_id = int(line_type.replace("conv", ""))
                place = i
        if place == None:
            return line, []
        dim = dim_list.pop(0)
        dim1 = dim[0]
        dim2 = dim[1]
    
    if "conv" in line_type:
        if (dim2 %2 == 0):
            line = "\n        X1_shape = X1.size()"
            line = line + "\n        if self.decompo_list != None:"
            line = line + "\n            if self.decompo_list[{}] == 1:".format(place)
            line = line + "\n                X1_0 = self.conv{}_0(X1)".format(conv_id)
            line = line + "\n                X1_1 = self.conv{}_1(X1)".format(conv_id)
            line = line + "\n                X1 = torch.cat((X1_0, X1_1), 1)".format(place)
            if (dim2 %4 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 2:".format(place)
                line = line + "\n                X1_0 = self.conv{}_0(X1)".format(conv_id)
                line = line + "\n                X1_1 = self.conv{}_1(X1)".format(conv_id)
                line = line + "\n                X1_2 = self.conv{}_2(X1)".format(conv_id)
                line = line + "\n                X1_3 = self.conv{}_3(X1)".format(conv_id)
                line = line + "\n                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)"
            if (dim1 %2 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 3:".format(place)
                line = line + "\n                X1_0 = self.conv{}_0(X1[:, :int(torch.floor_divide(X1_shape[1],2)), :, :])".format(conv_id)
                line = line + "\n                X1_1 = self.conv{}_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):, :, :])".format(conv_id)
                line = line + "\n                X1 = torch.add(X1_0, X1_1)"
            if (dim1 %4 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 4:".format(place)
                line = line + "\n                X1_0 = self.conv{}_0(X1[:, :int(torch.floor_divide(X1_shape[1],4)), :, :])".format(conv_id)
                line = line + "\n                X1_1 = self.conv{}_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2)), :, :])".format(conv_id)
                line = line + "\n                X1_2 = self.conv{}_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4)), :, :])".format(conv_id)
                line = line + "\n                X1_3 = self.conv{}_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):, :, :])".format(conv_id)
                line = line + "\n                X1 = X1_0 + X1_1 + X1_2 + X1_3"
            line = line + "\n            else:"
            line = line + "\n                X1 = self.conv{}(X1)".format(conv_id)
            line = line + "\n        else:"
            line = line + "\n            X1 = self.conv{}(X1)\n".format(conv_id)
            
        line = line + "\n        if self.dummy_list != None:"
        line = line + "\n            if self.dummy_list[{}] > 0:".format(place)
        line = line + "\n                dummy = np.zeros(X1.size()).astype(\"float32\")"
        line = line + "\n                for i in range(self.dummy_list[{}]):".format(place)
        line = line + "\n                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))\n\n"
    elif "relu" in line_type:
        line = "\n        if True:"
        line = line + "\n            X1 = self.relu(X1)"
        line = line + "\n        if self.deepen_list != None:"
        line = line + "\n            if self.deepen_list[{}] == 1:".format(place)
        line = line + "\n                X1 = self.{}_dp(X1)".format(master_type)
        line = line + "\n                X1 = self.relu(X1)"
        line = line + "\n        if self.skipcon_list != None:"
        line = line + "\n            if self.skipcon_list[{}] == 1:".format(place)
        line = line + "\n                X1_skip = X1"
        line = line + "\n                X1 = self.{}_sk(X1)".format(master_type)
        line = line + "\n                X1 = self.relu(X1 + X1_skip)\n\n"
    elif "fc" in line_type or "classifier" in line_type:
        if (dim2 %2 == 0):
            line = "\n        X1_shape = X1.size()"
            line = line + "\n        if self.decompo_list != None:"
            line = line + "\n            if self.decompo_list[{}] == 1:".format(place)
            line = line + "\n                X1_0 = self.{}_0(X1)".format(line_type)
            line = line + "\n                X1_1 = self.{}_1(X1)".format(line_type)
            line = line + "\n                X1 = torch.cat((X1_0, X1_1), 1)"
            if (dim2 %4 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 2:".format(place)
                line = line + "\n                X1_0 = self.{}_0(X1)".format(line_type)
                line = line + "\n                X1_1 = self.{}_1(X1)".format(line_type)
                line = line + "\n                X1_2 = self.{}_2(X1)".format(line_type)
                line = line + "\n                X1_3 = self.{}_3(X1)".format(line_type)
                line = line + "\n                X1 = torch.cat([X1_0, X1_1, X1_2, X1_3], 1)"
            if (dim1 %2 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 3:".format(place)
                line = line + "\n                X1_0 = self.{}_0(X1[:, :int(torch.floor_divide(X1_shape[1],2))])".format(line_type)
                line = line + "\n                X1_1 = self.{}_1(X1[:, int(torch.floor_divide(X1_shape[1],2)):])".format(line_type)
                line = line + "\n                X1 = torch.add(X1_0, X1_1)"
            if (dim1 %4 == 0):
                line = line + "\n            elif self.decompo_list[{}] == 4:".format(place)
                line = line + "\n                X1_0 = self.{}_0(X1[:, :int(torch.floor_divide(X1_shape[1],4))])".format(line_type)
                line = line + "\n                X1_1 = self.{}_1(X1[:, int(torch.floor_divide(X1_shape[1],4)):int(torch.floor_divide(X1_shape[1],2))])".format(line_type)
                line = line + "\n                X1_2 = self.{}_2(X1[:, int(torch.floor_divide(X1_shape[1],2)):int(3*torch.floor_divide(X1_shape[1],4))])".format(line_type)
                line = line + "\n                X1_3 = self.{}_3(X1[:, int(3*torch.floor_divide(X1_shape[1],4)):])".format(line_type)
                line = line + "\n                X1 = X1_0 + X1_1 + X1_2 + X1_3"
            line = line + "\n            else:"
            line = line + "\n                X1 = self.{}(X1)".format(line_type)
            line = line + "\n        else:"
            line = line + "\n             X1 = self.{}(X1)\n".format(line_type)
            
        line = line + "\n        if self.dummy_list != None:"
        line = line + "\n            if self.dummy_list[{}] > 0:".format(place)
        line = line + "\n                dummy = np.zeros(X1.size()).astype(\"float32\")"
        line = line + "\n                for i in range(self.dummy_list[{}]):".format(place)
        line = line + "\n                    X1 = torch.add(X1, torch.tensor(dummy, device = cuda_device))\n\n"
    return line, dim_list

'''Modify the original model template with added obfuscating knobs and save to a new file with _obf afterfix'''
def func_modifier(model_py, modify_list, save_to_new_file = True):
    with open("./model_file/" + model_py, "r") as in_file:
        buf = in_file.readlines()
    reduced_list = []
    for i in range(len(modify_list)):
        if "fc" in modify_list[i] or "conv" in modify_list[i] or "classifier" in modify_list[i]:
            reduced_list.append(modify_list[i])
    if save_to_new_file:
        model_py = model_py.split(".py")[0] + "_obf.py"
    with open("./model_file/" + model_py, "w") as out_file:
        dim_list = []
        # relu_place = 0
        master_line_type = ""
        for line in buf:
            try:
                if "X1" in line:
                    line_type = line.split(".")[1].split("(")[0]
                    if line_type in reduced_list:
                        if line.count(" ") == 10:
                            line, dim_list = replace_graph(line, reduced_list, dim_list, master_line_type)
                            master_line_type = line_type
                    elif ("relu" in line_type):
                        if line.count(" ") == 10:
                            line, dim_list = replace_graph(line, reduced_list, dim_list, master_line_type)

                            # relu_place += 1
                else:
                    broken_list = line.split("=")[0].split(".")
                    line_type = broken_list[1].replace(" ", "")
                    if line_type in reduced_list: # modify computation layer [Conv2d, Linear]
                        if (broken_list[0].count(" ")) == 8:
                            line, dim1, dim2 = replace_func(line, reduced_list)
                            dim_list.append([dim1, dim2])
                    elif ("fc" in line_type) or ("conv" in line_type) or ("classifier" in line_type): # modify batchnorm layer [BatchNorm2d, BatchNorm1d]
                        if (broken_list[0].count(" ")) == 8:
                            if "params" not in line:
                                line, _, _ = replace_func(line, reduced_list)
                out_file.write(line)
            except IndexError:
                out_file.write(line)
    return model_py

'''Test_Run'''
if __name__ == '__main__':
    func_modifier("model_4.py", ['reshape', 'conv0', 'conv1', 'maxpool', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'reshape', 'fc0', 'fc1', 'fc2', 'classifier', 'softmax'])