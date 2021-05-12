import numpy as np
import argparse

def make_label_from_string(model_id, input_string):
    model_file = "model_{}.npy".format(model_id + 1)
    content = []
    for char in input_string:
        try:
            content.append(int(char))
        except:
            raise("String content must be all digit!")
    content = np.asarray(content)
    np.save(model_file, content)
    print("Model Label saved for {}, is {}".format(model_file, str(content)))

def read_label_from_npy(npy_name):
    np_array = np.load(npy_name)
    print("Read label from {} is: ".format(npy_name), np_array)



# layer_int_to_name_map = {0:'conv', 1:'fc', 2:'pooling', 3:'bn', 4:'depthConv',5:'relu', 6:'pointConv', 7:'add',8:'softmax'}
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='./model_13.npy')
    parser.add_argument('--model_id', default=12, type=int, help='Which Model You Want to Make the Label for? Name the ID')
    parser.add_argument('--input_string', default="0000000000000000000000000000000218")
    args = parser.parse_args()

    
    #input_string for id=4 (vgg11-cifar10): 02020020020021118
    #input_string for id=5 (resnet20-cifar10):  000000000600000060000218/0000000000000000000218
    #input_string for id=6 (vgg13-cifar10): 0020020020020021118
    #input_string for id=7 (resnet32-cifar10):  000000000000060000000000600000000218/0000000000000000000000000000000218
    #input_string for id=8 (vgg19-imagenet):  0020020000200002000021118
    #input_string for id=9 (resnet18-imagenet):  020000000000000000218
    #input_string for id=10 (mobilenetV2-imagenet): "0" * 52 + "218"
    #input_string for id=10 (mobilenetV2-imagenet): "046646646646646646646646646646646646646646646646646618"
    #input_string for id=11 (ENAS-net): 46006466246606466466260646606466628
    # args.input_string = "0" * 18 + "218"
    # args.input_string = "0000000000000000000000000000000218"
    args.input_string = "0002460200000000001118"
    read_label_from_npy(args.input_file)
    make_label_from_string(args.model_id, args.input_string)