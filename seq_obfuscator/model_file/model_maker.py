import torchvision.models as models
import torch

import sys
sys.path.append('../')
from torch_utils import summary

"""VGG-19"""
# from model_9 import custom_cnn_8

# vgg19 = models.vgg19()
# print(summary(vgg19, (3,224,224), device = None))

# custom_cnn = custom_cnn_8(150528)
# print(summary(custom_cnn, (3,224,224), device = None))

"""Resnet-18"""

# from model_10 import custom_cnn_9

# resnet18 = models.resnet18()
# print(summary(resnet18, (3,224,224), device = None))

# custom_cnn = custom_cnn_9(150528)
# print(summary(custom_cnn, (3,224,224), device = None))


# resnet18 = models.resnet18()
# print(summary(resnet18, (3,224,224), device = None))


# from model_6 import custom_cnn_5
# from vanilla_resnet_cifar import resnet20
# resnet20_model = resnet20()
# print(summary(resnet20_model, (3,32,32), device = None))

# custom_cnn = custom_cnn_5(3072)
# print(summary(custom_cnn, (3,32,32), device = None))

# from model_7 import custom_cnn_6

# custom_cnn = custom_cnn_6(3072)
# print(summary(custom_cnn, (3,32,32), device = None))

# from model_8 import custom_cnn_7
# from vanilla_resnet_cifar import resnet32
# resnet32_model = resnet32()
# print(summary(resnet32_model, (3,32,32), device = None))

# custom_cnn = custom_cnn_7(3072)
# print(summary(custom_cnn, (3,32,32), device = None))

mobilenet = models.MobileNetV2()
print(summary(mobilenet, (3,224,224), device = None))

from model_11 import custom_cnn_10
custom_cnn = custom_cnn_10(150528)
print(summary(custom_cnn, (3,224,224), device = None))