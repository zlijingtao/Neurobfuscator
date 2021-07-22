#essemble attack eg two adversary models
import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
import torch.nn.functional as F
from utils import setup_logger
import logging
# from models import *
import os
import sys
import random
import argparse
from model_5_obf import custom_cnn_4
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--victim_seed', type=int, default=123, help='victim seed')
args = parser.parse_args()

random_seed = args.seed
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
victim_model = custom_cnn_4(3072)
checkpoint = torch.load("./best_checkpoint_model0_seed{}.tar".format(args.victim_seed))

best_prec1 = checkpoint['best_prec1']
victim_model.load_state_dict(checkpoint['state_dict'])
print("Validation for vgg-11 model is {}".format(best_prec1))
victim_model = victim_model.cuda()
#vgg


class Attack():

    
    criterion = nn.NLLLoss().cuda()
    e = 0.05
    succ = 0
    

    def ensemble_attack1(self, model_name, target, org_target, logger):
        logger.debug(model_name)
        image_arr = torch.load('data100/class_'+ str(org_target) +'.pth.tar')
        victim_model.eval()
        fake_label = torch.LongTensor(1)
        # fake_label[0] = target
        fake_label[0] = org_target
        fake_label = torch.autograd.Variable(fake_label.cuda(), requires_grad=False)
        org_label = torch.zeros(image_arr.shape[0])
        attack_label = torch.zeros(image_arr.shape[0])
        succ_label = torch.zeros(image_arr.shape[0])
        succ_iter = torch.zeros(image_arr.shape[0])
        
        if model_name == 'victim':
            
            models = [victim_model]

        elif model_name == 'original':
            orig_model = custom_cnn_4(3072)
            checkpoint = torch.load("./best_checkpoint_model0_seed{}.tar".format(random_seed))
            best_prec1 = checkpoint['best_prec1']
            orig_model.load_state_dict(checkpoint['state_dict'])
            print("Validation for original vgg-11 model is {}".format(best_prec1))
            models = [orig_model.cuda()]

        elif model_name == 'extravgg_3':
            
            decompo_list_3 = [0, 0, 2, 4, 0, 4, 4, 0, 1, 4, 3] 

            deepen_list_3 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
            model_3 = custom_cnn_4(3072, decompo_list = decompo_list_3, deepen_list = deepen_list_3)
            checkpoint = torch.load("./best_checkpoint_extravgg_model2_seed{}.tar".format(random_seed))
            best_prec1 = checkpoint['best_prec1']
            model_3.load_state_dict(checkpoint['state_dict'])
            logger.debug("Validation for model_3 is {}".format(best_prec1))
            models = [model_3.cuda()]

        elif model_name == 'newvgg_9':
            
            decompo_list_3 = [0, 0, 2, 4, 0, 4, 4, 0, 1, 4, 3] 

            deepen_list_3 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

            widen_list_2 = [1.0] * len(deepen_list_3)

            widen_list_2[0] = 1.5
            widen_list_2[2] = 1.5
            widen_list_2[4] = 1.5

            kerneladd_list_3 = [0] * len(deepen_list_3)
            kerneladd_list_3[0] = 1
            kerneladd_list_3[2] = 1
            kerneladd_list_3[4] = 1
            model_1 = custom_cnn_4(3072, decompo_list = decompo_list_3, deepen_list = deepen_list_3, widen_list = widen_list_2, kerneladd_list = kerneladd_list_3)
            checkpoint = torch.load("./best_checkpoint_newvgg_model0_seed{}.tar".format(random_seed))
            best_prec1 = checkpoint['best_prec1']
            model_1.load_state_dict(checkpoint['state_dict'])
            logger.debug("Validation for model_9 is {}".format(best_prec1))
            models = [model_1.cuda()]

        diff_succ = 0.0
        diff_all  = 0.0

        for i in range(image_arr.shape[0]):
        #for i in range(2):
            org_image = torch.FloatTensor(1, 3, 32, 32)
            org_image[0] = image_arr[i]
            # print(image_arr[i])
            org_image = torch.autograd.Variable(org_image.cuda(), requires_grad=True)

            output = victim_model(org_image)

            # activation = nn.LogSoftmax(1)

            # output = activation(output)
            
            _, org_pred = output.topk(1, 1, True, True)
            org_pred = org_pred.data[0, 0]
            # if i < 50:
            #     print(org_pred)
            fake_image = org_image.clone()

            self.e = 0.04
            #modify the original image
            max_val = torch.max(org_image).item()
            min_val = torch.min(org_image).item()
            for iter in range(50): # PGD: 
                # calculate gradient
                grad = torch.zeros(1, 3, 32, 32).cuda()
                fake_image = torch.autograd.Variable(fake_image.cuda(), requires_grad=True)
                for m in models:
                    if type(m) == type([]):
                        m = m[0]
                        m.eval()
                        fake_image_ = F.upsample(fake_image, size=32, mode='bilinear')
                        output = m(fake_image_)
                        loss = self.criterion(output, fake_label)
                        loss.backward()
                        # print(loss)
                        grad -= F.upsample(torch.sign(fake_image.grad), size=32, mode='bilinear')
                    else:
                        m.eval()
                        zero_gradients(fake_image)
                        output = m(fake_image)
                        loss = self.criterion(output, fake_label)
                        loss.backward()
                        #print(loss)
                        grad -= torch.sign(fake_image.grad)

                fake_image = fake_image - grad * self.e # self.e is alpha in PGD:  https://arxiv.org/pdf/1706.06083.pdf
                fake_image[fake_image > max_val] = max_val
                fake_image[fake_image < min_val] = min_val
                output = victim_model(fake_image)

                _, fake_pred = output.topk(1, 1, True, True)
                fake_pred = fake_pred.data[0, 0]

                if fake_label.item() != fake_pred.item() or iter == 49:
                    # print(fake_pred.item(), fake_label.item())
                    attack_pred_list = []
                    for m in models:
                        if type(m) == type([]):
                            output = m[0](F.upsample(fake_image, size=32, mode='bilinear'))
                        else:
                            output = m(fake_image)
                        _, attack_pred = output.topk(1, 1, True, True)
                        attack_pred_list.append(attack_pred.data[0, 0].item())

                    if (i + 1) % 20 == 0:
                        print(str(i) + '\torg prediction: ' + str(org_pred.item()) + '\tfake prediction: ' + str(fake_pred.item()) +
                          '\tattack_pred: ' + str(attack_pred_list) + '\te: ' + str(self.e) + '\titer: ' + str(iter) + '\tsucc: ' + str(self.succ))

                    org_label[i] = org_pred.item()
                    attack_label[i] = fake_pred.item()
                    succ_iter[i] = iter + 1
                    
                    diff = torch.sum((org_image - fake_image) ** 2).item()
                    diff_all += diff

                    if fake_label.item() != fake_pred:
                        diff_succ += diff
                        self.succ += 1
                        succ_label[i] = 1
                    break


        diff_all /= (1.0 * image_arr.shape[0])
        if self.succ > 0:
            diff_succ /= (1.0 * self.succ)
        #print('total: ' + str(i + 1) + '\tsuccess: ' + str(self.succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()))
        
        str_log = 'src: ' + str(org_target) + '\ttar: ' + str(target)+ '\ttotal: ' + str(i + 1) + '\tsuccess: ' + str(self.succ) + '\tavg iter: ' + str(torch.mean(succ_iter).item()) + '\tdiff_suc: ' + str(diff_succ) + '\tdif_total: ' + str(diff_all) 
        # print(str_log)
        logger.debug(str_log)
        str_filename = './attack_result/' + model_name + 'seed{}_PGDnotarget.log'.format(random_seed)
        f = open(str_filename, "a")
        f.write(str_log)
        f.write("\n")
        f.close()
        self.succ = 0


def main():

    if not os.path.isdir("./attack_result"):
        os.mkdir("./attack_result")

    model_log_file = "./attack_result/PGD_Notarget_attack_newvgg11_seed{}.log".format(random_seed)
    logger = setup_logger('first_logger', model_log_file, level = logging.DEBUG)
    if args.victim_seed != args.seed:
        model_list_name = ['original', 'newvgg_9', 'extravgg_3']
    else:
        model_list_name = ['newvgg_9', 'extravgg_3']
    
    attack = Attack()
    history_list = []
    
    iter_max = 10
    for temporal in range(iter_max):
        org_target = temporal
        attack_target = (org_target + 5)%9
        history_list.append(org_target)
        print("###Round", temporal, "  src_label:", org_target, "  target_label:", attack_target)
        #print(temporal, attack_target)
        for model_name in model_list_name:
            attack.ensemble_attack1(model_name, attack_target, org_target, logger)
            
                

if __name__ == '__main__':
    main()










