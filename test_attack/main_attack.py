from model_5_obf import custom_cnn_4
from model_5_d1 import custom_cnn_d1
from model_5_d2 import custom_cnn_d2
from model_5_d3 import custom_cnn_d3
from model_6_obf import custom_cnn_5
from model_6_d4 import custom_cnn_d4
from model_6_d5 import custom_cnn_d5
from model_6_d6 import custom_cnn_d6
import torch
import torch.optim as optim
import torch.nn as nn
import time
from utils import setup_logger, AverageMeter
assert torch.cuda.is_available()
import logging
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123, help='random seed')
args = parser.parse_args()

random_seed = args.seed
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        m.bias.data.zero_()

def main():
    cuda_device = torch.device("cuda")  # device object representing GPU
    # model_log_file = "train_res20_seed{}.log".format(random_seed)
    model_log_file = "train_newvgg_seed{}.log".format(random_seed)
    logger = setup_logger('first_logger', model_log_file, level = logging.DEBUG)

    input_features = 3072

    model_0 = custom_cnn_4(3072)

    decompo_list_1 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0] # use for vgg-11
    model_1 = custom_cnn_4(3072, decompo_list = decompo_list_1)

    decompo_list_2 = [0, 0, 0, 0, 0, 1, 2, 1, 2, 4, 0] # use for vgg-11
    # decompo_list_2 = [0, 0, 0, 0, 0, 1, 2, 1, 1, 4, 0] # use for vgg-11
    model_2 = custom_cnn_4(3072, decompo_list = decompo_list_2)

    # decompo_list_3 = [0, 0, 3, 4, 1, 1, 1, 2, 3, 0, 0] # use for vgg-11
    # # decompo_list_3 = [0, 0, 3, 4, 1, 1, 2, 1, 1, 4, 0] # use for vgg-11

    # deepen_list_3 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] # use for vgg-11

    # skipcon_list_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] # use for vgg-11

    # model_3 = custom_cnn_4(3072, decompo_list = decompo_list_3, deepen_list = deepen_list_3, skipcon_list = skipcon_list_3)

    model_4 = custom_cnn_d1(3072)

    model_5 = custom_cnn_d2(3072)

    model_6 = custom_cnn_d3(3072)

    # model_list = [model_0, model_1, model_2, model_3, model_4, model_5, model_6]

    '''resnet-20'''

    # model_0_res = custom_cnn_5(3072)

    # decompo_list_1 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] # use for resnet-20
    # model_1_res = custom_cnn_5(3072, decompo_list = decompo_list_1)

    # decompo_list_2 = [0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2] # use for resnet-20
    # model_2_res = custom_cnn_5(3072, decompo_list = decompo_list_2)

    # decompo_list_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 0, 2, 2, 0, 3] # use for resnet-20

    # deepen_list_3 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # use for resnet-20

    # model_3_res = custom_cnn_5(3072, decompo_list = decompo_list_3, deepen_list = deepen_list_3)

    # model_4_res = custom_cnn_d4(3072)

    # model_5_res = custom_cnn_d5(3072)

    # model_6_res = custom_cnn_d6(3072)

    # model_list = [model_0_res, model_1_res, model_2_res, model_3_res, model_4_res, model_5_res, model_6_res]

    '''extravgg'''
    decompo_list_3 = [0, 0, 2, 4, 0, 4, 4, 0, 1, 4, 3] 

    deepen_list_3 = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    decompo_list_1 = [2, 0, 1, 4, 0, 4, 3, 0, 1, 0, 4]

    deepen_list_1 = [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]

    decompo_list_2 = [0, 4, 0, 0, 4, 1, 3, 0, 2, 1, 4]

    deepen_list_2 = [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    model_7 = custom_cnn_4(3072, decompo_list = decompo_list_1, deepen_list = deepen_list_1)

    model_8 = custom_cnn_4(3072, decompo_list = decompo_list_2, deepen_list = deepen_list_2)

    model_9 = custom_cnn_4(3072, decompo_list = decompo_list_3, deepen_list = deepen_list_3)
    
    widen_list_1 = [1.0] * len(deepen_list_2)

    widen_list_1[0] = 1.25
    widen_list_1[2] = 1.25
    widen_list_1[4] = 1.25

    widen_list_2 = [1.0] * len(deepen_list_2)

    widen_list_2[0] = 1.5
    widen_list_2[2] = 1.5
    widen_list_2[4] = 1.5

    kerneladd_list_3 = [0] * len(deepen_list_2)
    kerneladd_list_3[0] = 1
    kerneladd_list_3[2] = 1
    kerneladd_list_3[4] = 1

    model_7d = custom_cnn_4(3072, decompo_list = decompo_list_1, deepen_list = deepen_list_1, widen_list = widen_list_1)

    model_8d = custom_cnn_4(3072, decompo_list = decompo_list_2, deepen_list = deepen_list_2, widen_list = widen_list_2)

    model_9d = custom_cnn_4(3072, decompo_list = decompo_list_3, deepen_list = deepen_list_3, widen_list = widen_list_2, kerneladd_list = kerneladd_list_3)

    model_list = [model_0, model_1, model_2, model_4, model_5, model_6, model_7, model_7d, model_8d, model_9d]
    model_list = reversed(model_list)
    # model_list = [model_8, model_9]
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss() # Use NLL because we already have softmax in model
    criterion = criterion.cuda()

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])

    cifar10_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]))

    cifar10_val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
            cifar10_train_dataset,
            batch_size=128, shuffle=True,
            num_workers=4, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            cifar10_val_dataset,
            batch_size=128, shuffle=False,
            num_workers=4, pin_memory=True)

    for model_id, model in enumerate(model_list):
        best_prec1 = 0
        model.apply(init_weights)
        model = model.to(cuda_device)
        # optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        for epoch in range(0, 200):

            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, logger)

            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, logger)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1

            if is_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, is_best, filename=os.path.join("./", 'best_checkpoint_newvgg_model{}_seed{}.tar'.format(model_id, random_seed)))
            best_prec1 = max(prec1, best_prec1)
        del model, optimizer

def train(train_loader, model, criterion, optimizer, epoch, logger):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    torch.autograd.set_detect_anomaly(True)

    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=False)
        target = target.cuda(non_blocking=False)

        # compute output
        optimizer.zero_grad()

        output = model(input)

        loss = criterion(output, target)

        loss.backward()
        
        optimizer.step()

        output = output.float()
        loss = loss.float()
        
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 50 == 0:
            logger.debug('Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
        del loss, output

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    if epoch%30==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2


def validate(val_loader, model, criterion, logger):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda(non_blocking=False)
        target = target.cuda(non_blocking=False)

        # compute output
        with torch.no_grad():
            output = model(input)
            loss = criterion(output, target)

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            logger.debug('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time, loss=losses,
                      top1=top1))

    logger.debug(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


if __name__ == '__main__':
    main()