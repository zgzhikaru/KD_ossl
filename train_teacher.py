from __future__ import print_function

import os
import argparse
import socket
import time
import logging
import json

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import model_dict as MODEL_DICT

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_test, DATASET_SAMPLES, DATASET_CLASS
from helper.util import adjust_learning_rate
from helper.loops import train_vanilla as train, validate
from utils.utils import init_logging

"""
Fully Supervised training for a single architecture on single source of labeled data.
Can be used for obtaining either a pretrained teacher or a student baseline.
"""

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=120, help='save frequency')
    
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # labeled dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--num_classes', type=int, action='store', help='number of classes in the split dataset')
    
    parser.add_argument('--num_samples', type=int, action='store', help='Number of samples per class in all datasets')
    parser.add_argument('--split_seed', type=int, default=12345, help='random seed for reproducing dataset split')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    
    # model
    parser.add_argument('--arch', type=str, default='resnet110',
                        choices=['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                                 'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19',
                                 'MobileNetV2', 'ShuffleV1', 'ShuffleV2', ])
    parser.add_argument('--save_path', type=str, default='results/teacher/', help='Saving path of result model')
    
 
    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()
    
    # set different learning rate from these 4 models
    if opt.arch in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.num_classes is None:
        opt.num_classes = DATASET_CLASS[opt.dataset]
    if opt.num_samples is None:
        opt.num_samples = DATASET_SAMPLES[opt.dataset]

    method = 'supCE'
    opt.model_name = 'M:{}_arch:{}_ID:{}_ic:{}_total:{}_trial:{}'.format(method, opt.arch, opt.dataset, 
                                                                opt.num_classes, opt.num_samples,
                                                                opt.trial                                                                                
                                                                )  

    opt.model_path = os.path.join(opt.save_path, 'model', opt.model_name)
    os.makedirs(opt.model_path, exist_ok=True)

    opt.log_path = os.path.join(opt.save_path, 'log', opt.model_name)
    os.makedirs(opt.log_path, exist_ok=True)

    init_logging(log_root=logging.getLogger(), models_root=opt.log_path)

    # Save configuration in logging folder
    config_name = os.path.join(opt.log_path, 'config.json')
    with open(config_name, "w") as outfile: 
        json.dump(vars(opt), outfile)

    opt.tb_path = os.path.join(opt.save_path, 'tensorboard', opt.model_name)
    os.makedirs(opt.tb_path, exist_ok=True)

    return opt


def main():
    best_acc = 0
    opt = parse_option()

    logger = SummaryWriter(opt.tb_path) 

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, _ = \
            get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,
                                    num_id_class=opt.num_classes, num_ood_class=0,
                                    num_samples=opt.num_samples, num_labels=opt.num_samples,
                                    split_seed=opt.split_seed, class_split_seed=opt.split_seed)
        val_loader = get_cifar100_test(batch_size=opt.batch_size//2,
                                        num_workers=opt.num_workers//2,
                                        num_classes=opt.num_classes,
                                        #num_samples=opt.num_samples,
                                        split_seed=opt.split_seed)

    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = MODEL_DICT[opt.arch](num_classes=opt.num_classes)

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True


    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, model, criterion, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        #test_acc, test_acc_top5, test_loss = validate(val_loader, model, criterion, opt)
        metric_dict, test_loss = validate(val_loader, model, criterion, opt)
        test_acc = metric_dict["acc1"]

        logger.add_scalar('train/train_acc', train_acc, epoch)
        logger.add_scalar('train/train_loss', train_loss, epoch)
        logger.add_scalar('test/test_acc', test_acc, epoch)
        logger.add_scalar('test/test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'name': opt.arch,
                'num_head': opt.num_classes,
                'split_seed': opt.split_seed,
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.model_path, '{}_best.pth'.format(opt.arch))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'name': opt.arch,
                'num_head': opt.num_classes,
                'split_seed': opt.split_seed,
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

        msg = "Epoch %d test_acc %.3f, best_acc %.3f" % (epoch, test_acc, best_acc)
        logging.info(msg)

    # save model
    print('best accuracy:', best_acc)
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'name': opt.arch,
        'num_head': opt.num_classes,
        'split_seed': opt.split_seed,
    }
    save_file = os.path.join(opt.model_path, '{}_last.pth'.format(opt.arch))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
