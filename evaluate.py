"""
the general evaluation framework
"""

from __future__ import print_function

import os
import argparse
import time
import logging

import numpy as np
import pandas as pd
import json

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict
from models.util import ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser
from distiller_zoo.AIN import transfer_conv, statm_loss
from dataset.cifar100 import get_cifar100_test, DATASET_CLASS
from helper.util import adjust_learning_rate

from helper.loops import validate
from helper.util import accuracy
from utils.utils import get_model_name, init_logging


def parse_option():

    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')

    #parser.add_argument('--save_path', type=int, default=40, help='save ')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    # labeled dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--split_seed', type=int, default=12345, help='random seed for reproducing dataset split')

    # select unlabeled dataset
    parser.add_argument('--num_eval_classes', type=int, default=50, help='number of classes in the augment dataset')
    parser.add_argument('--lb_prop', type=float, default=1.0, help='labeled sample proportion within target dataset')

    # model
    parser.add_argument('--model', type=str, default='resnet8x4',
                        choices=['resnet8x4', 'wrn_40_1',  'ShuffleV1'])
    parser.add_argument('--load_path', type=str, default=None, help='teacher model snapshot')


    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.model_s in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    #opt.load_path = './Results/kdssl/student_model'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    return opt


def main():

    opt = parse_option()
    
    assert os.path.exists(opt.path), "Model path does not exists"
    parent_path = opt.path.dirname(opt.path)

    # Loading model parameters
    model_t, num_model_class = get_model_name(opt.path)
    model = model_dict[model_t](num_classes=num_model_class)
    model.load_state_dict(torch.load(opt.path)['model'])
    assert model.fc.out_features == num_model_class, "Number of classifier heads does not match"
  
    # Setup test dataset
    test_loader = get_cifar100_test(batch_size=opt.batch_size,
                                    num_workers=opt.num_workers,
                                    is_instance=True, is_sample=False,
                                    num_classes=opt.num_eval_classes,
                                    split_seed=opt.split_seed)
    assert len(test_loader.dataset.classes) == opt.num_eval_classes, "Wrong number of class split"
    
    # Check model num classifier head; Prune head to match num test class(100 - num_unseen_class)

    eval_classes = test_loader.dataset.classes
    if len(eval_classes) < DATASET_CLASS['cifar100']:     # Redundant classifier head exists
        eval_metrics = {"acc1": lambda y_hat, y: accuracy(y_hat, y, output_cls=eval_classes)}
    else:
        eval_metrics = {"acc1": accuracy}
    """
    eval_metrics = {
        'accuracy': accuracy, 
        'precision': precision, 
        'recall': recall, 
    }
    """
    metric_vals = validate(test_loader, model, None, opt, metrics=eval_metrics)

    
    print("Test top-1 accuracy: ", metric_vals["acc1"])

    save_name = os.path.join(parent_path, "evaluation.json")
    with open(save_name, "w") as outfile: 
        json.dump(metric_vals, outfile)

    print("Saved to %s" % save_name)


if __name__ == '__main__':
    main()