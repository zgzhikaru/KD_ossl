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
from dataset.cifar100 import get_cifar100_test, get_class_idx, DATASET_CLASS
from helper.util import adjust_learning_rate

from helper.loops import validate
from helper.util import accuracy, is_sorted
from utils.utils import get_model_name, init_logging


def parse_option():

    parser = argparse.ArgumentParser('argument for evaluation')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')

    # labeled dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    parser.add_argument('--split_seed', type=int, default=12345, help='random seed for reproducing test dataset split')
    #parser.add_argument('--model_split_seed', type=int, default=12345, help='random seed for reproducing train dataset split used by the model')

    # select unlabeled dataset
    parser.add_argument('--num_eval_classes', type=int, default=100, help='number of classes in the augment dataset')
    parser.add_argument('--lb_prop', type=float, default=1.0, help='labeled sample proportion within target dataset')

    # model
    #parser.add_argument('--model', type=str, default='resnet8x4',
    #                    choices=['resnet8x4', 'wrn_40_1',  'ShuffleV1'])
    parser.add_argument('--path', type=str, default=None, help='model snapshot')

    opt = parser.parse_args()

    return opt


def main():

    opt = parse_option()

    num_full_class=DATASET_CLASS[opt.dataset]
    
    # Loading model parameters
    assert os.path.exists(opt.path), "Model path does not exists"
    parent_path = os.path.dirname(opt.path)

    state_dict = torch.load(opt.path)

    # NOTE: Consider saving & loading the entire model train class-set; 
    try:
        num_model_head = state_dict["num_head"]  
        #model_name = state_dict["name"]
    except:
        model_name, num_model_head = get_model_name(opt.path)  
        num_model_head = int(num_model_head.split(':')[1])
    print("model_name", model_name)
    print("num_model_head", num_model_head)
    #model_split_seed = state_dict["split_seed"]    # TODO: Consider add split_seed as a saved attribute into the model state_dict
    model_split_seed = opt.split_seed   # NOTE: Assuming model train and test set share the same split seed

    if opt.num_eval_classes > num_model_head:
        print("Number of target test class cannot exceed number of available classifier heads")
        return -1
    

    if num_model_head < num_full_class:
        model_cls = get_class_idx(num_model_head, num_full_class, split_seed=model_split_seed)
        assert len(model_cls) == num_model_head
    else:   # Equal between eval_classes and model_head
        model_cls = np.arange(num_model_head)
    #print("model_cls", model_cls)

    if opt.num_eval_classes < num_model_head:
        test_cls = get_class_idx(opt.num_eval_classes, num_full_class, split_seed=opt.split_seed)
        assert len(test_cls) == opt.num_eval_classes
    else:
        test_cls = model_cls
        
    #print("test_cls", test_cls)

    if np.any(np.setdiff1d(test_cls, model_cls)):
        print("Test set contains more classes than model heads support")
        return -1

    # Load model parameters
    model = model_dict[model_name](num_classes=num_model_head)
    model.load_state_dict(state_dict['model'])
    assert model.fc.out_features == num_model_head, "Number of classifier heads does not match"
  
    if torch.cuda.is_available():
        model = model.cuda()
        cudnn.benchmark = True

    # Setup test dataset
    test_loader = get_cifar100_test(batch_size=opt.batch_size,
                                    num_workers=opt.num_workers,
                                    num_classes=opt.num_eval_classes,
                                    split_seed=opt.split_seed)
    assert len(test_loader.dataset.classes) == opt.num_eval_classes, "Wrong number of class split"
    


    if len(test_cls) < len(model_cls):  # Redundant classifier head exists
        assert is_sorted(test_cls) and is_sorted(model_cls), "Class idxs need to be in sorted order"
        print("Evaluating part of {} heads on {} classes".format(num_model_head, len(test_cls)))
        
        model_cls_idx = np.searchsorted(model_cls, test_cls)    # Map test class idx into train class idx of the model
        output_cls = torch.from_numpy(model_cls_idx)
        if torch.cuda.is_available():
            output_cls = output_cls.cuda()
        eval_metrics = {"acc1": lambda y_hat, y: accuracy(y_hat, y, output_cls=output_cls)}
    else:
        eval_metrics = {"acc1": accuracy}


    metric_vals, _ = validate(test_loader, model, None, opt, metrics=eval_metrics)

    
    print("Test top-1 accuracy: ", metric_vals["acc1"])

    save_name = os.path.join(parent_path, "evaluation.json")
    with open(save_name, "w") as outfile: 
        json.dump(metric_vals, outfile)

    print("Saved to %s" % save_name)


if __name__ == '__main__':
    main()