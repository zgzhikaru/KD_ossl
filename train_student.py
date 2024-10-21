"""
the general training framework
"""

from __future__ import print_function

import os
import argparse
import time
import logging
import json

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import model_dict as MODEL_DICT
from models.util import ConvReg, LinearEmbed
from models.util import Connector, Translator, Paraphraser
from distiller_zoo.AIN import transfer_conv, statm_loss

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_test, DATASET_CLASS, DATASET_SAMPLES
from helper.util import adjust_learning_rate
from distiller_zoo import DistillKL, HintLoss, Attention, Similarity, Correlation, VIDLoss, RKDLoss
from distiller_zoo import PKT, ABLoss, FactorTransfer, KDSVD, FSP, NSTLoss,CRDLoss

from helper.loops import train_ssldistill as train_ssl

from helper.loops import validate
from helper.util import accuracy
from helper.pretrain import init
from utils.utils import init_logging, load_model


def parse_option():

    parser = argparse.ArgumentParser('argument for training')
    
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=120, help='save frequency')
    
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # labeled dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='Target dataset')
    parser.add_argument('--num_classes', type=int, default=200, help='number of classes in the target dataset')
  
    # select unlabeled dataset
    parser.add_argument('--ood', type=str, default='tin', choices=['tin', 'places', 'None'], help='The augment Out-of-distribution dataset')
    parser.add_argument('--num_ood_class', type=int, default=200, help='number of classes in the augment dataset')

    parser.add_argument('--samples_per_cls', type=int, action='store', help='Number of samples per class in all datasets')
    parser.add_argument('--lb_prop', type=float, default=1.0, help='labeled sample proportion within target dataset')
    parser.add_argument('--split_seed', type=int, default=12345, help='random seed for reproducing dataset split')

    parser.add_argument('--include-labeled', default=True, help='include labeled-set data into unlabeled-set')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # model
    parser.add_argument('--arch', type=str, default='wrn_40_1',
                        choices=['resnet8x4', 'wrn_40_1',  'ShuffleV1'], help="Student model architecture")
    parser.add_argument('--tc_path', type=str, required=True, help='path to the pretrained teacher model parameters')
    parser.add_argument('--save_path', type=str, default="results/student/", help='path to the pretrained teacher model parameters')


    # distillation
    parser.add_argument('--distill', type=str, default='kd')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--temp', type=float, default=4, help='temperature for KD distillation')

    # NCE distillation
    parser.add_argument('--feat_dim', default=128, type=int, help='feature dimension')
    parser.add_argument('--mode', default='exact', type=str, choices=['exact', 'relax'])
    parser.add_argument('--nce_k', default=16384, type=int, help='number of negative samples for NCE')
    parser.add_argument('--nce_t', default=0.07, type=float, help='temperature parameter for softmax')
    parser.add_argument('--nce_m', default=0.5, type=float, help='momentum for non-parametric updates')

    # hint layer
    parser.add_argument('--hint_layer', default=2, type=int, choices=[0, 1, 2, 3, 4])

    parser.add_argument('-t', '--trial', type=int, default=0, help='the experiment id')

    opt = parser.parse_args()

    # set different learning rate from these 4 models
    if opt.arch in ['MobileNetV2', 'ShuffleV1', 'ShuffleV2']:
        opt.learning_rate = 0.01

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))


    # Load attributes from teacher model file
    state_dict = torch.load(opt.tc_path)
    teacher_name, num_tc_head = state_dict["name"], state_dict["num_head"]
    #model_split_seed = state_dict["split_seed"]

    assert opt.num_classes > 0, "Must specify a positive number of class"

    #if opt.num_classes < num_tc_head:
    #    pass
    assert not num_tc_head < opt.num_classes, "Number of teacher heads are insufficient to classify requested number of class"
    #assert num_tc_head == opt.num_classes, "Teacher and student must have same number of heads"
    #assert model_split_seed == opt.split_seed, "Warning: Teacher and student are not trained on the same target data split"


    # Initialize saving directories
    #num_total_class = opt.num_classes + opt.num_ood_cls
    smp_val = DATASET_SAMPLES[opt.dataset]//DATASET_CLASS[opt.dataset]
    opt.model_name = 'M:{}_T:{}_arch:{}_ID:{}_ic:{}_OOD:{}_oc:{}_smp:{}_lb:{}_split:{}_trial:{}'.format(opt.distill, teacher_name, opt.arch, 
                                                                                  opt.dataset, opt.num_classes,
                                                                                  opt.ood, opt.num_ood_class, 
                                                                                  smp_val, opt.lb_prop,
                                                                                  opt.split_seed, opt.trial                                                                                 
                                                                                  )
    # set the path
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
        opt.num_classes
        # TODO: Get class_idx and pass the shared argument into both train & test set constructor.
        train_loader, utrain_loader = get_cifar100_dataloaders(batch_size=opt.batch_size, num_workers=opt.num_workers,
                                                                samples_per_cls=opt.samples_per_cls, num_id_class=opt.num_classes,
                                                                ood=opt.ood, num_ood_class=opt.num_ood_class,
                                                                lb_prop=opt.lb_prop, include_labeled=opt.include_labeled, 
                                                                split_seed=opt.split_seed, class_split_seed=opt.split_seed)
        val_loader = get_cifar100_test(batch_size=opt.batch_size//2,
                                        num_workers=opt.num_workers//2,
                                        num_classes=opt.num_classes,
                                        split_seed=opt.split_seed)
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_model(opt.tc_path)
    model_s = MODEL_DICT[opt.arch](num_classes=opt.num_classes)

    data = torch.randn(2, 3, 32, 32)
    model_t.eval()
    model_s.eval()
    feat_t, logit_t = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)
    tc_classes = logit_t.shape[-1]

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.temp)

    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.temp)
    elif opt.distill == 'hint':
        criterion_kd = HintLoss()
        if opt.hint_layer < 4:  # Feature-map layers
            connector = ConvReg(feat_s[opt.hint_layer].shape, feat_t[opt.hint_layer].shape)
        else:       # Embedding Feature layer
            opt.s_dim = feat_s[opt.hint_layer].shape[1]
            opt.t_dim = feat_t[opt.hint_layer].shape[1]
            connector = transfer_conv(opt.s_dim, opt.t_dim)
        module_list.append(connector)
        trainable_list.append(connector)
        
    elif opt.distill == 'crd':
        opt.s_dim = feat_s[-1].shape[1]
        opt.t_dim = feat_t[-1].shape[1]
        opt.n_data = len(train_loader.dataset)   #n_data
        criterion_kd = CRDLoss(opt)
        module_list.append(criterion_kd.embed_s)
        module_list.append(criterion_kd.embed_t)
        trainable_list.append(criterion_kd.embed_s)
        trainable_list.append(criterion_kd.embed_t)
    elif 'srd' in opt.distill:
        opt.s_dim = feat_s[-2].shape[1]
        opt.t_dim = feat_t[-2].shape[1]
        #opt.s_dim = feat_s[opt.hint_layer].shape[1]
        #opt.t_dim = feat_t[opt.hint_layer].shape[1]
        connector = transfer_conv(opt.s_dim, opt.t_dim)
        module_list.append(connector)
        # add this because connector need to to updated
        trainable_list.append(connector)
        criterion_kd = statm_loss()
    elif opt.distill == 'attention':
        criterion_kd = Attention()
    elif opt.distill == 'nst':
        criterion_kd = NSTLoss()
    elif opt.distill == 'similarity':
        criterion_kd = Similarity()
    elif opt.distill == 'rkd':
        criterion_kd = RKDLoss()
    elif opt.distill == 'pkt':
        criterion_kd = PKT()
    elif opt.distill == 'kdsvd':
        criterion_kd = KDSVD()
    elif opt.distill == 'correlation':
        criterion_kd = Correlation()
        embed_s = LinearEmbed(feat_s[-1].shape[1], opt.feat_dim)
        embed_t = LinearEmbed(feat_t[-1].shape[1], opt.feat_dim)
        module_list.append(embed_s)
        module_list.append(embed_t)
        trainable_list.append(embed_s)
        trainable_list.append(embed_t)
    elif opt.distill == 'vid':
        s_n = [f.shape[1] for f in feat_s[1:-1]]
        t_n = [f.shape[1] for f in feat_t[1:-1]]
        criterion_kd = nn.ModuleList(
            [VIDLoss(s, t, t) for s, t in zip(s_n, t_n)]
        )
        # add this as some parameters in VIDLoss need to be updated
        trainable_list.append(criterion_kd)
    elif opt.distill == 'abound':
        s_shapes = [f.shape for f in feat_s[1:-1]]
        t_shapes = [f.shape for f in feat_t[1:-1]]
        connector = Connector(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(connector)
        init_trainable_list.append(model_s.get_feat_modules())
        criterion_kd = ABLoss(len(feat_s[1:-1]))
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification
        module_list.append(connector)
    elif opt.distill == 'factor':
        s_shape = feat_s[-2].shape
        t_shape = feat_t[-2].shape
        paraphraser = Paraphraser(t_shape)
        translator = Translator(s_shape, t_shape)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(paraphraser)
        criterion_init = nn.MSELoss()
        init(model_s, model_t, init_trainable_list, criterion_init, train_loader, logger, opt)
        # classification
        criterion_kd = FactorTransfer()
        module_list.append(translator)
        module_list.append(paraphraser)
        trainable_list.append(translator)
    elif opt.distill == 'fsp':
        s_shapes = [s.shape for s in feat_s[:-1]]
        t_shapes = [t.shape for t in feat_t[:-1]]
        criterion_kd = FSP(s_shapes, t_shapes)
        # init stage training
        init_trainable_list = nn.ModuleList([])
        init_trainable_list.append(model_s.get_feat_modules())
        init(model_s, model_t, init_trainable_list, criterion_kd, train_loader, logger, opt)
        # classification training
        pass
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)  # classification loss
    criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)   # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    #metric_dict, test_loss = validate(val_loader, model_t, criterion_cls, opt)
    assert len(val_loader.dataset.classes) == opt.num_classes
    if opt.num_classes< tc_classes:
        print("Casting teacher's heads({}) to test classes({})".format(tc_classes, opt.num_classes))
        eval_metrics = {"acc1": lambda y_hat, y: accuracy(y_hat, y, output_cls=val_loader.dataset.classes)}
    else:
        eval_metrics = {"acc1": accuracy}
    metric_dict, _ = validate(val_loader, model_t, None, opt, metrics=eval_metrics)
    teacher_acc = metric_dict["acc1"]
    print('teacher accuracy: ', teacher_acc)


    # NOTE: Originally total_data = len(train_loader.dataset)//batch_size
    """
    u_data_len = len(utrain_loader.dataset) if utrain_loader is not None else 0
    #total_data = (len(train_loader.dataset) + u_data_len)//2 if not opt.include_labeled else u_data_len//2 
    total_data = u_data_len
    if opt.include_labeled:
        total_data += len(train_loader.dataset)
    iter_per_epoch = total_data // opt.batch_size
    """
    iter_per_epoch = len(train_loader.dataset) // opt.batch_size  #DATASET_SAMPLES[opt.dataset] // opt.batch_size
    print("iter per epoch:", iter_per_epoch)

    # routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train_ssl(epoch, iter_per_epoch, train_loader, utrain_loader, module_list, criterion_list, optimizer, opt, logger=logger)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))
        
        #test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)  # TODO: Input (acc1, acc5) as two metrics
        metric_dict, test_loss = validate(val_loader, model_s, criterion_cls, opt)
        test_acc = metric_dict["acc1"]

        logger.add_scalar('train/train_loss', train_loss, epoch)
        logger.add_scalar('train/train_acc', train_acc, epoch)
        logger.add_scalar('test/test_acc', test_acc, epoch)
        logger.add_scalar('test/test_loss', test_loss, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
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
                'model': model_s.state_dict(),
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
        'model': model_s.state_dict(),
        'optimizer': optimizer.state_dict(),
        'name': opt.arch, 
        'num_head': opt.num_classes,
        'split_seed': opt.split_seed,
    }
    save_file = os.path.join(opt.model_path, '{}_last.pth'.format(opt.arch))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
