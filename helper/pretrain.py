from __future__ import print_function, division

import time
import sys
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from .util import AverageMeter


def init(model_s, model_t, init_modules, criterion, train_loader, logger, opt, criterion_cls=None):
    model_t.eval()
    model_s.eval()
    init_modules.train()

    if torch.cuda.is_available():
        model_s.cuda()
        model_t.cuda()
        init_modules.cuda()
        cudnn.benchmark = True

    if opt.arch in ['resnet8', 'resnet14', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110',
                       'resnet8x4', 'resnet32x4', 'wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2'] and \
            opt.distill == 'factor':
        lr = 0.01
    else:
        lr = opt.learning_rate
    optimizer = optim.SGD(init_modules.parameters(),
                          lr=lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    for epoch in range(1, opt.init_epochs + 1):
        batch_time.reset()
        data_time.reset()
        losses.reset()
        end = time.time()
        for idx, data in enumerate(train_loader):   # TODO: Adapt utrain_loader
            curr_iter = idx + epoch * len(train_loader)
            if opt.distill in ['crd']:
                input, target, index, contrast_idx = data
            else:
                input, target, index = data
            data_time.update(time.time() - end)

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                index = index.cuda()
                if opt.distill in ['crd']:
                    contrast_idx = contrast_idx.cuda()

            # ============= forward ==============
            preact = (opt.distill == 'abound')
            feat_s, _ = model_s(input, is_feat=True, preact=preact)
            with torch.no_grad():
                feat_t, _ = model_t(input, is_feat=True, preact=preact)
                feat_t = [f.detach() for f in feat_t]

            if opt.distill == 'abound':
                g_s = init_modules[0](feat_s[1:-1])
                g_t = feat_t[1:-1]
                loss_group = criterion(g_s, g_t)
                loss = sum(loss_group)
            elif opt.distill == 'factor':
                f_t = feat_t[-2]
                _, f_t_rec = init_modules[0](f_t)
                loss = criterion(f_t_rec, f_t)
            elif opt.distill == 'fsp':
                loss_group = criterion(feat_s[:-1], feat_t[:-1])
                loss = sum(loss_group)
            elif opt.distill == 'pad':
                f_t = feat_t[-1]
                f_s = feat_s[-1]
                loss_kd = criterion(f_s, f_t)
                if criterion_cls is not None:
                    loss_cls = criterion_cls(f_s, target)
                    loss = opt.gamma *loss_cls  + opt.pre_beta * loss_kd
                else:
                    loss = loss_kd
            else:
                raise NotImplemented('Not supported in init training: {}'.format(opt.distill))

            losses.update(loss.item(), input.size(0))

            # ===================backward=====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if criterion_cls is not None and opt.gamma:
                logger.add_scalar('pretrain/ce_loss', loss_cls.detach().item(), curr_iter)
            if opt.pre_beta:
                logger.add_scalar('pretrain/distill_loss', loss_kd.detach().item(), curr_iter)

            # print info
            if idx % opt.print_freq == 0:
                print('ssl Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
                sys.stdout.flush()

        # end of epoch
        logger.add_scalar('pretrain/train_loss', losses.avg, epoch)
        print('Epoch: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'losses: {losses.val:.3f} ({losses.avg:.3f})'.format(
               epoch, opt.init_epochs, batch_time=batch_time, losses=losses))
        sys.stdout.flush()
