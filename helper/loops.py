from __future__ import print_function, division

import sys
import time
import torch
import torch.nn.functional as F
from .util import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        losses.update(loss.item(), input.size(0))
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()

    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
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

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence

        if opt.distill in ['crd']:
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.detach().item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('bl Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, losses.avg


def train_ssldistill(epoch, iter_per_epoch, train_loader, utrain_loader, 
                     module_list, criterion_list, optimizer, opt, logger=None):
    """One epoch distillation"""
    for module in module_list:
        module.train()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]
    model_t = model_t.eval()

    data = torch.randn(2, 3, 32, 32).cuda()
    logit_t = model_t(data)

    class_idx = None
    tc_cls = logit_t.shape[-1]
    lb_dataset = train_loader.dataset
    if lb_dataset.num_classes < tc_cls:
        print("Casting teacher's head to {} classes".format(len(lb_dataset.classes)))
        class_idx = lb_dataset.classes

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    
    ul_ood_exist = utrain_loader is not None

    u_iter = iter(utrain_loader) if ul_ood_exist else None
    l_iter = iter(train_loader)

    #for batch_idx in range(len(train_loader)):
    for batch_idx in range(iter_per_epoch):
        # ==================labeled data================
        curr_iter = epoch * iter_per_epoch + batch_idx

        input_l, target_l, index_l = next(l_iter)

        batch_size = input_l.shape[0]

        input_l = input_l.float().cuda()
        target_l = target_l.cuda()

        inputs = input_l

        # ==================unlabeled data================
        if ul_ood_exist: 
            #(input_ul, _), index_u = data
            input_ul, index_u = next(u_iter)

            input_ul = input_ul.float().cuda()

            inputs = torch.cat((input_l, input_ul), dim=0).cuda()
            
        data_time.update(time.time() - end)


        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True


        feat_s, logit_s = model_s(inputs, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_t, logit_t = model_t(inputs, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        #loss_cls = criterion_cls(logit_s[:batch_size], target_l)
        logit_ce = logit_s[:batch_size]
        loss_cls = criterion_cls(logit_ce, target_l)
        if ul_ood_exist and opt.include_labeled:     # Compute only distillation loss for unlabeled set
            feat_t, logit_t = [feat[batch_size:] for feat in feat_t], logit_t[batch_size:]
            feat_s, logit_s = [feat[batch_size:] for feat in feat_s], logit_s[batch_size:]
            
        # Reindex logit_s given output class_idx of student; Match student & teacher's head
        if class_idx is not None:
            logit_t = logit_t[:, class_idx] 
            # NOTE: Assuming teacher's class are sorted in the same order as student class idx

        # Compute distillation loss
        if opt.alpha > 0 or opt.beta > 0:
            # for all
            loss_div = criterion_div(logit_s, logit_t)

            # other kd beyond KL divergence
            if opt.distill == 'kd':
                loss_kd = 0
            elif opt.distill == 'hint':
                f_s = module_list[1](feat_s[opt.hint_layer])
                f_t = feat_t[opt.hint_layer]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'srd':
                f_s = feat_s[-2]
                f_s = module_list[1](f_s)
                f_t = feat_t[-2]
                logit_tc = model_t(x=None, feat_s=f_s, feat_t=f_t)
                if opt.model_s=='ShuffleV1':
                    loss_kd = criterion_kd(f_s, f_t) * 10+ F.mse_loss(logit_tc, logit_t)
                elif opt.model_s  in ['resnet8x4','wrn_40_1']:
                    #loss_kd = F.mse_loss(logit_tc, logit_t) # + criterion_kd(f_s, f_t) 
                    if opt.beta == 0:
                        loss_div = criterion_kd(f_s, f_t)
                        loss_kd = 0
                    else:
                        loss_div = 0
                        loss_kd = opt.alpha/opt.beta  * criterion_kd(f_s, f_t) +  F.mse_loss(logit_tc, logit_t)
                else:
                    raise NotImplementedError

            elif opt.distill == 'srdv2':
                f_s = feat_s[-2]
                f_s = module_list[1](f_s)
                f_t = feat_t[-2]
                logit_tc = model_t(x=None, feat_s=f_s, feat_t=f_t)
                loss_kd = criterion_kd(f_s, f_t) * 10+ F.mse_loss(logit_tc, logit_t)

            elif opt.distill == 'attention':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'nst':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'similarity':
                g_s = [feat_s[-2]]
                g_t = [feat_t[-2]]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'rkd':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'pkt':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'kdsvd':
                g_s = feat_s[1:-1]
                g_t = feat_t[1:-1]
                loss_group = criterion_kd(g_s, g_t)
                loss_kd = sum(loss_group)
            elif opt.distill == 'correlation':
                f_s = module_list[1](feat_s[-1])
                f_t = module_list[2](feat_t[-1])
                loss_kd = criterion_kd(f_s, f_t)
            elif opt.distill == 'vid':
                if opt.hint_layer < 4:
                    g_s = feat_s[1:-1]
                    g_t = feat_t[1:-1]
                    loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
                    loss_kd = sum(loss_group)
                else:   # Distill from the penultimate layer
                    f_s = feat_s[-1]
                    f_t = feat_t[-1]
                    loss_kd = criterion_kd(f_s, f_t)
                
                logger.add_scalar('train/log_scale', criterion_kd.log_scale.detach().item(), curr_iter)
            elif opt.distill == 'abound':
                # can also add loss to this stage
                loss_kd = 0
            elif opt.distill == 'fsp':
                # can also add loss to this stage
                loss_kd = 0
            elif opt.distill == 'factor':
                factor_s = module_list[1](feat_s[-2])
                factor_t = module_list[2](feat_t[-2], is_factor=True)
                loss_kd = criterion_kd(factor_s, factor_t)
            elif opt.distill == 'pad':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
                #loss_kd = criterion_kd(f_s, f_t)
                loss_kd, avg_logvar = criterion_kd(f_s, f_t, return_logvar=True)
                logger.add_scalar('train/avg_logvar', avg_logvar.detach().item(), curr_iter)
            else:
                raise NotImplementedError(opt.distill)
            
            if ul_ood_exist and opt.include_labeled:    # NOTE: Multiply by two to account for double batch-size on distillation loss
                loss = opt.gamma * loss_cls + (opt.alpha * loss_div + opt.beta * loss_kd) * 2
            else:
                loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd
            #div_losses.update(loss_div.detach().item(), inputs.size(0))
            #kd_losses.update(loss_kd.detach().item(), inputs.size(0))
            
            if loss_div and opt.alpha:
                logger.add_scalar('train/kl_loss', loss_div.detach().item(), curr_iter)
            if loss_kd and opt.beta:
                logger.add_scalar('train/distill_loss', loss_kd.detach().item(), curr_iter)
        else:
            loss = loss_cls
        
        #ce_losses.update(loss_cls.detach().item(), inputs.size(0))
        logger.add_scalar('train/ce_loss', loss_cls.detach().item(), curr_iter)

        #acc1, acc5 = accuracy(logit_s[:batch_size], target_l, topk=(1, 5))
        acc1, acc5 = accuracy(logit_ce, target_l, topk=(1, 5))
        losses.update(loss.detach().item(), inputs.size(0))
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if batch_idx % opt.print_freq == 0:
            print('ssl Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, batch_idx, iter_per_epoch, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    
    #logger.add_scalar('train/ce_loss', ce_losses.avg, epoch)
    #logger.add_scalar('train/kl_loss', div_losses.avg, epoch)
    #logger.add_scalar('train/distill_loss', kd_losses.avg, epoch)

    return top1.avg, losses.avg



def validate(val_loader, model, criterion, opt, metrics={"acc1": accuracy}):
    """validation"""
    batch_time = AverageMeter()
    if criterion is not None:
        losses = AverageMeter()
    metric_avg = {metric: AverageMeter() for metric in metrics}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        #for idx, (input, target) in enumerate(val_loader):
        for idx, (input, target, _) in enumerate(val_loader):
            batch_size = input.size(0)
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            if criterion is not None:
                loss = criterion(output, target)
                losses.update(loss.item(), batch_size)

            # measure accuracy and record loss
            #acc1, acc5 = accuracy(output, target, topk=(1, 5))

            for m in metrics:
                #results = metric(output, target)
                res = metrics[m](output, target)
                #print(res)
                metric_avg[m].update(res[0].item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                msg =  ('Test: [{0}/{1}]\t' + \
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t').format(
                            idx, len(val_loader), batch_time=batch_time)
                if criterion is not None:
                    msg += 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(loss=losses)
                for m in metrics:
                    msg += '{name} {metr.val:.3f} ({metr.avg:.3f})\t'.format(name=m, metr=metric_avg[m])
                print(msg)

        msg = 'VAL * '
        for m in metrics:
            msg += '{name} {val.avg:.3f}\t'.format(name=m, val=metric_avg[m])
        print(msg)
            

    return {metric: metric_avg[metric].avg for metric in metric_avg}, \
            losses.avg if criterion is not None else None
