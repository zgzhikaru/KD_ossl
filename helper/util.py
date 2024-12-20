from __future__ import print_function

import torch
import numpy as np
import torch.nn.functional as F


is_sorted = lambda a: np.all(a[:-1] <= a[1:])

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,), output_cls=None):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        if output_cls is not None: 
            # Mask-out non-existing class before computing accuracy
            _, pred = output[:,output_cls].topk(maxk, 1, True, True)
        else:
            _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def precision(output, target, topk=(1,)):
    # TODO: Implement
    return 0

def recall(output, target, topk=(1,)):
    # TODO: Implement
    return 0


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


if __name__ == '__main__':

    pass
