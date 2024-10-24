import torch
import math
import torch.nn as nn

"""Knowledge distillation via softmax regression representation learning
code:https://github.com/jingyang2017/KD_SRRL
"""

class transfer_conv(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Connectors = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feature),
            nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, student):
        if len(student.size()) == 2:
            student = student.view(student.shape[0], student.shape[1], 1, 1)
            student = self.Connectors(student)
            return student.squeeze((-1,-2))
        else:
            return self.Connectors(student)
            


class statm_loss(nn.Module):
    def __init__(self):
        super(statm_loss, self).__init__()

    def forward(self,x, y):
        x = x.view(x.size(0),x.size(1),-1)
        y = y.view(y.size(0),y.size(1),-1)
        x_mean = x.mean(dim=2)#BC
        y_mean = y.mean(dim=2)
        mean_gap = (x_mean-y_mean).pow(2).mean(1)
        return mean_gap.mean()
