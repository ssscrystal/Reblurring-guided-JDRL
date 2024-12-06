import torch
import torch.nn as nn
import utils.arch_util as arch_util
import functools


class Pred(nn.Module):
    def __init__(self, nf=32):
        super(Pred, self).__init__()

        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.block1 = basic_block()
        self.block2 = basic_block()
        self.block3 = basic_block()
        self.out = nn.Conv2d(32, 35, 3, 1, 1, bias=True)

        self._init_weights()

    def forward(self, im_b):
        x = self.conv_first(im_b)
        x = self.block3(self.block2(self.block1(x)))
        core = self.out(x)
        return core

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.1)