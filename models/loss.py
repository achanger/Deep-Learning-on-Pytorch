
# define your special Loss Function here.
# 在这里定义一些特殊的损失函数。

import torch
import torch.nn as nn
from torch.nn import functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target, loss_type='jaccard'):
        target = target.type(torch.cuda.FloatTensor)
        smooth = 1e-5
        inse = torch.sum(input * target)

        if loss_type == 'jaccard':
            l = torch.sum(input * input)
            r = torch.sum(target * target)
        elif loss_type == 'sorensen':
            l = torch.sum(input)
            r = torch.sum(target)
        else:
            raise Exception("Unknow loss_type")

        dice = (2. * inse + smooth) / (l + r + smooth)
        dice_loss = 1 - torch.mean(dice)
        return dice_loss


def dice_loss(input, target):
    loss_function = DiceLoss()
    loss = loss_function(input, target, loss_type='jaccard')
    size_average = False
    if size_average:
        loss /= float(target.numel())
    return loss




class _Reduction:
    # NB: Keep this class in sync with enums in THNN/Reduction.h

    @staticmethod
    def get_enum(reduction):
        if reduction == 'none':
            return 0
        if reduction == 'elementwise_mean':
            return 1
        if reduction == 'sum':
            return 2
        raise ValueError(reduction + " is not a valid value for reduction")

    # In order to support previous versions, accept boolean size_average and reduce
    # and convert them into the new constants for now

    # We use these functions in torch/legacy as well, in which case we'll silence the warning
    @staticmethod
    def legacy_get_string(size_average, reduce, emit_warning=True):
        warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

        if size_average is None:
            size_average = True
        if reduce is None:
            reduce = True

        if size_average and reduce:
            ret = 'elementwise_mean'
        elif reduce:
            ret = 'sum'
        else:
            ret = 'none'
        if emit_warning:
            import warnings
            warnings.warn(warning.format(ret))
        return ret

    @staticmethod
    def legacy_get_enum(size_average, reduce, emit_warning=True):
        return _Reduction.get_enum(_Reduction.legacy_get_string(size_average, reduce, emit_warning))

class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class _WeightedLoss(_Loss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean'):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)

class CrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, size_average=None, ignore_index=-100,
                 reduce=None, reduction='elementwise_mean'):
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)





