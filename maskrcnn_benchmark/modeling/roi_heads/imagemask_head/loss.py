from torch import nn
from torch.nn import functional as F
import torch


class SegmentationMaskLoss(object):
    def __init__(self, num_cls):
        super(SegmentationMaskLoss, self).__init__()
        self.num_cls = num_cls

    def __call__(self, x, gt):
        assert x.size()[0] == 1
        x = x.squeeze(0)
        assert x.size()[0] == self.num_cls
        gt = torch.tensor(gt) # in case of being np.ndarray
        assert gt.size() == x.size()[-2:]
        real_gt = torch.zeros_like(x)

        # thats lame, but should works
        for i in range(gt.size(0) + 1):
            for j in range(gt.size(1) + 1):


def make_roi_imagemask_loss_evaluator(cfg):
    loss_evaluator = SegmentationMaskLoss(cfg.ROI_IMAGEMASK_HEAD.NUM_CLASSES)
    return loss_evaluator