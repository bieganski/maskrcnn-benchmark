from torch import nn
from torch.nn import functional as F
import torch


class SegmentationMaskLoss(object):
    def __init__(self, num_cls):
        super(SegmentationMaskLoss, self).__init__()
        self.num_cls = num_cls
        self.soft = F.softmax
        self.loss_fn = F.binary_cross_entropy

    def __call__(self, x, gt):
        assert x.size()[0] == 1
        x = x.squeeze(0)
        assert x.size()[0] == self.num_cls
        gt = torch.tensor(gt) # in case of being np.ndarray
        assert gt.size() == x.size()[-2:]
        real_gt = torch.zeros_like(x, requires_grad=True)

        # thats lame, but should works
        for i in range(gt.size(0) + 1):
            for j in range(gt.size(1) + 1):
                cls = gt[i, j]
                if cls != 0:
                    real_gt[i, j, cls] = 1

        assert real_gt.sum() == torch.nonzero(gt).sum()

        x = self.soft(x, dim=0)
        loss = self.loss_fn(x, real_gt)
        return loss


def make_roi_imagemask_loss_evaluator(cfg):
    loss_evaluator = SegmentationMaskLoss(cfg.ROI_IMAGEMASK_HEAD.NUM_CLASSES)
    return loss_evaluator