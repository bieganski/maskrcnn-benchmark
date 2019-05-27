from torch.nn import functional as F
import torch


class SegmentationMaskLoss(object):
    def __init__(self, num_cls):
        super(SegmentationMaskLoss, self).__init__()
        self.num_cls = num_cls
        # self.soft = F.softmax
        # self.loss = F.binary_cross_entropy
        self.loss2 = F.binary_cross_entropy_with_logits

    def __call__(self, x, gt):
        assert gt.shape == x.shape

        # TODO tu moze zle dzialac przez mismatche obrazkow, zrobic wtedy dla kazdej warstwy pojedynczo
        # x = self.soft(x, dim=1)
        # loss = self.loss(x, gt)
        # print("<<<<<<<<<<<<LOSS:")
        # print(loss)
        loss = self.loss2(x, gt)
        return loss


def make_roi_imagemask_loss_evaluator(cfg):
    loss_evaluator = SegmentationMaskLoss(cfg.MODEL.ROI_IMAGEMASK_HEAD.NUM_CLASSES)
    return loss_evaluator