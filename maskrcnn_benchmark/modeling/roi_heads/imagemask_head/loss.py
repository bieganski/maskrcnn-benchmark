from torch.nn import functional as F
import torch


class SegmentationMaskLoss(object):
    def __init__(self, num_cls):
        super(SegmentationMaskLoss, self).__init__()
        self.num_cls = num_cls
        self.soft = F.softmax
        self.loss = F.binary_cross_entropy

    def __call__(self, x, gt):
        # gt - lista targetów
        # assert False, (x.size(), gt[0].size())
        # AssertionError: (torch.Size([1, 60, 426, 500]), torch.Size([426, 500]))
        assert x.size()[0] == 1
        x = x.squeeze(0)
        assert x.size()[0] == self.num_cls
        gt = gt[0] # TODO wieksze batche
        assert gt.size() == x.size()[-2:]
        dim_ok_gt = torch.zeros_like(x, requires_grad=True)
        print(type(gt))
        print(gt)
        # thats lame, but should works
        for i in range(gt.size(0) + 1):
            for j in range(gt.size(1) + 1):
                cls = gt[i, j]
                dim_ok_gt[cls, i, j] = 1

        assert dim_ok_gt.sum() == torch.nonzero(gt).sum()

        x = self.soft(x, dim=0)
        loss = self.loss(x, dim_ok_gt)
        print("<<<<<<<<<<<<LOSS:")
        print(loss)
        return loss


def make_roi_imagemask_loss_evaluator(cfg):
    loss_evaluator = SegmentationMaskLoss(cfg.MODEL.ROI_IMAGEMASK_HEAD.NUM_CLASSES)
    return loss_evaluator