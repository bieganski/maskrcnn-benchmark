from torch.nn import functional as F

import torch

class SegmentationMaskLoss(object):
    def __init__(self, num_cls):
        super(SegmentationMaskLoss, self).__init__()
        self.num_cls = num_cls
        self.soft = F.softmax
        self.loss = F.binary_cross_entropy
        self.interp = F.interpolate
        # self.loss2 = F.binary_cross_entropy_with_logits

    def __call__(self, x, gt, img_sizes):
        def including_rectangle(shapes):
            w, h = 0, 0
            for shape in shapes:
                w = max(w, shape[-2])
                h = max(h, shape[-1])
            return w, h

        w, h = gt.shape[-2], gt.shape[-1] # including_rectangle(img_sizes)
        new_shape = tuple([x.shape[0], x.shape[1], w, h]) # batch size, 60, w, h

        res = torch.zeros(new_shape, device='cuda')
        for i, single_feature_map in enumerate(x):
            w, h = (img_sizes[i][-2], img_sizes[i][-1])
            res[i, :, :w, :h] = self.interp(single_feature_map.unsqueeze(0),
                                            size=(w, h),
                                            mode='nearest')


        assert gt.shape == x.shape

        # TODO tu moze zle dzialac przez mismatche obrazkow, zrobic wtedy dla kazdej warstwy pojedynczo
        x = self.soft(x, dim=1)
        loss = self.loss(x, gt)
        print("<<<<<<<<<<<<LOSS:")
        print(loss)
        # loss = self.loss2(x, gt)
        return loss


def make_roi_imagemask_loss_evaluator(cfg):
    loss_evaluator = SegmentationMaskLoss(cfg.MODEL.ROI_IMAGEMASK_HEAD.NUM_CLASSES)
    return loss_evaluator