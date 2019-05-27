import torch
import torch.nn.functional as F

from .roi_imagemask_predictors import make_roi_imagemask_predictor
from .loss import make_roi_imagemask_loss_evaluator

class ImageMaskHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ImageMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.predictor = make_roi_imagemask_predictor(cfg) # (1 x K x 128 x 128)
        self.loss_evaluator = make_roi_imagemask_loss_evaluator(cfg)

    def _to_proposals(self, x):
        # x - (N, K, 128, 128), where K is num classes + 1 (for detail - 60)
        # return - proposal (N, 128, 128), where proposal[i, j] = k iff pixel`s (i, j) class is k
        proposal = torch.max(x, dim=1)[1] # torch max returns (value, indices)
        return proposal

    def forward(self, features, img_sizes, targets=None):

        x = self.predictor(features, img_sizes)

        proposals = self._to_proposals(x)

        if self.training:
            assert targets is not None
            loss_imagemask = self.loss_evaluator(x, targets)
            return x, proposals, dict(loss_imagemask=loss_imagemask)

        return x, proposals, {}

def build_roi_imagemask_head(cfg, in_channels):
    assert in_channels == cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    assert in_channels == 256
    return ImageMaskHead(cfg)