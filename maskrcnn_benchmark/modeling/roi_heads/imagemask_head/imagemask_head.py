import torch
import torch.nn.functional as F

from .roi_imagemask_predictors import make_roi_imagemask_predictor
from .loss import make_roi_imagemask_loss_evaluator

class ImageMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ImageMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.predictor = make_roi_imagemask_predictor(cfg)
        self.loss_evaluator = make_roi_imagemask_loss_evaluator(cfg)

    def forward(self, features, img_sizes, targets=None):
        """
        Arguments:
            features (tuple[Tensor]): feature-maps from possibly several levels
            features (list(Int, Int)): [(W, H)], but len(lst) == 1 # TODO
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # features - sth like (tensor[1,256,_w,_h],,)
        assert(features.size()[0] == 1) # only single batch supported
        # assert(targets.size()[0] == 1) # only single batch supported


        x = self.predictor(features, img_sizes)

        if self.training:


        if self.training:
            assert targets is not None
            assert len(targets) == 1 # single batch




def build_roi_imagemask_head(cfg, in_channels):
    return ImageMaskHead(cfg, in_channels)