import torch
import torch.nn.functional as F

from .roi_imagemask_predictors import make_roi_imagemask_predictor
from .loss import make_roi_imagemask_loss_evaluator

class ImageMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ImageMaskHead, self).__init__()
        self.cfg = cfg.clone()
        self.predictor = make_roi_imagemask_predictor(cfg, in_channels, cfg.MODEL.ROI_IMAGEMASK_HEAD.NUM_CLASSES)
        self.loss_evaluator = make_roi_imagemask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (tuple[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        assert(targets.size()[0] == 1) # only single batch supported

        # if self.training:
        #     with torch.no_grad():
        #         proposals = self.loss_evaluator.subsample(proposals, targets)
        #
        # x = self.feature_extractor(features, proposals)
        # kp_logits = self.predictor(x)
        #
        # if not self.training:
        #     result = self.post_processor(kp_logits, proposals)
        #     return x, result, {}
        #
        # loss_kp = self.loss_evaluator(proposals, kp_logits)
        #
        # return x, proposals, dict(loss_kp=loss_kp)


def build_roi_imagemask_head(cfg, in_channels):
    return ImageMaskHead(cfg, in_channels)