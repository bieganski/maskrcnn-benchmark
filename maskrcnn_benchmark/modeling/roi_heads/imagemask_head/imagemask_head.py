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
        # x - (N, 128, 128, K), where K is num classes + 1 (for detail - 60)
        # return - proposal (N, 128, 128, 1), where proposal[i, j] = k iff pixel`s (i, j) class is k
        assert x.size()[-1] == self.cfg.MODEL.ROI_IMAGEMASK_HEAD.NUM_CLASSES, x.size()[-1]
        proposal = torch.max(x, dim=1)[1]
        assert list(proposal.size()) == [1, x.size()[0], x.size()[1], 1]

    def forward(self, features, img_sizes, targets=None):
        """
        Arguments:
            features (tuple[Tensor]): feature-maps from possibly several levels
            img_sizes (list(Int, Int)): [(W, H)], but temporarily len(lst) == 1 # TODO
            targets (list[torch.Tensor], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposal (Tensor): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # features - sth like (tensor[1,256,_w,_h],,)
        assert(features[0].size()[0] == 1) # only single batch supported
        # assert(targets.size()[0] == 1) # only single batch supported


        x = self.predictor(features, img_sizes)

        proposals = self._to_proposals(x)

        if self.training:
            assert targets is not None
            assert len(targets) == 1 # single batch
            loss_imagemask = self.loss_evaluator(x, targets)
            return x, proposals, dict(loss_imagemask=loss_imagemask)

        return x, proposals, {}

def build_roi_imagemask_head(cfg, in_channels):
    assert in_channels == cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    assert in_channels == 256
    return ImageMaskHead(cfg)