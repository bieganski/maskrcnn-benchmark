from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.poolers import Pooler

from maskrcnn_benchmark.layers import Conv2d


@registry.ROI_IMAGEMASK_FEATURE_EXTRACTORS.register("ImageMaskRCNNFeatureExtractor")
class ImageMaskRCNNFeatureExtractor(nn.Module):
    def __init__(self):
        pass

    # Semantyczna segmentacja nie używa proposali z sieci RPN -
    # bierze bezpośrednio output z FPN, upsampluje do rozmiaru obrazka
    # i liczy loss.
    def forward(self, x, features):
        pass


def make_roi_keypoint_feature_extractor(cfg, in_channels):
    func = registry.ROI_IMAGEMASK_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)