# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        # print(images.image_sizes) # height, width [torch.Size([133, 100]), torch.Size([152, 100])]
        # print(">>>>>>>>>>>>>")
        # print(targets)
        # print("<<<<<<<<<<<<<")
        # type(features) = tuple
        # print(features)
        # C4 extractor:
        #   torch.Size([2, 256, 32, 40])
        #   torch.Size([2, 256, 16, 20])
        #   torch.Size([2, 256, 8, 10])
        #   torch.Size([2, 256, 4, 5])
        #   torch.Size([2, 256, 2, 3])

        # should-be Conv 1x1 extractor:
        #   torch.Size([2, 256, 40, 32])
        #   torch.Size([2, 256, 20, 16])
        #   torch.Size([2, 256, 10, 8])
        #   torch.Size([2, 256, 5, 4])
        #   torch.Size([2, 256, 3, 2])

        # for el in features:
        #     print(el.size())
        # print("------------------")
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
