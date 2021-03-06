# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from .classification_feature_extractors import make_classification_feature_extractor
from .classification_predictors import make_classification_predictor
from .inference import make_classification_post_processor
from .loss import make_classification_loss_evaluator


class ClassificationHead(torch.nn.Module):
    """
    Generic Classification Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ClassificationHead, self).__init__()
        self.feature_extractor = make_classification_feature_extractor(cfg, in_channels)
        self.predictor = make_classification_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_classification_post_processor(cfg)
        self.loss_evaluator = make_classification_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        # class_logits, box_regression = self.predictor(x)
        class_logits = self.predictor(x)

        if not self.training:
            # result = self.post_processor((class_logits, box_regression), proposals)
            result = self.post_processor((class_logits), proposals)
            return x, result, {}

        # loss_classifier, loss_box_reg = self.loss_evaluator(
        #     [class_logits], [box_regression]
        # )
        loss_classifier = self.loss_evaluator(
            [class_logits]
        )
        # return (
        #     x,
        #     proposals,
        #     dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg),
        # )
        return (
            x,
            proposals,
            dict(loss_classifier=loss_classifier),
        )


def build_classification_head(cfg, in_channels):
    """
    Constructs a new classification head.
    By default, uses ClassificationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ClassificationHead(cfg, in_channels)
