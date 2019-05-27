# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import numpy as np

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .imagemask_head.imagemask_head import build_roi_imagemask_head


def getMulticlassMask(boxlist):
    masks = boxlist.get_field('semantic_masks')
    shape = masks[0].size()
    multicategorical_mask = np.zeros((shape[0], shape[1]))

    for mask, cat in zip(masks, boxlist.get_field('labels')):
        mask = (mask.cpu()).numpy()
        cols = np.argmax(mask[0] == -1) # first_padded_column_id
        if cols == 0 and mask[0][0] != -1: cols = mask.shape[1]
        mask = mask[mask != -1]
        if len(mask) > 0:
            rows = int(len(mask) / cols)
            multicategorical_mask[np.nonzero(mask.reshape((rows, cols)))] = cat.cpu()

    return torch.Tensor(multicategorical_mask, device='cpu')

segm_cats_number = 60 # including background
# CWK stands for Class (including background) x Weight x Height
def getCWHMulticlassMask(boxlist):
    masks = boxlist.get_field('semantic_masks')
    shape = masks[0].size()
    multicategorical_mask = np.zeros((segm_cats_number, shape[0], shape[1]), dtype=int)
    multicategorical_mask[0] = np.ones((shape[0], shape[1]), dtype=int) # background_mask init

    for mask, cat in zip(masks, boxlist.get_field('labels')):
        mask = (mask.cpu()).numpy()
        cols = np.argmax(mask[0] == -1) # first_padded_column_id
        if cols == 0 and mask[0][0] != -1: cols = mask.shape[1]
        mask = mask[mask != -1]
        if len(mask) > 0:
            rows = int(len(mask) / cols)
            cat = cat.cpu()
            nonzero_elems = np.nonzero(mask.reshape((rows, cols)))
            multicategorical_mask[cat][nonzero_elems] = 1
            multicategorical_mask[0][nonzero_elems] = 0

    return multicategorical_mask

class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def _filterSegmentation(self, boxlist):
        ids = []
        sgms = boxlist.get_field('masks')
        for index, poly in enumerate(sgms.polygons):
            valid = True
            for inner_poly in poly.polygons:
                if len(inner_poly) <= 4:
                    valid = False        
            if valid and len(poly.polygons) > 0:
                ids.append(index)
        if len(ids) == 0:
            print("!!!!!")
        boxes = [(boxlist.bbox[index]).tolist() for index in ids]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # print(type(boxes))
        boxlist.bbox = boxes.cuda()

        filtered_category_ids = boxlist.get_field('labels')
        filtered_category_ids = torch.tensor([filtered_category_ids[index] for index in ids])
        boxlist.add_field("labels", filtered_category_ids.cuda())

        filtered_sgms = boxlist.get_field('masks')
        filtered_sgms.polygons = [filtered_sgms.polygons[index] for index in ids]
        boxlist.add_field('masks', filtered_sgms)

        filtered_kpts = boxlist.get_field('keypoints')
        keypoints = [(filtered_kpts.keypoints[index]).tolist() for index in ids]
        # print(keypoints.device)
        # print(type(keypoints.device))
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        filtered_kpts.keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = filtered_kpts.keypoints.shape[0]
        # print(filtered_kpts.keypoints.shape)
        if num_keypoints:
            filtered_kpts.keypoints = (filtered_kpts.keypoints.view(num_keypoints, -1, 3)).cuda()
        boxlist.add_field('keypoints', filtered_kpts)

        assert len(boxes) == len(filtered_category_ids) == len(filtered_sgms.polygons) == len(filtered_kpts.keypoints)
        return boxlist

    def _filterKeypoints(self, boxlist):
        ids = []
        kpts = boxlist.get_field('keypoints')
        for index, kpt in enumerate(kpts.keypoints):
            if torch.sum(kpt) > 0:
                ids.append(index)

        boxes = [(boxlist.bbox[index]).tolist() for index in ids]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        # print(type(boxes))
        boxlist.bbox = boxes.cuda()

        filtered_category_ids = boxlist.get_field('labels')
        filtered_category_ids = torch.tensor([filtered_category_ids[index] for index in ids])
        boxlist.add_field("labels", filtered_category_ids.cuda())

        filtered_sgms = boxlist.get_field('masks')
        filtered_sgms.polygons = [filtered_sgms.polygons[index] for index in ids]
        boxlist.add_field('masks', filtered_sgms)

        filtered_kpts = boxlist.get_field('keypoints')
        keypoints = [(filtered_kpts.keypoints[index]).tolist() for index in ids]
        # print(keypoints.device)
        # print(type(keypoints.device))
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        filtered_kpts.keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = filtered_kpts.keypoints.shape[0]
        if num_keypoints:
            filtered_kpts.keypoints = (filtered_kpts.keypoints.view(num_keypoints, -1, 3)).cuda()
        boxlist.add_field('keypoints', filtered_kpts)

        assert len(boxes) == len(filtered_category_ids) == len(filtered_sgms.polygons) == len(filtered_kpts.keypoints)
        return boxlist

    def forward(self, features, proposals, targets=None):
        losses = {}
        test = not bool(targets)
        # box_targets = targets
        # maybe do that in preprocessing and add a flag to determine id choices
        # probably not the most efficient way to do that

        # uncomment to see multiclass masks
        # for x in targets:
        #     print(getCWHMulticlassMask(x))

        if not test:
            nonsemantic_targets = []
            for it, boxlist in enumerate(targets):
                ids = []
                for _index, cat in enumerate(boxlist.get_field('labels')):
                    if cat <= 20:
                        ids.append(_index)

                boxes = [(boxlist.bbox[index]).tolist() for index in ids]
                boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
                boxlist.bbox = boxes.cuda()

                filtered_category_ids = boxlist.get_field('labels')
                filtered_category_ids = torch.tensor([filtered_category_ids[index] for index in ids])
                boxlist.add_field("labels", filtered_category_ids.cuda())

                filtered_sgms = boxlist.get_field('masks')
                filtered_sgms.polygons = [filtered_sgms.polygons[index] for index in ids]
                boxlist.add_field('masks', filtered_sgms)

                filtered_kpts = boxlist.get_field('keypoints')
                keypoints = [(filtered_kpts.keypoints[index]).tolist() for index in ids]

                device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
                filtered_kpts.keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
                num_keypoints = filtered_kpts.keypoints.shape[0]

                if num_keypoints:
                    filtered_kpts.keypoints = (filtered_kpts.keypoints.view(num_keypoints, -1, 3)).cuda()
                boxlist.add_field('keypoints', filtered_kpts)

                assert len(boxes) == len(filtered_category_ids) == len(filtered_sgms.polygons) == len(
                    filtered_kpts.keypoints)
                if len(ids) > 0:
                    nonsemantic_targets.append((it, boxlist))

            if (len(nonsemantic_targets) == 0):
                return losses

            ids = [x[0] for x in nonsemantic_targets]
            targets = [x[1] for x in nonsemantic_targets]
            features = [features[index] for index in ids]
            proposals = [proposals[index] for index in ids]

        # uncomment to see multiclass masks
        # for x in targets:
        #     print(getMulticlassMask(x).shape)  # ndarray
        #
        # semantic_features = features
        # semantic_proposals = proposals

        imagemasks = [getCWHMulticlassMask(x) for x in targets]
        shapes = [x.shape for x in imagemasks]

        def including_rectangle(shapes):
            w, h = 0, 0
            for shape in shapes:
                w = max(w, shape[0])
                h = max(h, shape[1])
            return (w, h)

        new_shape = including_rectangle(shapes)

        def _pad(shape, array, padval = -1):
            padded = np.full(tuple(shape), padval)
            padded[:array.shape[-2], :array.shape[-1]] = array
            return padded

        resized_imagemasks = [_pad(new_shape, x, 0) for x in imagemasks]

        semantic_targets = [torch.FloatTensor(x).unsqueeze(0).cuda() for x in resized_imagemasks]
        semantic_targets = torch.cat(tuple(semantic_targets))
        # torch.set_printoptions(profile="full")
        assert False, semantic_targets.shape
        # exit(1)
        # semantic segmentation does not need boxes
        if self.cfg.MODEL.IMAGEMASK_ON:
            err = ("BATCH SIZE OF {} ERROR: Semantic Segmentation works on single batch, "
                   + "due to resizing FPN output").format(features[0].size()[0])
            assert features[0].size()[0] == 1, err
            y, proposals_imagemask, loss_imagemask \
                = self.imagemask(features, [tuple(x.shape[-2:]) for x in semantic_targets], semantic_targets)
            if self.training:
                losses.update(loss_imagemask)
            print(losses)
            return y, None, losses

        ################### TODO
        # return None, None, losses
        ###################


        # targets = box_targets
        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        # detections = proposals
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x

            ids = []
            for index, boxlist in enumerate(targets):
                sgms = boxlist.get_field('masks')
                valid = False
                for poly in sgms.polygons:
                    for inner_poly in poly.polygons:
                        if len(inner_poly) > 4:
                            valid = True        
                if valid:
                    ids.append(index)

            filtered_mask_features = [mask_features[index] for index in ids]
            filtered_mask_detections = [detections[index] for index in ids]
            filtered_mask_targets = [targets[index] for index in ids]

            # box_targets = 
            # keypoint_targets = targets
            filtered_mask_targets = [self._filterSegmentation(target) for target in filtered_mask_targets]

            if len(ids) > 0:
                # targets = filtered_mask_targets
                # During training, self.box() will return the unaltered proposals as "detections"
                # this makes the API consistent during training and testing
                x, detections, loss_mask = self.mask(filtered_mask_features, filtered_mask_detections, filtered_mask_targets)
                losses.update(loss_mask)

        if self.cfg.MODEL.KEYPOINT_ON:
            keypoint_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads, then we can reuse the features already computed
            if (
                self.training
                and self.cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                keypoint_features = x

            # targets = keypoint_targets
            # maybe do that in preprocessing and add a flag to determine id choices
            # probably not the most efficient way to do that
            ids = []
            for index, boxlist in enumerate(targets):
                kpts = boxlist.get_field('keypoints')
                counter = 0
                for matrix in kpts.keypoints:
                    # print(matrix)
                    # print(matrix[:,2])
                    # print(sum(matrix[:,2]))
                    counter += sum(matrix[:,2])
                if counter >= 5: # 5 not 10, because hardly any pictures have that many keypoints
                    ids.append(index)

            # those may better be something more sophiticated than lists
            filtered_keypoint_features = [keypoint_features[index] for index in ids]
            filtered_keypoint_detections = [detections[index] for index in ids]
            filtered_keypoint_targets = [targets[index] for index in ids]

            filtered_keypoint_targets = [self._filterKeypoints(target) for target in filtered_keypoint_targets]

            # print(ids)
            # print(len(filtered_keypoint_features))
            # print(len(filtered_detections))
            # print(len(filtered_targets))

            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            all_detections = detections # get detections for all pictures to preserve behavior specified in aforementioned comment

            if len(ids) > 0:
                # x, detections, loss_keypoint = self.keypoint(keypoint_features, detections, targets)
                x, detections, loss_keypoint = self.keypoint(filtered_keypoint_features, filtered_keypoint_detections, filtered_keypoint_targets)
                losses.update(loss_keypoint)

            detections = all_detections # restore detections, maybe there is a problem outside of training with that
        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    # if cfg.MODEL.CLASSIFICATION_ON:
    #     roi_heads.append(("classification", build_classification_head(cfg, in_channels)))
    if cfg.MODEL.IMAGEMASK_ON:
        roi_heads.append(("imagemask", build_roi_imagemask_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
