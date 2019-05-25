# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import numpy as np
from detail import mask as maskUtils

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints


min_keypoints_per_image = 10


def _pad(shape, array):
    padded = np.full(tuple(shape), -1)
    padded[:array.shape[0],:array.shape[1]] = array
    return padded

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)


def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)


def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # we don't care here whether keypoints are present, 'cause we will check for them in roi_heads.py
    return True

    # # keypoints task have a slight different critera for considering
    # # if an annotation is valid
    # if "keypoints" not in anno[0]:
    #     return True
    # # for keypoint detection tasks, only consider valid images those
    # # containing at least min_keypoints_per_image
    # if _count_visible_keypoints(anno) >= min_keypoints_per_image:
    #     return True
    # return False

def mask_pixel(value, pair):
    if pair[1] == 0:
        return pair[0]
    else:
        return value


class COCODataset(torchvision.datasets.coco.CocoDetection):
    def __init__(
        self, ann_file, root, remove_images_without_annotations, transforms=None
    ):
        super(COCODataset, self).__init__(root, ann_file)
        # sort indices for reproducible results
        self.ids = sorted(self.ids)

        # filter images without detection annotations
        if remove_images_without_annotations:
            ids = []
            for img_id in self.ids:
                ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=None)
                anno = self.coco.loadAnns(ann_ids)
                if has_valid_annotation(anno):
                    ids.append(img_id)
            self.ids = ids

        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.coco.getCatIds())
        }
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        masks = [obj["segmentation"] for obj in anno]
        masks = SegmentationMask(masks, img.size)
        target.add_field("masks", masks)

        # padding required because pytorch wants even sized dims
        semantic_masks = [maskUtils.decode(obj["semantic"]) if obj["semantic"] != [[]] else np.asarray([[]]) for obj in anno]
        shape = [0,0]
        for x in semantic_masks:
            if x.shape[0] > shape[0]: shape[0] = x.shape[0]
            if x.shape[1] > shape[1]: shape[1] = x.shape[1]
        semantic_masks = [_pad(shape, x) for x in semantic_masks]
        semantic_masks = torch.tensor(semantic_masks)
        target.add_field("semantic_masks", semantic_masks)

        # assert (isinstance(semantic_masks, torch.Tensor))
        assert (list(semantic_masks.size()) == list(img.size)), str(((list(semantic_masks.size()), list(img.size))))

        if anno:
            keypoints = [obj["keypoints"] for obj in anno if "keypoints" in obj]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data
