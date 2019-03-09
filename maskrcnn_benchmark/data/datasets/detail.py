import os
import torch.utils.data
import torch
from PIL import Image

from detail import Detail
from .voc import PascalVOCDataset


from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import Keypoints # TODO PersonKeypoints?


# example DetailDataset initialisation:
# imgDir='/home/mateusz/zpp/maskrcnn-benchmark/pascal/VOCdevkit/MINIMAL/JPEGImages'
# annFile = '/home/mateusz/zpp/maskrcnn-benchmark/pascal/trainval_withkeypoints.json'
# split = 'train' / 'trainval' / 'test'


# similar to VOC Dataset, but containing more annotations
# for additional tasks, like boundary or occlusion recognition.
# target BoxList contains additional fields:
# 'mask', 'kpts', 'bounds', 'occl'.
# mask and kpts in standard maskrcnn-benchmark.structure format, and
# boundings and occlusions in Pascal in Detail format (TODO check it out)
# TODO implement minimal
# upper tasks are rather easy, because "minimal" is for instance just taking 1/10 images
class DetailDataset(torch.utils.data.Dataset):

    CLASSES = PascalVOCDataset.CLASSES # TODO to chyba nie wszystkie, Detail.getCats() zwraca wiecej

    def __init__(self, img_dir, ann_file, split, minimal=False, transforms=None):
        self.img_dir = img_dir
        self.image_set = split
        self.transforms = transforms
        self.anno = ann_file

        self.detail = Detail(ann_file, img_dir, split, minimal, divider=10)

        imgs = self.detail.getImgs()
        idxs = range(len(imgs))
        self.idx_to_img = dict(zip(idxs, imgs))

        # TODO może się przydać, zrobic to poprawnie, uważając na underscore
        # self.img_to_idx = dict(zip([x.image_id for x in imgs], idxs))

        cls = DetailDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __len__(self):
        return len(self.idx_to_img)

    def _img_size(self, img):
        return (img['width'], img['height'])

    def get_groundtruth(self, idx):
        img = self.idx_to_img[idx]
        boxes = self.detail.getBboxes(img)
        # example of 'boxes':
        # [{'bbox': [250, 209, 241, 149], 'category': 'motorbike'},
        # {'bbox': [312, 139, 109, 191], 'category': 'person'}]
        boxes = [box['bbox'] for box in boxes]  # TODO gubimy informację o otoczonym przedmiocie
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, self._img_size(img), mode="xywh").convert("xyxy")
        target = target.clip_to_image(remove_empty=True)

        img_keypoints = self.detail.getKpts(img)
        keypoints = [skelton['keypoints'] for skelton in img_keypoints]

        # TODO keypoints - gubimy informację o bbox
        target.add_field("kpts", Keypoints(keypoints, self._img_size(img)))
        target.add_field("mask", SegmentationMask(self.detail.getMask(img).tolist(), size=self._img_size(img)))
        target.add_field("bounds", self.detail.getBounds(img))
        target.add_field("occl", self.detail.getOccl(img))
        # TODO human parts?

        return target


    def __getitem__(self, idx):
        img = self.idx_to_img[idx]
        # example img object:
        # {'file_name': '2008_000002.jpg', 'phase': 'val', 'height': 375, 'width': 500,
        #  'date_captured': '31-May-2015 17:44:04', 'image_id': 2008000002, 'annotations': [1, 62295],
        #  'categories': [454, 427], 'parts': [16], 'keypoints': []}
        img = Image.open(os.path.join(self.img_dir, img['file_name'])).convert('RGB')
        target = self.get_groundtruth(idx)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target, idx


    def get_img_info(self, idx):
        img = self.idx_to_img[idx]
        return {"height": img['height'], "width": img['width']}