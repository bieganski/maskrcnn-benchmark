import os
import torch.utils.data

from .voc import PascalVOCDataset

from detail import Detail



details = None

def detail_test(details):
    annFile = '/home/mateusz/zpp/maskrcnn-benchmark/pascal/detail-api/trainval_withkeypoints.json'
    imgDir='/home/mateusz/zpp/maskrcnn-benchmark/pascal/detail-api/VOCdevkit/MINIMAL/JPEGImages'
    phase = 'trainval'

    details = Detail(annFile, imgDir, phase)
    cats = details.getCats()
    print(cats)


# similar to VOC Dataset, but containing more anns
# for additional tasks
class DetailDataset(torch.utils.data.Dataset):

    CLASSES = PascalVOCDataset.CLASSES

    def __init__(self, data_dir, anno, split, use_difficult=False, transforms=None):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms

        # TODO - now it must be absolute path, join it
        self.anno = anno
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        self.ids = [x.strip("\n") for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = DetailDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
