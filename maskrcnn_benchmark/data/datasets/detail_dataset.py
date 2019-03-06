import torch.utils.data.Dataset
from .voc import PascalVOCDataset

class DetailDataset(torch.utils.data.Dataset):
    CLASSES = PascalVOCDataset.CLASSES

    def __init__(self):
        from .coco import COCODataset # TODO podobne do tego
        pass