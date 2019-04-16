#!/usr/bin/python3

import json
import cv2
import numpy as np
from pycocotools import mask
from skimage import measure
from detail import mask as maskUtils
from os.path import join

TRAINVAL_PATH='./pascal/detail-api'
DETAIL_ANNS = './trainval_withkeypoints.json'
OUTPUT_DIR='./pascal'

OUTPUT = 'kpt.json'

def compressedRLEtoPolys(segmentation):
    mask_list = maskUtils.decode(segmentation)
    ground_truth_binary_mask = mask_list
    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "bbox": ground_truth_bounding_box.tolist()
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)
        
    return annotation

class DetailToCoco:
    # real detail keypoints
    HEAD = 1
    NECK = 2
    LEFT_SHOULDER = 3
    RIGHT_SHOULDER = 9
    LEFT_ELBOW = 4
    RIGHT_ELBOW = 10
    LEFT_WRIST = 5
    RIGHT_WRIST = 11
    LEFT_HIP = 6
    RIGHT_HIP = 12
    LEFT_KNEE = 7
    RIGHT_KNEE = 13
    LEFT_ANKLE = 8
    RIGHT_ANKLE = 14

    def __init__(self):
        self.KPT = 'kpt.json'
        self.INST = 'inst.json'
        self.KPT_TEST = 'lol.json'
        self.TRAINVAL_PATH = './pascal'
        self.DETAIL_ANNS = './trainval_withkeypoints.json'
        self.OUTPUT_DIR = './tococo'
        self.d = json.load(open(join(TRAINVAL_PATH, DETAIL_ANNS), 'r'))
        self.INST_CATS = self.d['categories']

        self.REAL_KPTS = [
            [self.HEAD, self.NECK],
            [self.NECK, self.LEFT_SHOULDER],
            [self.LEFT_SHOULDER, self.LEFT_ELBOW],
            [self.LEFT_ELBOW, self.LEFT_WRIST],
            [self.NECK, self.RIGHT_SHOULDER],
            [self.RIGHT_SHOULDER, self.RIGHT_ELBOW],
            [self.RIGHT_ELBOW, self.RIGHT_WRIST],
            [self.LEFT_SHOULDER, self.LEFT_HIP],
            [self.LEFT_HIP, self.LEFT_KNEE],
            [self.LEFT_KNEE, self.LEFT_ANKLE],
            [self.RIGHT_SHOULDER, self.RIGHT_HIP],
            [self.RIGHT_HIP, self.RIGHT_KNEE],
            [self.RIGHT_KNEE, self.RIGHT_ANKLE]
        ]

        for cat in self.INST_CATS:
            cat['id'] = cat['category_id']
            del cat['category_id']
            if cat['name'] == 'person':
                cat['skeleton'] = self.REAL_KPTS
        self.change_id_format()


    def change_id_format(self):
        imgs = self.d['images']
        for img in imgs:
            self.swap('image_id', 'id', img)

    def swap(self, ex, nonex, dic):
        dic[nonex] = dic[ex]
        del dic[ex]

    def convert_kpts(self):
        kpts = self.d['annos_joints']
        id = 1
        for kpt_obj in kpts:
            kpt_obj['segmentation'] = [[]]
            kpt_obj['id'] = id
            kpt_obj['iscrowd'] = 0
            id += 1
        print(len(kpts))
        return kpts

    def convert_instances(self):
        segm = self.d['annos_segmentation']
        for el in segm:
            poly_segm_anno = compressedRLEtoPolys(el['segmentation'])
            el['segmentation'] = poly_segm_anno['segmentation']
            
            # full check is ommitted because sometimes estimated bbox size differs from the ground truth one
            # and we should probably stick to the one given; area is always ok
            # if we use only segmentation we can make
            # compressedRLEtoPolys return only that
            assert el['area'] == poly_segm_anno['area']
            # assert el['area'] == segm['area'] and el['bbox'] == segm['bbox']
            
            el['iscrowd'] = 0
            el["keypoints"] = 42*[0]
            el['num_keypoints'] = 0
        print(len(segm))
        return segm

    def mergeeeeeee(self):
       a = self.convert_kpts()
       a.extend(self.convert_instances())
       return a

    def to_dict(self, cats):
        res = dict()
        res['info'] = self.d['info']
        res['annotations'] = self.mergeeeeeee()
        img_set = set()
        for ann in res['annotations']:
          img_set.add(ann['image_id'])
        res['images'] = [x for x in self.d['images'] if x['id'] in img_set]
        res['categories'] = cats
        return res

    def dump(self):
        output = self.to_dict(self.INST_CATS)
        with open(join(OUTPUT_DIR, OUTPUT), 'w') as outfile:
            json.dump(output, outfile)


if __name__ == '__main__':
    dc = DetailToCoco()
    dc.dump()
