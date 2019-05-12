#!/usr/bin/env python3

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

OUTPUT = 'detail.json'
OUTPUT_TRAIN = 'detail_train.json'
OUTPUT_TEST = 'detail_test.json'
OUTPUT_VAL = 'detail_val.json'

TRAIN = 'train'
TEST = 'test'
VAL = 'val'

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
        print('Keypoints:', len(kpts))
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
        print('Segmentation:', len(segm))
        return segm

    def mergeeeeeee(self):
       a = self.convert_kpts()
       a.extend(self.convert_instances())
       return a

    def save_img_names(self, img_list, file_name):
        with open(join(OUTPUT_DIR, file_name), 'w') as f:
            for img in img_list:
                f.write(img + '.jpg\n')
        

    def to_dict(self, cats):
        train_data = dict()
        test_data = dict()
        val_data = dict()
        
        # Setting info
        train_data['info'] = self.d['info']
        test_data['info'] = self.d['info']
        val_data['info'] = self.d['info']
        
        # Merging annotations
        annotations = self.mergeeeeeee()

        # Preparing images to split
        # I remove images without annotations
        annos_img_set = set()
        with_annos = 0
        wo_annos = 0
        img_list = []  # Contains list of images with annotations
        for anno in annotations:
            annos_img_set.add(anno['image_id'])
        for x in self.d['images']:
            if not (x['id'] in annos_img_set):
                wo_annos += 1
                print(x)
            else:
                with_annos += 1
                img_list.append(x)
        print('With annos:', with_annos, '\nwithout annos:', wo_annos)

        # Spliting images accordingly to 'phase'
        train_img, test_img, val_img = self.split(img_list)
        train_data['images'] = train_img
        test_data['images'] = test_img
        val_data['images'] = val_img

        # Preparing list of ids of all the images in each split
        train_id = set()
        test_id = set()
        val_id = set()
        for img in train_img:
            train_id.add(img['id'])
        for img in test_img:
            test_id.add(img['id'])
        for img in val_img:
            val_id.add(img['id'])

        # Preparing list of annotations for each split
        train_annos = []
        test_annos = []
        val_annos = []
        for anno in annotations:
            if anno['image_id'] in train_id:
                train_annos.append(anno)
            elif anno['image_id'] in test_id:
                test_annos.append(anno)
            elif anno['image_id'] in val_id:
                val_annos.append(anno)
            else:
                print("Image not found:", anno['image_id'])
                print(anno)
                assert False

        # Making sure there is enough annotations for all the images
        assert len(train_annos) >= len(train_img)
        assert len(test_annos) >= len(test_img)
        assert len(val_annos) >= len(val_img)
        
        # Saving list of files in order to split images between folders
        self.save_img_names(train_img, 'train_list')
        self.save_img_names(test_img, 'test_list')
        self.save_img_names(val_img, 'val_list')

        # Setting annotations
        train_data['annotations'] = train_annos
        test_data['annotations'] = test_annos
        val_data['annotations'] = val_annos

        # Setting categories
        train_data['categories'] = cats
        test_data['categories'] = cats
        val_data['categories'] = cats

        return train_data, test_data, val_data

    def split(self, img_list):
        """Splits img_list accordingly to phase.

        Arguments:
        img_list -- list of images (in JSON)
        """
        train = []
        test = []
        val = []
        for img in img_list:
            if img['phase'] == TRAIN:
                train.append(img)
            elif img['phase'] == TEST:
                test.append(img)
            elif img['phase'] == VAL:
                val.append(img)
            else:
                print("Unknown phase", img['id'], img['phase'])
                assert False
        print('Train:', len(train))
        print('Test', len(test))
        print('Val', len(val))
        return train, test, val


    def dump(self):
        train, test, val = self.to_dict(self.INST_CATS)
        print('Saving..')
        with open(join(OUTPUT_DIR, OUTPUT_TRAIN), 'w') as outfile:
            json.dump(train, outfile)
        with open(join(OUTPUT_DIR, OUTPUT_TEST), 'w') as outfile:
            json.dump(test, outfile)
        with open(join(OUTPUT_DIR, OUTPUT_VAL), 'w') as outfile:
            json.dump(val, outfile)


if __name__ == '__main__':
    dc = DetailToCoco()
    dc.dump()
