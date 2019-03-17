#!/usr/bin/python3

import json
from os.path import join

TRAINVAL_PATH='./pascal'
DETAIL_ANNS = './trainval_withkeypoints.json'
OUTPUT_DIR='./tococo'

KPT = 'kpt.json'
INST = 'inst.json'
KPT_TEST = 'lol.json'


class DetailToCoco:
    # TODO ukradzione wprost z COCOAPI, liczę, że pokrywa się z detail api
    KPT_CATS = [{'supercategory': 'person',
                 'id': 1,
                 'name': 'person',
                 'keypoints': ['nose',
                               'left_eye',
                               'right_eye',
                               'left_ear',
                               'right_ear',
                               'left_shoulder',
                               'right_shoulder',
                               'left_elbow',
                               'right_elbow',
                               'left_wrist',
                               'right_wrist',
                               'left_hip',
                               'right_hip',
                               'left_knee',
                               'right_knee',
                               'left_ankle',
                               'right_ankle'],
                 'skeleton': [[16, 14],
                              [14, 12],
                              [17, 15],
                              [15, 13],
                              [12, 13],
                              [6, 12],
                              [7, 13],
                              [6, 7],
                              [6, 8],
                              [7, 9],
                              [8, 10],
                              [9, 11],
                              [2, 3],
                              [1, 2],
                              [1, 3],
                              [2, 4],
                              [3, 5],
                              [4, 6],
                              [5, 7]]}]

    def __init__(self):
        self.KPT = 'kpt.json'
        self.INST = 'inst.json'
        self.KPT_TEST = 'lol.json'
        self.TRAINVAL_PATH = './pascal'
        self.DETAIL_ANNS = './trainval_withkeypoints.json'
        self.OUTPUT_DIR = './tococo'
        self.d = json.load(open(join(TRAINVAL_PATH, DETAIL_ANNS), 'r'))
        self.INST_CATS = self.d['categories']
        for cat in self.INST_CATS:
            cat['id'] = cat['category_id']
            del cat['category_id']
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
            id += 1
        return kpts

    def convert_instances(self):
        segm = self.d['annos_segmentation']  # contains semgmentation and bboxes
        for el in segm:
            el['iscrowd'] = 1  # sometimes it is 0 wtf
        return segm

    def to_dict(self, annFun, cats):
        res = dict()
        res['info'] = self.d['info']
        res['images'] = self.d['images']
        res['annotations'] = annFun()
        res['categories'] = cats
        return res

    def dumpKpt(self):
        kpts = self.to_dict(self.convert_kpts, self.KPT_CATS)
        with open(join(OUTPUT_DIR, KPT), 'w') as outfile:
            json.dump(kpts, outfile)

    def dumpInst(self):
        inst = self.to_dict(self.convert_instances, self.INST_CATS)
        with open(join(OUTPUT_DIR, INST), 'w') as outfile:
            json.dump(inst, outfile)

    def dumpTest(self):
        kpts = self.to_dict(self.convert_kpts, self.INST_CATS)
        with open(join(OUTPUT_DIR, KPT_TEST), 'w') as outfile:
            json.dump(kpts, outfile)


if __name__ == '__main__':
    dc = DetailToCoco()
    dc.dumpInst()
    dc.dumpKpt()
    dc.dumpTest()