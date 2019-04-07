#!/usr/bin/python3

# TODO
# ########### ISSUES
# showAnns for segmentation all in one colour


import json
from os.path import join

TRAINVAL_PATH='./pascal/detail-api'
DETAIL_ANNS = './trainval_withkeypoints.json'
OUTPUT_DIR='./pascal'

KPT = 'kpt.json'
INST = 'inst.json'

class DetailToCoco:
    # TODO ukradzione wprost z COCOAPI, liczę, że pokrywa się z detail api
    # edit: nie pokrywa się z detail, bowiem keypointy COCO mają po 17 punktów,
    # natomiast keypyointy detailowe mają tylko 14 (brakuje oczu i uszu),
    # dlatego nie będziemy ich używać, napiszemy swoje na podstawie tych detailowych.
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
                cat['skeleton'] = self.REAL_KPTS # self.KPT_CATS[0]['skeleton'] <-- that one sucks (14 vs 17)
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
        kpts = self.to_dict(self.convert_kpts, self.INST_CATS)
        with open(join(OUTPUT_DIR, KPT), 'w') as outfile:
            json.dump(kpts, outfile)

    def dumpInst(self):
        inst = self.to_dict(self.convert_instances, self.INST_CATS)
        with open(join(OUTPUT_DIR, INST), 'w') as outfile:
            json.dump(inst, outfile)


if __name__ == '__main__':
    dc = DetailToCoco()
    dc.dumpInst()
    dc.dumpKpt()