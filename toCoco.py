#!/usr/bin/python3

import json
from os.path import join

TRAINVAL_PATH='./pascal/detail-api'
DETAIL_ANNS = './trainval_withkeypoints.json'
OUTPUT_DIR='./pascal'

OUTPUT = 'kpt.json'

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
            kpt_obj['iscrowd'] = 0
            id += 1
        print(len(kpts))
        print(type(kpts))
        return kpts

    def convert_instances(self):
        segm = self.d['annos_segmentation']  # contains semgmentation and bboxes
        for el in segm:
            el['segmentation'] = [[]]
            el['iscrowd'] = 0  # sometimes it is 0 wtf
            el["keypoints"] = 42*[0]
            el['num_keypoints'] = 0
        print(len(segm))
        print(type(segm))
        return segm

    def mergeeeeeee(self):
       a = self.convert_kpts()
       a.extend(self.convert_instances())
       return a

    def to_dict(self, cats):
        res = dict()
        res['info'] = self.d['info']
        # res['images'] = self.d['images']
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
