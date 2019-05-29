#!/usr/bin/env	python3

import json

a = json.load(open('./pascal/detail_val.json', 'rb'))

ZOSTAW_JEDEN_NA = 100

i = 0
nowe_zdjecia = []
zdjecia = set()
for img in a['images']:
	if i % ZOSTAW_JEDEN_NA == 0:
		zdjecia.add(img['id'])
		nowe_zdjecia.append(img)
	i += 1

a['images'] = nowe_zdjecia

nowe_anno = []

for anno in a['annotations']:
	if anno['image_id'] in zdjecia:
		nowe_anno.append(anno)

a['annotations'] = nowe_anno

with open('./pascal/coco_detail_minival.json', 'w') as outfile:
	json.dump(a, outfile, sort_keys=True, indent=2)


