#!/usr/bin/env	python3

import json

a = json.load(open('./pascal/detail_val.json', 'rb'))

ZOSTAW_JEDEN_NA = 25

i = 0
nowe_zdjecia = []
for img in a['images']:
	if i % ZOSTAW_JEDEN_NA == 0:
		nowe_zdjecia.append(img)
	i += 1

a['images'] = nowe_zdjecia


with open('./pascal/coco_detail_minival.json', 'w') as outfile:
	json.dump(a, outfile, sort_keys=True, indent=2)


