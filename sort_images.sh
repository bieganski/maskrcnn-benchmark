#!/bin/bash

VOC_PATH=pascal/detail-api/VOCdevkit/VOC2010
TRAIN_PATH=${VOC_PATH}/train
TEST_PATH=${VOC_PATH}/test
VAL_PATH=${VOC_PATH}/val
IMAGES_PATH=${VOC_PATH}/JPEGImages

if [ -f "pascal/train_list" ] ; then
    if [ -d "${TRAIN_PATH}" ] ; then
        rm -rf ${TRAIN_PATH}
    fi
    mkdir ${TRAIN_PATH}
    while read f ; do
        cp ${IMAGES_PATH}/${f} ${TRAIN_PATH}/${f}
    done < pascal/train_list
else 
    echo "No file for train split"
fi

# if [ -f "pascal/test_list" ] ; then
#     if [ -d "${test_PATH}" ] ; then
#         rm -rf ${test_PATH}
#     fi
#     mkdir ${test_PATH}
#     while read $f ; do
#         cp ${IMAGES_PATH}/${f} ${test_PATH}/${f}
#     done < pascal/test_list
# else 
#     echo "No file for test split"
# fi

if [ -f "pascal/val_list" ] ; then
    if [ -d "${val_PATH}" ] ; then
        rm -rf ${val_PATH}
    fi
    mkdir ${val_PATH}
    while read f ; do
        cp ${IMAGES_PATH}/${f} ${VAL_PATH}/${f}
    done < pascal/val_list
else 
    echo "No file for val split"
fi
