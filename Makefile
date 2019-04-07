VENV_NAME = zpp


add_alias:
	echo 'alias $(VENV_NAME)='source activate $(VENV_NAME)'' >> ~/.bashrc

create_venv:
	$(info Preparing env...)
	conda create --name $(VENV_NAME)


_download_torchvision:
	# do NOT install by 'conda install'! it won't work
	$(info Installing torchvision...)
	cd ./github && \
	git clone https://github.com/pytorch/vision.git;

_download_coco:
	$(info Installing pycocotools...)
	cd ./github && \
	git clone https://github.com/cocodataset/cocoapi.git

download_pascal:
	$(info Installing Pascal in detail (from our fork)...)
	mkdir pascal; \
	cd ./pascal && \
	git clone https://github.com/bieganski/detail-api


download: _download_torchvision _download_coco download_pascal

# use it to build github projects, without redownloading them
github_build:
	cd ./github/vision && python3 setup.py install
	cd ./github/cocoapi/PythonAPI && python3 setup.py build_ext install



# needs activation
pascal_dload:
	$(info Building Pascal in Detail...)
	chmod +x pascal/detail-api/download.py
	cd pascal/detail-api && python3 ./download.py pascal .
	cd pascal/detail-api && python3 ./download.py trainval_withkeypoints .



pascal_build:
	make -C pascal/detail-api/PythonAPI
	make install -C pascal/detail-api/PythonAPI


# remember to keep somewhere trainval_withkeypoints.json and VOCkit saved
_detail_redload:
	rm -rf ./pascal/detail-api
	cd pascal && git clone https://github.com/bieganski/detail-api


detail_update: _detail_redload pascal_build

# run it in case of CUDA any undefined symbol error
mask_build:
	$(info Building Mask-RCNN...)
	rm -rf build/
	python3 setup.py build develop


# needs activation
build_all: github_build pascal_build mask_build



prepare_download: create_venv download pascal_dload



# needs activation
install_deps:
	conda install --yes --file requirements.txt
	# conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
	conda install --yes pytorch torchvision cudatoolkit=9.0 -c pytorch
	# TODO - from https://pytorch.org/get-started/locally/ choose your version of cuda
	pip install yacs scikit-image
	# TODO nccl might be usuful, not listed in requirements.txt
	# !!! in case of any problems it might be helpful to reinstall pytorch (nightly?)


# needs activation
build: build_all


modify_bashrc:
	cat to_bashrc.txt >> ~/.bashrc



VOC=./pascal/detail-api/VOCdevkit
NEW=MINIMAL
OLD=VOC2010
NUM_TRAIN_IMAGES=500
DETAIL=./pascal/trainval_withkeypoints.json

# remember not to use 2007 images, there are no annotations with them!
create_minimal_voc_dataset:
	# reproduce directory structure, with minimal test cases
	rm -rf ${VOC}/${NEW}
	mkdir ${VOC}/${NEW}
	# the way below - kinda lame, thus commented
	# rsync -a --include '*/' --exclude '*' ${VOC}/${OLD}/ ${VOC}/${NEW}
	# ls -1 ${VOC}/${OLD}/ImageSets/Main | grep "train" | head -n 2 | xargs -I {} cp ${VOC}/${OLD}/ImageSets/Main/{} ${VOC}/${NEW}/ImageSets/Main
	rsync -a ${VOC}/${OLD}/ ${VOC}/${NEW}
	cd ${VOC}/${NEW}/ImageSets/Main && sed -i '${NUM_TRAIN_IMAGES},$$ d' train.txt
	if [ ! -d "./datasets" ]; then mkdir datasets; fi
	if [ ! -f pascal ]; then ln -s ../pascal/ ./datasets/pascal; fi


# shows number of images (test, train, val) used by Detail API.
show_dataset_split:
	$(info you need to have jq installed)
	$(info wait several seconds...)
	@jq '.images[].phase' ${DETAIL} | cut -d \" -f 2 | sort | uniq -c

train:
	if [ -d "./trash" ]; then \
		cd ./trash && rm -rf *; \
	else mkdir trash; \
	fi
	python3 ./tools/train_net.py --config-file "./configs/detail/zpp_config.yaml"

NGPUS=4
multitrain:
	if [ -d "./trash3" ]; then \
		cd ./trash3 && rm -rf *; \
	else mkdir trash3; \
	fi
	python -m torch.distributed.launch --nproc_per_node=${NGPUS} ./tools/train_net.py --config-file "./configs/detail/zpp_config_4GPU.yaml"


to_coco:
	chmod +x ./toCoco.py
	./toCoco


.PHONY: github
.PHONY: pascal
.PHONY: build


