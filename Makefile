VENV_NAME = zpp


add_alias:
	echo 'alias $(VENV_NAME)='source activate $(VENV_NAME)'' >> ~/.bashrc

create_venv:
	$(info Preparing env...)
	conda create --name $(VENV_NAME)


download:
	$(info Downloading deps from github...)

	# do NOT install by 'conda install'! it won't work
	$(info Installing torchvision...)
	cd ./github; \
	git clone https://github.com/pytorch/vision.git;


	$(info Installing pycocotools...)
	cd ./github; \
	git clone https://github.com/cocodataset/cocoapi.git

	$(info Installing Pascal in detail...)
	cd ./pascal; \
	git clone https://github.com/ccvl/detail-api



# use it to build github projects, without redownloading them
github_build:
	cd ./github/vision; python3 setup.py install
	cd ./github/cocoapi/PythonAPI; python3 setup.py build_ext install



# needs activation
pascal_dload:
	$(info Building Pascal in Detail...)
	cd pascal/detail-api; python3 ./download.py pascal .
	cd pascal/detail-api; python3 ./download.py trainval_withkeypoints .



pascal_build:
	make -C pascal/detail-api/PythonAPI
	make install -C pascal/detail-api/PythonAPI


# needs activation
build_all: github_build pascal_build
	$(info Building Mask-RCNN...)
	python3 setup.py build develop



prepare_download: create_venv download pascal_dload



# needs activation
install_deps:
	conda install --yes --file requirements.txt
	conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
	# TODO - from https://pytorch.org/get-started/locally/ choose your version of cuda
	pip install yacs
	# TODO nccl might be usuful, not listed in requirements.txt
	# !!! in case of any problems it might be helpful to reinstall pytorch (nightly?)


# needs activation
build: build_all


.PHONY: github
.PHONY: pascal
.PHONY: build
