VENV_NAME=zpp

default:
	build

prepare_venv:
	$(info Preparing env...)
	conda create --name $(VENV_NAME)
	source activate $(VENV_NAME); \
	conda install ipython; \
	pip install ninja yacs cython matplotlib; \
	conda install pytorch-nightly -c pytorch


github:
	$(info Downloading from github...)
	# mkdir -p ./github

	# install torchvision
	$(info Installing torchvision...)
	cd ~/github; \
	git clone https://github.com/pytorch/vision.git; \
	cd vision; \
	source activate $(VENV_NAME); \
	python3 setup.py install

	# install pycocotools
	$(info Installing pycocotools...)
	cd ~/github; \
	git clone https://github.com/cocodataset/cocoapi.git; \
	cd cocoapi/PythonAPI; \
	source activate $(VENV_NAME); \
	python3 setup.py build_ext install


build:
	$(info Building Mask-RCNN...)
	source activate $(VENV_NAME); \
	python3 setup.py build develop


all: prepare_venv github build


.PHONY: github

