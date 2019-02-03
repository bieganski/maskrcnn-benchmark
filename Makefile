VENV_NAME=zpp

default:
	build

create_venv:
	$(info Preparing env...)
	conda create --name $(VENV_NAME)

install_deps:
	source activate $(VENV_NAME); \
	conda install ipython; \
	pip install ninja yacs cython matplotlib; \
	conda install pytorch-nightly -c pytorch
	conda install -c anaconda cudnn


github:
	$(info Downloading from github...)
	# mkdir -p ./github

	# install torchvision
	$(info Installing torchvision...)
	cd ~/github; \
	git clone https://github.com/pytorch/vision.git; \
	cd vision; \
	python3 setup.py install

	# install pycocotools
	$(info Installing pycocotools...)
	cd ~/github; \
	git clone https://github.com/cocodataset/cocoapi.git; \
	cd cocoapi/PythonAPI; \
	python3 setup.py build_ext install


build:
	$(info Building Mask-RCNN...)
	python3 setup.py build develop


all: prepare_venv github build


.PHONY: github

#
# uwaga !
# create_env - normalnie
# install_deps, github etc. - musi być zrobione source activate <nazwa venva>
#
