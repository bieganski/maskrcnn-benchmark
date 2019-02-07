VENV_NAME = zpp


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
	conda install jupyter
	conda install requests
	conda install opencv
	conda install -c anaconda pillow
	# TODO nccl moze sie przydać, ale na razie nie instalować (może coś popsuć)
	# conda install -c anaconda nccl



# uwaga - powinno dzialac, ale jak dalej bedzie cos zle
# to warto sprobowac zrobic jeszcze conda install pytorch.
# to downgraduje jakies pakiety, ale moze pomoc


github:
	$(info Downloading deps from github...)

	# install torchvision
	# nie instalowac przez conda install! wtedy nie działa
	$(info Installing torchvision...)
	cd ./github; \
	git clone https://github.com/pytorch/vision.git; \
	cd vision; \
	python3 setup.py install


	# install pycocotools
	$(info Installing pycocotools...)
	cd ./github; \
	git clone https://github.com/cocodataset/cocoapi.git; \
	cd cocoapi/PythonAPI; \
	python3 setup.py build_ext install


build:
	$(info Building Mask-RCNN...)
	python3 setup.py build develop



# source ...
pascal:
	cd pascal; git clone https://github.com/ccvl/detail-api
	cd pascal/detail-api; python3 ./download.py pascal .
	cd pascal/detail-api; python3 ./download.py trainval_withkeypoints .



# to robic z 'source ...'
all:  install_deps github pascal build



.PHONY: github

#
# uwaga !
# create_env - normalnie
# install_deps, github etc. - musi być zrobione source activate <nazwa venva>
#
