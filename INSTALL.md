## Installation

### Requirements:
- PyTorch 1.0 from a nightly release. Installation instructions can be found in https://pytorch.org/get-started/locally/
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name maskrcnn_benchmark
conda activate maskrcnn_benchmark

# this installs the right pip and dependencies for the fresh python
conda install ipython

# maskrcnn_benchmark and coco api dependencies
pip install -r requirements.txt

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install pytorch-nightly cudatoolkit=9.0 -c pytorch

export INSTALL_DIR=$PWD
# install torchvision
cd $INSTALL_DIR
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```

### Option 2: Docker Image (Requires CUDA, Linux only)

Build image with defaults (`CUDA=9.0`, `CUDNN=7`):

    nvidia-docker build -t maskrcnn-benchmark docker/
    
Build image with other CUDA and CUDNN versions:

    nvidia-docker build -t maskrcnn-benchmark --build-arg CUDA=9.2 --build-arg CUDNN=7 docker/ 
    
Build and run image with built-in jupyter notebook(note that the password is used to log in jupyter notebook):

    nvidia-docker build -t maskrcnn-benchmark-jupyter docker/docker-jupyter/
    nvidia-docker run -td -p 8888:8888 -e PASSWORD=<password> -v <host-dir>:<container-dir> maskrcnn-benchmark-jupyter


### Option 3: Makefile

You can simply build everything & download all the datasets using makefile:
```bash
make prepare_download
conda activate zpp
make install_deps
make build
```

Unfortunately, it takes quite some time and downloads a lot of data. However, you can use longer version and comment out unnecessary lines:

```bash
# ---------- make prepare_download ----------
make create_venv       # Creates conda venv called 'zpp'
make download           # Download all the github repositories
make pascal_dload       # Downloads Pascal in detail datasets

conda activate zpp      # Activates conda venv

make install_deps       # Installs all the required pasckages

# ---------- make build -----------
make github_build       # Builds & installs torchvision & coco-api for Python
make pascal_build       # Builds & installs Pascal in detail-api for Python
make mask_build         # Builds & installs maskrcnn-benchmark
```

The above path has been tested on Ubunut 18.04
