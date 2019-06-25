## Installation

### Requirements:
- Anaconda3

The rest of the requirements are downloaded in the deployment process.

### Deployment

You can simply build everything & download detail-api using makefile:
```bash
make prepare_download
conda activate zpp
make install_deps
make build
```

Unfortunately, it takes quite some time and downloads a lot of data. Plus you'll have to download the dataset using your web browser. However, you can also use longer version and comment out unnecessary lines:

```bash
# ---------- make prepare_download ----------
make create_venv        # Creates conda venv called 'zpp'
make download           # Download all the github repositories
cd pascal/detali-api    # Not included in make prepare_downlaod
wget http://students.mimuw.edu.pl/~kb392558/uploaded/download.tar
# the above downlaods detali annotations; Not included in make prepare_download
cd ./../..              # Not included in make prepare_download
make pascal_dload       # Downloads Pascal in detail datasets

conda activate zpp      # Activates conda venv

make install_deps       # Installs all the required pasckages

# ---------- make build -----------
make github_build       # Builds & installs torchvision & coco-api for Python
make pascal_build       # Builds & installs Pascal in detail-api for Python
make mask_build         # Builds & installs maskrcnn-benchmark
```

The above path has been tested on Ubunut 18.04

### Training

In order to run the training using our configuration you should first change the Detail annotations, prepare the images, prepare the images, and finally you can run the training process.

```bash
chmod +x toCoco.py
./toCoco.py

chmod +x sort_images.sh
./sort_images.sh

make train 			# uses 1 GPU
make multitrain		# uses 4 GPUs
```
