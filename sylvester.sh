#!/bin/bash

# script which sets up the maskrcnn-benchmark on Sylvester

echo "Downloading Anaconda..."
cd ./..
echo "Destination: `pwd`"
wget https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
chmod +x Anaconda3-2018.12-Linux-x86_64.sh
echo "About to install anaconda..."
echo "You'll have to agree to their terms"
echo "Then, the instalation begins"
echo "You should agree to add anaconda to .bashrc"
echo "Don't install VSCode on serwers"
echo "Do you copy? [Y/n]"
read ans
if [ ${ans} == "n" ]; then
    echo "OK, bye"
    exit 1
fi
./Anaconda3-2018.12-Linux-x86_64.sh

echo "Anaconda installed, setting up maskrcnn-benchmark"
cd ./maskrcnn-benchmark
echo "Running make in `pwd`"
make create_venv
make download
cd ./pascal/detail-api
echo "Downlaoding detail dataset to `pwd`"
wget https://students.mimuw.edu.pl/~kb392558/uploaded/download.tar
cd ./../..
make pascal_dload
conda activate zpp
echo "Installing dependencies..."
make install_deps
echo "Building..."
make build
echo "Done! Have a good day, bye."

