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
echo "Anaconda installed"

echo "Adding cuda 9.0 to PATH in ~/.bashrc"
echo "You should make backup of it"
echo "Proceed? [Y/n]"
read ans
if [ ${ans} == "n" ]; then
    echo "OK, bye"
    exit 1
fi
echo "export PATH=/usr/local/cuda-9.0/bin\${PATH:+:\${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" >> ~/.bashrc
source ~/.bashrc
echo "You should see nvcc version 9.0"
nvcc --version
echo "You should now use \"source .bashrc\""
echo "I hope the instalation went well."

