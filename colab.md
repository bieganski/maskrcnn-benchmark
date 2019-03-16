# Google Colab

If you don't have a PC capable of running training on GPU, you can use Google Colab. All you have to do is to open the notebook and make sure, you're using GPU runtime. Then simply run the following code in one (or more) cell(s).

```
!git clone https://github.com/bieganski/maskrcnn-benchmark.git
!cd maskrcnn-benchmark && make download
!cd maskrcnn-benchmark/pascal/detail-api && wget http://students.mimuw.edu.pl/~kb392558/uploaded/download.tar
!cd maskrcnn-benchmark && make pascal_dload
!cd maskrcnn-benchmark && pip3 install -r requirements.txt
!pip3 install torch torchvision yacs scikit-image
!cd maskrcnn-benchmark && make build
!cd maskrcnn-benchmark && make create_minimal_voc_dataset
```

And you're ready to go. You can check if you can train the network.

```
!cd maskrcnn-benchmark && make train
```

Remember, that if you lose your connection with the machine, you'll have to set it up once again.
