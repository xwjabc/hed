## A PyTorch Reimplementation of HED

#### Introduction

This is a PyTorch reimplementation of [Holistically-nested Edge Detection (HED)](https://arxiv.org/abs/1504.06375). The code is evaluated on Python 3.6 with PyTorch 1.0 (CUDA9, CUDNN7) and MATLAB R2018b.

#### Instructions

##### Prepare

1. Clone the repository:

   ```bash
   git clone https://github.com/xwjabc/hed.git
   ```

2. Download and extract the data:

   ```bash
   cd hed
   wget https://cseweb.ucsd.edu/~weijian/static/datasets/hed/hed-data.tar
   tar xvf ./hed-data.tar
   ```

##### Train and Evaluate

1. Train:

   ```bash
   python hed.py --vgg16_caffe ./data/5stage-vgg.py36pickle
   ```

   The results are in `output` folder. In the default settings, the HED model is trained for 40 epochs, which takes ~27hrs with one NVIDIA Geforce GTX Titan X (Maxwell). 

2. Evaluate:

   ```bash
   cd eval
   (echo "data_dir = '../output/epoch-39-test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
   ```

   The evaluation process takes ~7hrs with Intel Core i7-5930K CPU @ 3.50GHz.

Besides, based on my observation, the evaluated performance is somewhat stable after 5 epochs (5 epochs: **ODS=0.788 OIS=0.808** vs. 40 epochs: **ODS=0.787 OIS=0.807**).

##### Evaluate the Pre-trained Models

1. Evaluate the my pre-trained version:

   ```bash
   python hed.py --checkpoint ./data/hed_checkpoint.pt --output ./output-mypretrain --test
   cd eval
   (echo "data_dir = '../output-mypretrain/test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
   ```

   The result should be similar to **ODS=0.787 OIS=0.807**.

2. Evaluate the official pre-trained version:

   ```bash
   python hed.py --caffe_model ./data/hed_pretrained_bsds.py36pickle --output ./output-officialpretrain --test
   cd eval
   (echo "data_dir = '../output-officialpretrain/test'"; cat eval_edge.m)|matlab -nodisplay -nodesktop -nosplash
   ```

   The result should be similar to **ODS=0.788 OIS=0.806**.

#### Acknowledgement

This reimplementation is based on lots of prior works. Thanks to [Saining](https://github.com/s9xie/hed) for the original Caffe implementation. Thanks to [@meteorshowers](https://github.com/meteorshowers/hed-pytorch) for a PyTorch implementation where I adopt most of the code from. Thanks to [@jmbuena](https://github.com/jmbuena/toolbox.badacost.public) for a fixed version of Piotr's Toolbox. Thanks to [Berkeley Institute for Data Science](https://github.com/BIDS/BSDS500) which provides a mirror of BSDS500 dataset (the original link to the dataset seems broken).
