# E6040-super-resolution-project

TODO : 
- [ ] testing 
- [ ] visualisation
- [x] comments
- [x] readme

### Group: DPLN
* 	Huixiang Zhuang hz2538@COLUMBIA.EDU
>https://github.com/hz2538
* 	Hadrien Cornier hc3040@COLUMBIA.EDU
>https://github.com/Hadrien-Cornier

### Getting Started
These instructions will provide you a guideline for our basic functions as well as how to running on your machine for development and testing purposes.

#### Description 
This repo aims at reproducing the results of the paper https://arxiv.org/abs/1803.01417. Here is the abstract :
>High-resolution (HR) magnetic resonance images (MRI) provide detailed anatomical information important for clinical application and> quantitative
image analysis. However, HR MRI conventionally comes at the cost of longer
scan time, smaller spatial coverage, and lower signal-to-noise ratio (SNR). Recent studies have shown that single image super-resolution (SISR), a technique
to recover HR details from one single low-resolution (LR) input image, could
provide high quality image details with the help of advanced deep convolutional
neural networks (CNN). However, deep neural networks consume memory heavily and run slowly, especially in 3D settings. In this paper, we propose a novel
3D neural network design, namely a multi-level densely connected super-resolution network (mDCSRN) with generative adversarial network >(GAN)–guided
training. The mDCSRN trains and inferences quickly, and the GAN promotes
realistic output hardly distinguishable from original HR images. Our results from
experiments on a dataset with 1,113 subjects shows that our new architecture
outperforms other popular deep learning methods in recovering 4x resolutiondowngraded images and runs 6x faster.

#### Prerequisites
To install the necessary packages, run the following command :

<code> pip install nibabel numpy pickle-mixin pandas matplotlib torch torchvision</code>
#### Introducing the files in project
DCSRN+SRGAN.ipynb
>This script is our main jupyter notebook. Implemented all experimental results.
>

Pretrain_G.ipynb
>This is the jupyter notebook for generator pretraining. 
>The results are merged into main DCSRN+SRGAN.ipynb.

Train_WGAN.ipynb
>This is the jupyter notebook for WGAN training (formal training).
>The results are merged into main DCSRN+SRGAN.ipynb.

loaddata.ipynb
>An example of data loading from google storage.

*ecbm6040 backend*

* ecbm6040/patching
>contains patchloader.py which takes full medical 3D images of dimensions 256x320x320 as input and cuts them into 4x5x5=100 patches of size 64x64x64

* ecbm6040/model
>contains mDCSRN_WGAN.py which is the torch file containing the definition of the Generator and Discriminator neural networks

* ecbm6040/dataloader
>contains a custom dataloader than can read medical images using the specialized nibabel library
