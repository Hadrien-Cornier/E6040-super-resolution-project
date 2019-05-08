# E6040-super-resolution-project

TODO : 
testing 
visualisation
comments
readme

### Group: DPLN
* 	Huixiang Zhuang hz2538@COLUMBIA.EDU
>https://github.com/hz2538
* 	Hadrien Cornier hc3040@COLUMBIA.EDU
>https://github.com/Hadrien-Cornier

### Getting Started
These instructions will provide you a guideline for our basic functions as well as how to running on your machine for development and testing purposes.
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

ecbm6040/patching
>contains patchloader.py which takes full medical 3D images of dimensions 256x320x320 as input and cuts them into 4x5x5=100 patches of size 64x64x64

ecbm6040/model
>contains mDCSRN_WGAN.py which is the torch file containing the definition of the Generator and Discriminator neural networks

ecbm6040/dataloader
>contains a custom dataloader than can read medical images using the specialized nibabel library
