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
TODO
#### Introducing the files in project
main.ipynb
>This script is our main jupyter notebook. Implemented our whole project.
>
loaddata.ipynb
>This script is our second jupyter notebook. However, you have to run it first, since it will read the tiff file and convert it to array
>make sure you hace already install libtiff before using it.
./DPLN/pyramidLSTM.py
>
>
./DPLN/Net_PyramidLSTM.py
>
>
./DPLN/MDLSTM.py
>
>
./DPLN/Net_MDLSTM.py
>Above py files define the layers.
>
./DPLN/ProcGenerator.py
>This py file is refered to as a batch generator.
>
./DPLN/ProcessingV2.py
>This py file provides the basic function for image preprocessing, including ROF denoise, Z-score normalization and augmentation.
>
test-volume.tiff, train-volume.tiff, train-label.tiff
>They are the training data, labels and test data. Because it is a little tricky to download them, we provide them in the submission
>
./model
>Save the trained model in this folder
mainOut256.iqynb
>this is the automated produced version when we trained our model on the serverl. It record the printout output while training the Pyramid-LSTM, with a batch size 256*256
>
