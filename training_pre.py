import os
import random
import time
import math
import numpy as np
import pickle
import re
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
import torchvision.utils as vutils
from ecbm6040.dataloader.CustomDatasetFromCSV import CustomDatasetFromCSV
from ecbm6040.patching.patchloader import patching

def training_pre(model, dataloaders, dataset_sizes,
                 criterion, device, ngpu,
                 max_step=500000, lr=1e-4, 
                 patch_size=2, cube_size=64, 
                 usage=1.0, pretrained=' '):
    """
    This function is the pretraining of Generator: Multi-level Densely Connected Super-Resolution Network (mDCSRN). 
    This is separated from WGAN_GP function for you can directly utilize it if you don't want to induce the with-GAN training.
    Args:
        dataloaders (torch.utils.data.DataLoader) - the torch dataloader you defined. For a default setting, use a dictionary with phase dataloaders['train', 'val']. See in the main.ipynb, or the loaddata.ipynb.
        dataset_sizes (dictionary) - a dictionary that with phase ['train', 'val']. The size of training set and the validation set are contained. 
        criterion (torch.nn.modules.loss) - the predefined loss function for the generator, in this project, we use nn.L1Loss().
        device (torch.device) - the device you set.
        ngpu (int) - how many GPU you use.
        max_step (int) - the maximum step of pretraining. By default, the value is 500000 according to the paper. (Note that we only use 250000)
        lr (float) - the learning rate for pretraining. By default, the value is 1e-4.
        patch_size (int) - the number of patches once send into the model. By default, the value is 2.
        cube_size (int) - the size of one patch (eg. 64 means a cubic patch with size: 64x64x64), this is exact the size of the model input. By default, the value is 64.
        usage (float) - the percentage of usage of one cluster of patches. For example: usage= 0.5 means to randomly pick 50% patches from a cluster of 200 patches. This is only used in training period. By default, the value is 1.0.
        pretrained (string) - the root of the saved pretrained model. 
    """
    since = time.time()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print ("Generator pre-training...")
    if pretrained != ' ':
        model.load_state_dict(torch.load(pretrained))
        # if transfer from a single gpu case, set multi-gpu here again.
        if (device.type == 'cuda') and (ngpu > 1):
            model = nn.DataParallel(model, list(range(ngpu)))
        step = int(re.sub("\D", "", pretrained))  #start from the pretrained model's step
    else:
        step = 0
    while(step < max_step):
        print('Step {}/{}'.format(step, max_step))
        print('-' * 10)
        epoch_loss = 0.0
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to training mode
            batch_loss = 0.0
            
            for lr_data, hr_data in dataloaders[phase]:
                patch_loader=patching(lr_data, hr_data, 
                                      patch_size = patch_size, 
                                      cube_size = cube_size, 
                                      usage=usage, is_training=True)
                patch_count = 0
                patch_loss = 0.0
                for lr_patches, hr_patches in patch_loader:
                    lr_patches=lr_patches.cuda(device)
                    hr_patches=hr_patches.cuda(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        sr_patches = model(lr_patches)
                        loss = criterion(sr_patches, hr_patches)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            step += 1          # we count step here
                            #This print out is only for early inspection
#                             if (step % 500) == 0:
#                                 print('Step: {}, loss= {:.4f}'.format(step, loss.item()))
                            if (step % int(max_step // 10)) ==0:
                                # save intermediate models
                                torch.save(model,'models/pretrained_G_step{}'.format(step))
                                # save example lr, sr images
                                f=open('example_images/example_lr_step{}.txt'.format(step),'wb')
                                pickle.dump(lr_patches.data.cpu().numpy() ,f)
                                f.close()
                                f=open('example_images/example_sr_step{}.txt'.format(step),'wb')
                                pickle.dump(sr_patches.data.cpu().numpy() ,f)
                                f.close()
                                f=open('example_images/example_hr_step{}.txt'.format(step),'wb')
                                pickle.dump(hr_patches.data.cpu().numpy() ,f)
                                f.close()
                            if (step == max_step):
                                print('Complete {} steps'.format(step))
                                # save model for single GPU and multi GPU
                                if ngpu > 1:
                                    torch.save(model.module.state_dict(),'models/pretrained_G_step{}'.format(step))
                                else:
                                    torch.save(model.state_dict(),'models/pretrained_G_step{}'.format(step))
                                return model
                    # statistics
                    patch_count += lr_patches.size(0)
                    patch_loss += loss.item() * lr_patches.size(0)
                batch_loss += patch_loss / patch_count    
            epoch_loss = batch_loss / dataset_sizes[phase]
            print('Step: {}, {} Loss: {:.4f}'.format(step, phase, epoch_loss))
    
        time_elapsed = time.time() - since
        print('Now the training uses {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        print()