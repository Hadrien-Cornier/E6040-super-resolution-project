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
                 criterion, device,
                 max_step=500000, lr=1e-4, 
                 patch_size=2, cube_size=64, 
                 usage=1.0, pretrained=' '):
    since = time.time()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print ("Generator pre-training...")
    if pretrained != ' ':
        model = torch.load(pretrained)
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
            
            for lr_datas, hr_datas in dataloaders[phase]:
                patch_loader=patching(lr_datas, hr_datas, 
                                      patch_size = patch_size, 
                                      cube_size = cube_size, 
                                      usage=usage, is_training=True)
            
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
                    if (step % 5000) == 0:
                        print('Step: {}, loss= {:.4f}'.format(step, loss.item()))
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
                        torch.save(model,'models/pretrained_G')
                        return model
                    # statistics
                    batch_loss += loss.item() * lr_patches.size(0)
            epoch_loss = batch_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
    
        time_elapsed = time.time() - since
        print('Now the training uses {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
        print()