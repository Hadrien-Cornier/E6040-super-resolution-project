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
from ecbm6040.patching.patchloader import patching, depatching
from ecbm6040.metric.eval_metrics import ssim, psnr, nrmse

class WGAN_GP(object): 
    def __init__(self, netG, netD, 
             supervised_criterion, device, 
             lr=1e-6, joint_opt_param=0.001):
        self.netG = netG
        self.netD = netD
        self.supervised_criterion = supervised_criterion
        self.device = device
        self.lr = lr
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr)
        self.lmda = joint_opt_param
                
    def wasserstein_loss(self, D_fake, D_real= torch.Tensor([0.0])):
        D_real = D_real.cuda(self.device)
        D_loss = - (torch.mean(D_real) - torch.mean(D_fake))
        G_loss = - self.lmda * torch.mean(D_fake)
        return G_loss, D_loss
    
    def updateD(self, lr_patches, hr_patches):
        # forward
        for p in self.netG.parameters():
            p.requires_grad = False # to avoid computation
        for p in self.netD.parameters():
            p.requires_grad = True # to avoid computation
        # input SR to D (fake)
        sr_patches = self.netG(lr_patches)
        D_fake = self.netD(sr_patches)
        # input HR to D (real)
        D_real = self.netD(hr_patches)
        # Supervised Loss
        # Calculate L1 Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # WGAN's Loss
        # Calculate Wasserstein Loss
        G_loss, D_loss = self.wasserstein_loss(D_fake, D_real)
        # Semi-supervised Loss (main loss)
        loss = L1_loss + G_loss
        
        # backward + optimize only if in training phase
        D_loss.backward()
        self.optimizerD.step()

        # weight clipping
        for p in self.netD.parameters():
            p.data.clamp_(-0.01, 0.01)
        return sr_patches, D_loss, G_loss, loss
    
    def updateG(self, lr_patches, hr_patches):
        for p in self.netG.parameters():
            p.requires_grad = True # to avoid computation
        for p in self.netD.parameters():
            p.requires_grad = False # to avoid computation
        # input SR to D (fake)
        sr_patches = self.netG(lr_patches)
        D_fake = self.netD(sr_patches)
        # Supervised Loss
        # Calculate L1 Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # WGAN's Loss
        # Calculate Wasserstein Loss
        G_loss,_ = self.wasserstein_loss(D_fake)
        # Semi-supervised Loss (main loss)
        loss = L1_loss + G_loss
        # backward + optimize only if in training phase
        loss.backward()
        self.optimizerG.step()
        return sr_patches, G_loss, loss
    
    def forwardDG(self, lr_patches, hr_patches):
        # input SR to D (fake)
        sr_patches = self.netG(lr_patches)
        D_fake = self.netD(sr_patches)
        # input HR to D (real)
        D_real = self.netD(hr_patches)
        # Supervised Loss
        # Calculate L1 Loss
        L1_loss = self.supervised_criterion(sr_patches, hr_patches)
        # WGAN's Loss
        # Calculate Wasserstein Loss
        G_loss, D_loss = self.wasserstein_loss(D_fake, D_real)
        # Semi-supervised Loss (main loss)
        loss = L1_loss + G_loss
        return sr_patches, D_loss, G_loss, loss
        
    def training(self, dataloaders,
                 max_step=550000, first_steps=10000, 
                 patch_size=2, cube_size=64, usage=1.0, 
                 pretrainedG = ' ',pretrainedD =' '):
        since = time.time()

        print ("WGAN training...")
        if pretrainedG != ' ':
            self.netG = torch.load(pretrainedG)
            step = int(re.sub("\D", "", pretrainedG))  #start from the pretrained model's step
            train_loss=[]
            train_D_loss=[]
            val_loss=[]
            val_D_loss=[]
        else:
            # record loss function of the whole period
            step = 0
            train_loss=[]
            train_D_loss=[]
            val_loss=[]
            val_D_loss=[]
        if pretrainedD != ' ':
            self.netD = torch.load(pretrainedD)
            # recall the loss history to continue
            f=open('loss_history/train_loss_step{}.txt'.format(step),'rb')
            train_loss= pickle.load(f)
            f.close()
            f=open('loss_history/train_loss_D_step{}.txt'.format(step),'rb')
            train_D_loss= pickle.load(f)
            f.close()
            f=open('loss_history/val_loss_step{}.txt'.format(step),'rb')
            val_loss= pickle.load(f)
            f.close()
            f=open('loss_history/val_loss_D_step{}.txt'.format(step),'rb')
            val_D_loss= pickle.load(f)
            f.close()
        imbl = 1 # count for imbalance training
        extra = 1 # count for extra D training
        # pretrained step
        num_steps_pre = step
        
        while(step < max_step):
            print('Step {}/{}'.format(step, max_step))
            print('-' * 10)
            mean_generator_content_loss = 0.0
            mean_discriminator_loss = 0.0
            # Each epoch has 10 training and validation phases
            for fold in range(10):
                for phase in ['train', 'val']:
                    if phase == 'train':
                        self.netD.train()  # Set model to training mode
                        self.netG.train()
                    else:
                        self.netD.eval()   # Set model to training mode
                        self.netG.eval()
                    
                    batch_loss = []
                    batch_G_loss = []
                    batch_D_loss = []
                    val_ssim = []
                    val_psnr = []
                    val_nrmse = []

                    for lr_data, hr_data in dataloaders[phase][fold]:
                        # This time, validation period would be different 
                        # since they need to be merged again to measure the evaluation metrics.
                        
                        if phase == 'train':
                            patch_loader=patching(lr_data, hr_data, 
                                                  patch_size = patch_size, 
                                                  cube_size = cube_size, 
                                                  usage=usage, is_training=True)
                        else:
                            patch_loader=patching(lr_data, hr_data, 
                                                  patch_size = patch_size, 
                                                  cube_size = cube_size, 
                                                  usage=usage, is_training=False)
                            sr_data_cat = torch.Tensor([]) # for concatenation
                        for lr_patches, hr_patches in patch_loader:
                            lr_patches=lr_patches.cuda(self.device)
                            hr_patches=hr_patches.cuda(self.device)
                            # zero the parameter gradients
                            self.optimizerG.zero_grad()
                            self.optimizerD.zero_grad()

                            if phase == 'train':
                                # Training phase
                                with torch.set_grad_enabled(True):
                                ##########################################################
                                # (1) Update D network in following conditions:
                                #1.in first steps;
                                #2.every 500 steps for extra 200 steps;
                                #3.consecutive 7 steps.
                                # (2) Update G network in following conditions:
                                #1.after consecutive 7 steps Update D, update G for 1 step.
                                ##########################################################
                                    # Update D Case 1: in first steps
                                    if (step < num_steps_pre + first_steps):
                                        sr_patches, D_loss, G_loss, loss = self.updateD(lr_patches, hr_patches)
                                        step += 1          # we count step here
                                    # Regular training
                                    else: 
                                        if ((imbl != 7) and (extra == 0)):
                                            # Update D Case 3: consecutive 7 steps
                                            sr_patches, D_loss, G_loss, loss = self.updateD(lr_patches, hr_patches)
                                            step += 1
                                            imbl += 1
                                        if ((imbl == 7) and (extra == 0)):
                                            # Update G Case 1: update G for 1 step
                                            sr_patches, G_loss, loss = self.updateG(lr_patches, hr_patches)
                                            step += 1
                                            imbl = 1 # set to zero
                                        # Update D Case 2: every 500 steps for extra 200 steps
                                        if ((step % 500 == 0) or (extra != 0)):
                                            sr_patches, D_loss, G_loss, loss = self.updateD(lr_patches, hr_patches)
                                            step += 1
                                            extra += 1
                                            if (extra == 200):
                                                extra = 1
                                #This print out is only for early inspection
                                if (step % 500) == 0:
                                    print('Step: {}, loss= {:.4f}, D_loss= {:.4f}, G_loss= {:.4f}'.format(step, loss.item(), D_loss.item(), G_loss.item()))

                                # statistics
                                batch_loss = np.append(batch_loss, loss.item())
                                batch_G_loss = np.append(batch_G_loss, G_loss.item())
                                batch_D_loss = np.append(batch_D_loss, D_loss.item())
                                
                                if ((step - num_steps_pre) % int((max_step - num_steps_pre) // 10)) ==0:
                                    # save intermediate models
                                    torch.save(self.netG,'models/WGAN_G_step{}'.format(step))
                                    torch.save(self.netD,'models/WGAN_D_step{}'.format(step))
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
                                    # record instant loss
                                    train_loss = np.append(train_loss, batch_loss)
                                    train_D_loss = np.append(train_D_loss, batch_D_loss)
                                    f=open('loss_history/train_loss_step{}.txt'.format(step),'wb')
                                    pickle.dump(train_loss ,f)
                                    f.close()
                                    f=open('loss_history/train_loss_D_step{}.txt'.format(step),'wb')
                                    pickle.dump(train_D_loss ,f)
                                    f.close()
                                    f=open('loss_history/val_loss_step{}.txt'.format(step),'wb')
                                    pickle.dump(val_loss ,f)
                                    f.close()
                                    f=open('loss_history/val_loss_D_step{}.txt'.format(step),'wb')
                                    pickle.dump(val_D_loss ,f)
                                    f.close()                                    
                                
                                if (step == max_step):
                                    print("True")
                                    # record instant loss
                                    train_loss = np.append(train_loss, batch_loss)
                                    train_D_loss = np.append(train_D_loss, batch_D_loss)
                                    f=open('loss_history/train_loss_history.txt','wb')
                                    pickle.dump(train_loss ,f)
                                    f.close()
                                    f=open('loss_history/train_loss_D_history.txt','wb')
                                    pickle.dump(train_D_loss ,f)
                                    f.close()
                                    f=open('loss_history/val_loss_history.txt','wb')
                                    pickle.dump(val_loss ,f)
                                    f.close()
                                    f=open('loss_history/val_loss_D_history.txt','wb')
                                    pickle.dump(val_D_loss ,f)
                                    f.close()
                                    print('Complete {} steps'.format(step))

                                    torch.save(self.netG,'models/final_model_G')
                                    torch.save(self.netD,'models/final_model_D')
                                    return self.netG, self.netD
                            
                            else:
                                # Validation phase
                                with torch.set_grad_enabled(False):
                                    sr_patches, D_loss, G_loss, loss = self.forwardDG(lr_patches, hr_patches)
                                # statistics
                                batch_loss = np.append(batch_loss, loss.item())
                                batch_G_loss = np.append(batch_G_loss, G_loss.item())
                                batch_D_loss = np.append(batch_D_loss, D_loss.item())
                                # concatenate patches, send patches to cpu to save GPU memory
                                sr_data_cat = torch.cat([sr_data_cat, sr_patches.to("cpu")],0)
                                
                            
                        if phase == 'val':
                            # calculate the evaluation metric
                            sr_data = depatching(sr_data_cat, lr_data.size(0))
                            f=open('example_images/image_sr_step{}.txt'.format(step),'wb')
                            pickle.dump(sr_data.cpu().numpy() ,f)
                            f.close()
                            batch_ssim = ssim(hr_data, sr_data)
                            batch_psnr = psnr(hr_data, sr_data)
                            batch_nrmse = nrmse(hr_data, sr_data)
                            val_ssim = np.append(val_ssim, batch_ssim)
                            val_psnr = np.append(val_psnr, batch_psnr)
                            val_nrmse = np.append(val_nrmse, batch_nrmse)
                            
                    mean_generator_content_loss = np.mean(batch_loss)
                    mean_discriminator_loss = np.mean(batch_D_loss)
                    if phase == 'val':
                        mean_ssim = np.mean(val_ssim)
                        std_ssim = np.std(val_ssim)
                        mean_psnr = np.mean(val_psnr)
                        std_psnr = np.std(val_psnr)
                        mean_nrmse = np.mean(val_nrmse)
                        std_nrmse = np.std(val_nrmse)
                        val_loss = np.append(val_loss, batch_loss)
                        val_D_loss = np.append(val_D_loss, batch_D_loss)
                        print('No. {} {} period. Mean main loss: {:.4f}. Mean discriminator loss: {:.4f}.'.format(fold+1, phase, mean_generator_content_loss, mean_discriminator_loss))
                        print('Metrics: subject-wise mean SSIM = {:.4f}, std = {:.4f}; mean PSNR = {:.4f}, std = {:.4f}; mean NRMSE = {:.4f}, std = {:.4f}.'.format(mean_ssim, std_ssim, mean_psnr, std_psnr, mean_nrmse, std_nrmse))
                    else:
                        train_loss = np.append(train_loss, batch_loss)
                        train_D_loss = np.append(train_D_loss, batch_D_loss)
                        print('No.{} {} period. Mean main loss: {:.4f}. Mean discriminator loss: {:.4f}'.format(fold+1, phase, mean_generator_content_loss, mean_discriminator_loss))
                        
                time_elapsed = time.time() - since
                print('Now the training uses {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
                print()
        return self.netG, self.netD