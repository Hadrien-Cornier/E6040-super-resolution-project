## This file includes all the operations in patching. It can efficiently process patching with as small computation cost as possible.
import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import scipy.io

# Patch class for function 'patching'. It can get an item at a time.
class Patch(Dataset):
    def __init__(self, lr_patches, hr_patches):
        self.lr_data = lr_patches
        self.hr_data = hr_patches
       
    def transform(self, image):
        '''
        This function transforms the input patch (64,64,64) 
        from int16(12): 0-4095 to float: 0.0-1.0
        
        '''
        image_float = image.float() / 4095.0
        image_float = torch.unsqueeze(image_float,0)
        return image_float
    def __getitem__(self, idx):
        image_lr = self.lr_data[idx,:,:,:]
        image_hr = self.hr_data[idx,:,:,:]
        sample_hr = self.transform(image_hr)
        sample_lr = self.transform(image_lr)
        
        return (sample_lr, sample_hr)
    def __len__(self):
        return self.lr_data.shape[0]
    

def patching(lr_data, hr_data, patch_size = 2, cube_size = 64, usage = 1.0, is_training=True):
    '''
    This function makes patches from the input 3D image. It fulfills random 
    patch selection for training period, and sliding window patch seperation for 
    evaluation period.
    Note that dtype transform from int16(12): 0-4095 to float: 0.0-1.0 is 
    applied on patches in order to save memory.

    (Input) l/hr_data: a torch.ShortTensor (B,z,x,y) 
                  with dtype=int16 (the exact dtype is int12, from 0-4095)
    (Input) patch_size: define the patch size.
                        2 in this project (Default).
    (Input) cube_size: define the 3D cube size. 
                        64 in this project (Default).
    (Input) usage: The percentage of usage of one cluster of patches. 
                   For example: usage= 0.5 means to randomly pick 50% patches
                   from a cluster of 200 patches.
    (Input) is_training: True for training and validation set, 
                         False for evaluation and test set.
    (Output) patch_loader: a torch.DataLoader for picking patches from one batch.
    '''
    #import idx_mine to avoid unwanted patch indices
    mat = scipy.io.loadmat('ecbm6040/patching/idx_mine.mat')
    idx_mine = mat['idx_mine']
    idx_mine = idx_mine[0].tolist()
    # here we are not setting mines. We want all patches.
    idx_mine = []
    
    if is_training:
        stride = cube_size # patch stride
        lr_patches = lr_data.unfold(1, cube_size, stride).unfold(2, cube_size, stride).unfold(3, cube_size, stride)
        hr_patches = hr_data.unfold(1, cube_size, stride).unfold(2, cube_size, stride).unfold(3, cube_size, stride)
        lr_patches = lr_patches.contiguous().view(-1, cube_size, cube_size, cube_size)
        hr_patches = hr_patches.contiguous().view(-1, cube_size, cube_size, cube_size)
        patches = Patch(lr_patches, hr_patches)
        num_patches = len(patches)
        patch_split= usage # define the percentage of selecting patches in a batch
        patch_take = int(patch_split * num_patches) # the total patches selected in a batch 
        indices_undemined = list(range(num_patches))
        indices= list(set(indices_undemined) - set(idx_mine)) # exclude unwanted patch indices
        np.random.shuffle(indices)
        patch_indices = indices[:patch_take]
        patch_sampler = SubsetRandomSampler(patch_indices)
        patch_loader = torch.utils.data.DataLoader(dataset=patches, 
                                        batch_size=patch_size, 
                                        sampler=patch_sampler,
                                        shuffle=False)
        return patch_loader
    else:
        stride = cube_size # patch stride, in the recent paper, no overlapping (=cube_size)
        lr_patches = lr_data.unfold(1, cube_size, stride).unfold(2, cube_size, stride).unfold(3, cube_size, stride)
        hr_patches = hr_data.unfold(1, cube_size, stride).unfold(2, cube_size, stride).unfold(3, cube_size, stride)
        lr_patches = lr_patches.contiguous().view(-1, cube_size, cube_size, cube_size)
        hr_patches = hr_patches.contiguous().view(-1, cube_size, cube_size, cube_size)
        patches = Patch(lr_patches, hr_patches)
        num_patches = len(patches)
        indices_undemined = list(range(num_patches))
        patch_indices= list(set(indices_undemined) - set(idx_mine)) # exclude unwanted patch indices
        patch_sampler = SubsetRandomSampler(patch_indices)
        patch_loader = torch.utils.data.DataLoader(dataset=patches, 
                                        batch_size=patch_size, 
                                        sampler=patch_sampler,
                                        shuffle=False)
#         image_size=[256,320,320]
#         cube_size=64
#         margin=3
#         stride=cube_size-2*margin
#         padding=[20,17,17]
#         lr_data = torch.zeros([2,image_size[0]+2*padding[0],image_size[1]+2*padding[1],image_size[2]+2*padding[2]])
#         hr_data = torch.zeros([2,image_size[0]+2*padding[0],image_size[1]+2*padding[1],image_size[2]+2*padding[2]])
#         data = torch.ones([2,image_size[0],image_size[1],image_size[2]])
#         hr_data[:,padding[0]:image_size[0]+padding[0],padding[1]:image_size[1]+padding[1],padding[2]:image_size[2]+padding[2]]=data
#         hr_patches = hr_data.unfold(1, cube_size, stride).unfold(2, cube_size, stride).unfold(3, cube_size, stride)
        
#         hr_patches = hr_patches.contiguous().view(2,-1, cube_size, cube_size, cube_size)
#         real_patches = hr_patches[:,:,margin:-margin,margin:-margin,margin:-margin]
#         print(real_patches.shape)
#         real_images0=real_patches[:,0,(padding-margin):]
#         real_images1=real_patches[:,-1,:-(padding-margin)]
#         print(real_images0[0])
        return patch_loader
