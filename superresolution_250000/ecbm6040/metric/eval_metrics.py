import skimage.measure as measure
import torch
import numpy as np

def ssim(img_true, img_test):
    '''
    To be edited
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.numpy()
    
    img_test = img_test.numpy()
    
    ssim=[]
    for i in range(img_true.shape[0]):
        ssim = np.append(ssim, measure.compare_ssim(img_true[i], img_test[i]))
    return ssim

def psnr(img_true, img_test):
    '''
    To be edited
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.numpy()
    
    img_test = img_test.numpy()
    psnr=[]
    for i in range(img_true.shape[0]):
        psnr = np.append(psnr, measure.compare_psnr(img_true[i], img_test[i]))
    return psnr

def nrmse(img_true, img_test):
    '''
    To be edited
    '''
    img_true = img_true.float() / 4095.0
    img_true = img_true.numpy()
    
    img_test = img_test.numpy()
    nrmse=[]
    for i in range(img_true.shape[0]):
        nrmse = np.append(nrmse, measure.compare_nrmse(img_true[i], img_test[i]))
    return nrmse