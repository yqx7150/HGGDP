import os
import cv2
import glob
import h5py
import time
import math
import torch
import numpy as np
import pickle
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py as h5
from torchvision import transforms
from scipy.io import loadmat,savemat
from scipy import misc
from skimage.measure import compare_psnr,compare_ssim
from hggdp.siat.compare_hfen import compare_hfen
from hggdp.models.cond_refinenet_dilated import CondRefineNetDilated
from hggdp.siat.US_pattern import US_pattern
__all__ = ['SIAT_MULTICHANNEL_MODL']
def FT(x,csm):
    """ This is a the A operator as defined in the paper"""
    ncoil,nrow,ncol = csm.shape
    ccImg=np.reshape(x,(nrow,ncol) )
    coilImages=np.tile(ccImg,[ncoil,1,1])*csm;
    kspace=np.fft.fft2(coilImages)/np.sqrt(nrow * ncol)
    return kspace

def tFT(kspaceUnder,csm):
    """ This is a the A^T operator as defined in the paper"""
    ncoil,nrow,ncol = csm.shape
    #temp=np.zeros((ncoil,nrow,ncol),dtype=np.complex64)
    img=np.fft.ifft2(kspaceUnder)*np.sqrt(nrow*ncol)
    coilComb=np.sum(img*np.conj(csm),axis=0).astype(np.complex64)
    #coilComb=coilComb.ravel();
    return coilComb

#load the test image and the coil maps
#-----------------------------------------------------        
#unfortunately, due to complications in saving complex valued data, we save
#and load the complex and real parts seperately

filename='./MoDL_share_data/demoImage.hdf5' #set the correct path here
with h5.File(filename,'r') as f:
    org,csm,mask=f['tstOrg'][:],f['tstCsm'][:],f['tstMask'][:]
#print(org.shape,csm.shape,mask.shape)
orim = org[0]
csm = csm[0]
patt = mask[0]

#      other under sample pattern choose
#USp = US_pattern();
#patt = USp.generate_opt_US_pattern_1D([orim.shape[0], orim.shape[1]], R=6.7, max_iter=100, no_of_training_profs=15)
#patt = np.fft.fftshift(patt)
#misc.imsave('usp_patt_R2_%s.png'%str((np.sum(patt))/orim.shape[0]/orim.shape[1]),np.abs(patt))
#make the undersampled kspace
#usksp = ksp * np.tile(patt[:,:,np.newaxis],[1, 1, ksp.shape[2]])



class SIAT_MULTICHANNEL_MODL():
    def __init__(self, args, config):
        self.args = args
        self.config = config        
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        #assert os.path.exists(self.args.load_path),'load file path is not exists,please recheck your path'
            
    # save image
    def write_images(self,x,image_save_path):
        maxvalue = np.max(x)
        if maxvalue < 128:
            x = np.array(x*255.0,dtype=np.uint8)
        cv2.imwrite(image_save_path, x.astype(np.uint8))
    # test function
    def test(self):
        # make a network and load network weight to use later
        states = torch.load(os.path.join(self.args.log, 'checkpoint_100000.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet, device_ids=[0])
        scorenet.load_state_dict(states[0])
        scorenet.eval()
        # prepare test data and undersample mask
        
        undersample_method = 'random'
        undersample_factor = str((np.sum(patt))/orim.shape[0]/orim.shape[1])

        ori_complex = orim
        
        #ori_complex = loadmat(self.args.load_path)["Img"]
        ori_complex = ori_complex/np.max(np.abs(ori_complex))
        mask = patt
        #print(ori_complex.shape,csm.shape,mask.shape)
        ksp = FT(ori_complex,csm)
        print(ksp.shape)
        if len(mask.shape)==2:
            mask=np.tile(mask,(csm.shape[0],1,1))
        #get multi coil undersample kspace by mask
        
        usksp = ksp * mask

        zero_fiiled = tFT(usksp,csm)
        # use for getting degrade img and psnr,ssim,hfen in iteration
        psnr = compare_psnr(255*abs(zero_fiiled),255*abs(ori_complex),data_range=255)
        ssim = compare_ssim(abs(zero_fiiled),abs(ori_complex),data_range=1)
        hfen = compare_hfen(abs(zero_fiiled),abs(ori_complex))
        print("ZF:",'PSNR :', psnr,'SSIM :', ssim,'HFEN :', hfen)

        undersample_kspace = usksp

        self.write_images(255.0*patt,os.path.join(self.args.save_path,'MODL_mask.png'))
        
        print('current undersample method is'+undersample_method, np.sum(mask)/(mask.shape[0]*mask.shape[1]))
        
        #get ori png and degrade png to compare
        self.write_images(np.abs(zero_fiiled), os.path.join(self.args.save_path,'MODL_ZF_undersample_'+undersample_method+undersample_factor+'.png'))
        self.write_images(np.abs(ori_complex), os.path.join(self.args.save_path,'MODL_GT_undersample_'+undersample_method+undersample_factor+'.png'))
        
        #assert False
        x0 = nn.Parameter(torch.Tensor(1,6,256,232).uniform_(-1,1)).cuda()
        x01 = x0.clone()
        
        # set parameters
        step_lr=0.05*0.00003
        
        #number of inner iterations
        n_steps_each = 80
        
        # Noise amounts
        sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                           0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
                           
        start_start = time.time()
        # the outer itertaion loop
        for idx, sigma in enumerate(sigmas):
        
            start_out = time.time()
            lambda_recon = 1./sigma**2
            labels = torch.ones(1, device=x0.device) * idx
            labels = labels.long()

            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            
            print('current {} use sigma = {}'.format(idx,sigma))
            # the inner itertaion loop
            for step in range(n_steps_each):
                start_in = time.time()
                #prior update by ncsn
                
                noise1 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                print(x01.shape)
                grad1 = scorenet(x01, labels).detach()
                
                x0 = x0 + step_size * grad1
                x01 = x0 + noise1

                x0=np.array(x0.cpu().detach(),dtype = np.float32)
                # channel mean
                x_real = (x0.real.squeeze()[0,:,:]+x0.real.squeeze()[2,:,:]+x0.real.squeeze()[4,:,:])/3
                x_imag = (x0.real.squeeze()[1,:,:]+x0.real.squeeze()[3,:,:]+x0.real.squeeze()[5,:,:])/3

                x_complex = x_real + x_imag*1j
                # data consistance
                iterkspace = FT(x_complex,csm)*(1-mask)
                iterkspace = undersample_kspace + iterkspace#*(1-mask)
                x_complex  = tFT(iterkspace,csm)
                
                end_in = time.time()
                print("inner iteration cost time :%.2f s"%(end_in-start_in))
                
                psnr = compare_psnr(255*abs(x_complex),255*abs(ori_complex),data_range=255)
                ssim = compare_ssim(abs(x_complex),abs(ori_complex),data_range=1)
                hfen = compare_hfen(abs(x_complex),abs(ori_complex))
                print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim,'HFEN :', hfen)
                self.write_images(np.abs(x_complex), os.path.join(self.args.save_path,'MODL_rec_undersample_'+undersample_method+undersample_factor+'.png'))
                x_real,x_imag = x_complex.real,x_complex.imag
                x_real,x_imag = x_real[np.newaxis,:,:],x_imag[np.newaxis,:,:]

                x0 = np.stack([x_real,x_imag,x_real,x_imag,x_real,x_imag],1)
                x0 = torch.tensor(x0,dtype=torch.float32).cuda()
            end_out = time.time()
            print("out inner iteration cost time :%.2f s"%(end_out-start_out))
        
        end_end = time.time()
        print("one image reconstruction cost time :%.2f s"%(end_end-start_start))
       
