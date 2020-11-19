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
from torchvision import transforms
from scipy.io import loadmat,savemat
from scipy import misc
from skimage.measure import compare_psnr,compare_ssim
from hggdp.siat.compare_hfen import compare_hfen
from hggdp.models.cond_refinenet_dilated import CondRefineNetDilated
from hggdp.siat.US_pattern import US_pattern
__all__ = ['SIAT_MULTICHANNEL_DDP']
def FT (x):   
    #inp: [nx, ny]
    #out: [nx, ny, ns]
    return np.fft.fftshift(np.fft.fft2(sensmaps*np.tile(x[:,:,np.newaxis],[1,1,sensmaps.shape[2]]),axes=(0,1)),axes=(0,1))
    # np.fft.fftshift jiang dipin zhuanyi dao tuxiang zhongxin
    # np.fft.fft2 erwei fuliye

def tFT (x):
    #inp: [nx, ny, ns]
    #out: [nx, ny]
    temp = np.fft.ifft2(np.fft.ifftshift( x , axes=(0,1) ),  axes=(0,1)  )
    return np.sum( temp*np.conjugate(sensmaps) , axis=2)  / np.sum(sensmaps*np.conjugate(sensmaps),axis=2)


def UFT(x, uspat):
    #inp: [nx, ny], [nx, ny]
    #out: [nx, ny, ns] 
    return np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])*FT(x)

def tUFT(x, uspat):
    #inp: [nx, ny], [nx, ny]
    #out: [nx, ny]
    tmp1 = np.tile(uspat[:,:,np.newaxis],[1,1,sensmaps.shape[2]])
    return tFT(tmp1*x)

#load the test image and the coil maps
#-----------------------------------------------------        
#unfortunately, due to complications in saving complex valued data, we save
#and load the complex and real parts seperately
f = h5py.File('./DDP_share_data/acq_im_real.h5', 'r')
kspr = np.array((f['DS1']))
f = h5py.File('./DDP_share_data/acq_im_imag.h5', 'r')
kspi = np.array((f['DS1']))
ksp = np.rot90(np.transpose(kspr+1j*kspi),3)
#get the k-space data
ksp = np.fft.ifftn(np.fft.fftshift(np.fft.fftn(ksp,axes=[0,1]),axes=[0,1]),axes=[0,1])


#again we save and load the complex and real parts seperately for coil maps
f = h5py.File('./DDP_share_data/acq_coilmaps_espirit_real.h5', 'r')
espsr = np.array((f['DS1']))
f = h5py.File('./DDP_share_data/acq_coilmaps_espirit_imag.h5', 'r')
espsi = np.array((f['DS1']))

esps= np.rot90(np.transpose(espsr+1j*espsi),3)
sensmaps = esps.copy()

#rotate images for canonical orientation
sensmaps=np.rot90(np.rot90(sensmaps))
ksp=np.rot90(np.rot90(ksp))
#normalize the espirit coil maps
sensmaps=sensmaps/np.tile(np.sum(sensmaps*np.conjugate(sensmaps),axis=2)[:,:,np.newaxis],[1, 1, sensmaps.shape[2]])

#load the undersampling pattern  original mask
#patt = pickle.load(open('./DDP_share_data/acq_im_us_patt_R2','rb'))
#make the undersampled kspace
#usksp = ksp * np.tile(patt[:,:,np.newaxis],[1, 1, ksp.shape[2]])


orim = tFT(ksp) # the fully sampled image 

# to make the mr image divisible by four
orim_ = np.zeros([216,256],dtype=np.complex128)
orim_[5:210,:] = orim
orim = orim_

sensmaps_ = np.ones([216,256,15],dtype=np.complex128)
sensmaps_[5:210,:,:] = sensmaps
sensmaps = sensmaps_

ksp = FT(orim)


#load the undersampling pattern
USp = US_pattern();
patt = USp.generate_opt_US_pattern_1D([orim.shape[0], orim.shape[1]], R=3, max_iter=100, no_of_training_profs=15)
#misc.imsave('DDP_usp_patt_R3_%s.png'%str((np.sum(patt))/orim.shape[0]/orim.shape[1]),np.abs(patt))
#make the undersampled kspace
usksp = ksp * np.tile(patt[:,:,np.newaxis],[1, 1, ksp.shape[2]])

# normalize the kspace
tmp = tFT(usksp)
usksp=usksp/np.percentile(  np.abs(tmp).flatten()   ,99)

#assert False
def calc_rmse(rec,imorig):
     return 100*np.sqrt(np.sum(np.square(np.abs(rec) - np.abs(imorig))) / np.sum(np.square(np.abs(imorig))) )
class SIAT_MULTICHANNEL_DDP():
    def __init__(self, args, config):
        self.args = args
        self.config = config        
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
            
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
        
        undersample_method = 'cart'
        undersample_factor = str((np.sum(patt))/orim.shape[0]/orim.shape[1])
        # use for getting degrade img and psnr,ssim,hfen in iteration
        
        ori_complex = orim
        
        #ori_complex = loadmat(self.args.load_path)["Img"]
        ori_complex = ori_complex/np.max(np.abs(ori_complex))
        
        ksp = FT(ori_complex)
        print(ksp.shape)
        mask = patt
        #undersample multi coil kspace
        usksp = ksp * np.tile(patt[:,:,np.newaxis],[1, 1, ksp.shape[2]])
        
        zero_fiiled = tFT(usksp)
        # undersample_kspace
        undersample_kspace = usksp

        self.write_images(255.0*patt,os.path.join(self.args.save_path,'mask_DDP.png'))
        
        print('current undersample method is'+undersample_method, np.sum(mask)/(mask.shape[0]*mask.shape[1]))
        
        #undersample_kspace=np.multiply(mask,kspace)

        #get ori png and degrade png to compare
        self.write_images(np.abs(zero_fiiled), os.path.join(self.args.save_path,'DDP_ZF_undersample_'+undersample_method+undersample_factor+'.png'))
        self.write_images(np.abs(ori_complex), os.path.join(self.args.save_path,'DDP_GT_undersample_'+undersample_method+undersample_factor+'.png'))
        #assert False
        x0 = nn.Parameter(torch.Tensor(1,6,216,256).uniform_(-1,1)).cuda()
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
                iterkspace = UFT(x_complex,(1-mask))
                iterkspace = undersample_kspace + iterkspace#*(1-mask)
                x_complex  = tFT(iterkspace)
                
                end_in = time.time()
                print("inner iteration cost time :%.2f s"%(end_in-start_in))
                rmse = calc_rmse(x_complex,ori_complex)
                psnr = compare_psnr(255*abs(x_complex),255*abs(ori_complex),data_range=255)
                ssim = compare_ssim(abs(x_complex),abs(ori_complex),data_range=1)
                hfen = compare_hfen(abs(x_complex),abs(ori_complex))
                print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim,'HFEN :', hfen,'RMSE : ',rmse)
                self.write_images(np.abs(x_complex), os.path.join(self.args.save_path,'DDP_rec_undersample_'+undersample_method+undersample_factor+'.png'))
                x_real,x_imag = x_complex.real,x_complex.imag
                x_real,x_imag = x_real[np.newaxis,:,:],x_imag[np.newaxis,:,:]

                x0 = np.stack([x_real,x_imag,x_real,x_imag,x_real,x_imag],1)
                x0 = torch.tensor(x0,dtype=torch.float32).cuda()
            end_out = time.time()
            print("out inner iteration cost time :%.2f s"%(end_out-start_out))
        
        end_end = time.time()
        print("one image reconstruction cost time :%.2f s"%(end_end-start_start))
       
