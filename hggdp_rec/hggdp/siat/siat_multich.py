import os
import cv2
import glob
import h5py
import time
import math
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.io import loadmat,savemat
from skimage.measure import compare_psnr,compare_ssim
from hggdp.siat.compare_hfen import compare_hfen
from hggdp.models.cond_refinenet_dilated import CondRefineNetDilated

__all__ = ['SIAT_MULTICHANNEL']

class SIAT_MULTICHANNEL():
    def __init__(self, args, config):
        self.args = args
        self.config = config        
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        assert os.path.exists(self.args.load_path),'load file path is not exists,please recheck your path'
            
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
        
        undersample_method = 'radial'
        undersample_factor = '030'
        # use for getting degrade img and psnr,ssim,hfen in iteration
        
        ori_complex = loadmat(self.args.load_path)["Img"]
        ori_complex = ori_complex/np.max(np.abs(ori_complex))
        
        kspace=np.fft.fft2(ori_complex)
        
        mask=loadmat("./mask/mask_"+undersample_method+"_"+undersample_factor+".mat")["mask_"+undersample_method+"_"+undersample_factor]
        mask = np.fft.fftshift(mask)
        self.write_images(255.0*mask,os.path.join(self.args.save_path,'mask.png'))
        
        print('current undersample method is'+undersample_method, np.sum(mask)/(256*256))
        
        undersample_kspace=np.multiply(mask,kspace)
        
        zero_fiiled=np.fft.ifft2(undersample_kspace)
        #get ori png and degrade png to compare
        self.write_images(np.abs(zero_fiiled), os.path.join(self.args.save_path,'img_ZF_undersample_'+undersample_method+undersample_factor+'.png'))
        self.write_images(np.abs(ori_complex), os.path.join(self.args.save_path,'img_GT_undersample_'+undersample_method+undersample_factor+'.png'))

        x0 = nn.Parameter(torch.Tensor(1,6,256,256).uniform_(-1,1)).cuda()
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
                grad1 = scorenet(x01, labels).detach()
                
                x0 = x0 + step_size * grad1
                x01 = x0 + noise1

                x0=np.array(x0.cpu().detach(),dtype = np.float32)
                # channel mean
                x_real = (x0.real.squeeze()[0,:,:]+x0.real.squeeze()[2,:,:]+x0.real.squeeze()[4,:,:])/3
                x_imag = (x0.real.squeeze()[1,:,:]+x0.real.squeeze()[3,:,:]+x0.real.squeeze()[5,:,:])/3

                x_complex = x_real + x_imag*1j
                # data consistance
                iterkspace = np.fft.fft2(x_complex)
                iterkspace = undersample_kspace + iterkspace*(1-mask)
                x_complex  = np.fft.ifft2(iterkspace)
                
                end_in = time.time()
                print("inner iteration cost time :%.2f s"%(end_in-start_in))
                
                psnr = compare_psnr(255*abs(x_complex),255*abs(ori_complex),data_range=255)
                ssim = compare_ssim(abs(x_complex),abs(ori_complex),data_range=1)
                hfen = compare_hfen(abs(x_complex),abs(ori_complex))
                print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim,'HFEN :', hfen)
                self.write_images(np.abs(x_complex), os.path.join(self.args.save_path,'img_rec_undersample_'+undersample_method+undersample_factor+'.png'))
                x_real,x_imag = x_complex.real,x_complex.imag
                x_real,x_imag = x_real[np.newaxis,:,:],x_imag[np.newaxis,:,:]

                x0 = np.stack([x_real,x_imag,x_real,x_imag,x_real,x_imag],1)
                x0 = torch.tensor(x0,dtype=torch.float32).cuda()
            end_out = time.time()
            print("out inner iteration cost time :%.2f s"%(end_out-start_out))
        
        end_end = time.time()
        print("one image reconstruction cost time :%.2f s"%(end_end-start_start))
       
