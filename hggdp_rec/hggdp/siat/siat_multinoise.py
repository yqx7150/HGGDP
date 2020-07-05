import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import math
from hggdp.models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from hggdp.siat.compare_hfen import compare_hfen
from skimage.measure import compare_psnr,compare_ssim
import glob
import h5py
import time

savepath = './result_T80/'

__all__ = ['SIAT_MULTI_NOISE']
def show(image):
    plt.figure(1)
    plt.imshow(np.abs(image),cmap='gray')
    plt.show()
def write_Data(result_all,undersample_method,undersample_factor,i):
    with open(os.path.join(savepath,"psnr_"+undersample_method+undersample_factor+".txt".format(undersample_factor)),"w+") as f:
        #print(len(result_all))
        for i in range(len(result_all)):
            f.writelines('current image {} PSNR : '.format(i) + str(result_all[i][0]) + \
            "    SSIM : " + str(result_all[i][1]) + "    HFEN : " + str(result_all[i][2]))
            f.write('\n')
class SIAT_MULTI_NOISE():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        
    def test(self):

        # make a network and load network weight to use later
        states = torch.load(os.path.join(self.args.log, 'checkpoint_100000.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet, device_ids=[0])
        scorenet.load_state_dict(states[0])
        scorenet.eval()
        # prepare all test data and undersample masks
        files_list = glob.glob('./SIAT_test_image31/*.mat')
        files_list.sort()

        for undersample_method in ['radial','random','cart']:#'radial','random','cart'
            for undersample_factor in ['030','025','015','010']:#'030','025','015','010','007'
                result_all = np.zeros([32,3])
                for i,file_path in enumerate(files_list):
                
                    m = loadmat(file_path)
                    data=(m["Img"])
                    data=np.array(data)
                    data = data/np.max(np.abs(data))
                    kdata=np.fft.fft2(data)
                    print('value max min :',np.max(data),np.min(data))
                    mask=loadmat("./SIAT/mask_"+undersample_method+"_"+undersample_factor+".mat")["mask_"+undersample_method+"_"+undersample_factor]
                    mask = np.fft.fftshift(mask)
                    cv2.imwrite(os.path.join(self.args.image_folder, 'mask_.png' ),(mask*255).astype(np.uint8))
                    
                    print(sum(sum(mask))/(256*256))

                    ksample=np.multiply(mask,kdata)

                    sdata=np.fft.ifft2(ksample)

                    self.write_images(255*np.abs(sdata), os.path.join(savepath,'img_{}_ZF_undersample_'.format(i)+undersample_method+undersample_factor+'.png'))
                    
                    sdata=np.stack((sdata.real,sdata.imag))[np.newaxis,:,:,:]
                    
                    self.write_images(255*np.abs(data), os.path.join(savepath,'img_{}_GT_undersample_'.format(i)+undersample_method+undersample_factor+'.png'))

                    x0 = nn.Parameter(torch.Tensor(1,6,256,256).uniform_(-1,1)).cuda()
                    x01 = x0
                    x02 = x0
                    x03 = x0

                    step_lr=0.05*0.00003

                    # Noise amounts
                    sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                                       0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
                    n_steps_each = 80
                    max_psnr = 0
                    max_ssim = 0
                    min_hfen = 100
                    start_start = time.time()
                    for idx, sigma in enumerate(sigmas):
                        start_out = time.time()
                        print(idx)
                        lambda_recon = 1./sigma**2
                        labels = torch.ones(1, device=x0.device) * idx
                        labels = labels.long()

                        step_size = step_lr * (sigma / sigmas[-1]) ** 2
                        
                        print('sigma = {}'.format(sigma))
                        for step in range(n_steps_each):
                            start_in = time.time()
                            noise1 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                            noise2 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                            noise3 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                            grad1 = scorenet(x01, labels).detach()
                            grad2 = scorenet(x02, labels).detach()
                            grad3 = scorenet(x03, labels).detach()

                            x0 = x0 + step_size * (grad1 + grad2 + grad3)/3.0 

                            x01 = x0 + noise1
                            x02 = x0 + noise2
                            x03 = x0 + noise3

                            x0=np.array(x0.cpu().detach(),dtype = np.float32)

                            x_real = (x0.real.squeeze()[0,:,:]+x0.real.squeeze()[2,:,:]+x0.real.squeeze()[4,:,:])/3
                            x_imag = (x0.real.squeeze()[1,:,:]+x0.real.squeeze()[3,:,:]+x0.real.squeeze()[5,:,:])/3

                            x_complex = x_real + x_imag*1j

                            kx=np.fft.fft2(x_complex)
                            kx[mask==1]=ksample[mask==1]
                            x_complex = np.fft.ifft2(kx)
                            end_in = time.time()
                            print("内循环运行时间:%.2f秒"%(end_in-start_in))
                            psnr = compare_psnr(255*abs(x_complex),255*abs(data),data_range=255)
                            ssim = compare_ssim(abs(x_complex),abs(data),data_range=1)
                            hfen = compare_hfen(abs(x_complex),abs(data))

                            if max_psnr < psnr :
                                result_all[i,0] = psnr
                                max_psnr = psnr
                                result_all[31,0] = sum(result_all[:31,0])/31

                                savemat(os.path.join(savepath,'img_{}_Rec_'.format(i)+undersample_method+undersample_factor+'.mat'),{'data':np.array(x_complex,dtype=np.complex),'ZF':sdata})
                            if max_ssim < ssim :
                                result_all[i,1] = ssim
                                max_ssim = ssim
                                result_all[31,1] = sum(result_all[:31,1])/31

                            if min_hfen > hfen :
                                result_all[i,2] = hfen
                                min_hfen = hfen
                                result_all[31,2] = sum(result_all[:31,2])/31

                            write_Data(result_all,undersample_method,undersample_factor,i)

                            print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim,'HFEN :', hfen)

                            x_real,x_imag = x_complex.real,x_complex.imag
                            x_real,x_imag = x_real[np.newaxis,:,:],x_imag[np.newaxis,:,:]

                            x0 = np.stack([x_real,x_imag,x_real,x_imag,x_real,x_imag],1)
                            x0 = torch.tensor(x0,dtype=torch.float32).cuda()
                        end_out = time.time()
                        print("外循环运行时间:%.2f秒"%(end_out-start_out))
                    
                    end_end = time.time()
                    print("一张图循环运行时间:%.2f秒"%(end_end-start_start))
    def write_images(self, x,image_save_path):
        
        x = np.array(x,dtype=np.uint8)
        cv2.imwrite(image_save_path, x)
        
