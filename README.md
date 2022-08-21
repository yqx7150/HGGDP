# HGGDP

**Paper**: HGGDP: Homotopic Gradients of Generative Density Priors for MR Image Reconstruction

**Authors**: Cong Quan, Jinjie Zhou, Yuanzheng Zhu, Yang Chen, Shanshan Wang, Dong Liang*, Qiegen Liu*   

IEEE Transactions on Medical Imaging, https://ieeexplore.ieee.org/abstract/document/9435335   

Date : May-22-2021  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2020, Department of Electronic Information Engineering, Nanchang University.  

Deep learning, particularly generative model, has demonstrated tremendous potential to significantly speed up image reconstruction with reduced measurements recently. In this work, by taking advantage of the denoising score matching, deep gradients of generative density priors (HGGDP) are proposed for MRI reconstruction. More precisely, to tackle the low-dimensional manifold and low data density region issues in generative density prior, we estimate the target gradients in higher-dimensional space. We train a more powerful noise conditional score network by forming higher-dimensional tensor as the network input at the training phase. More artificial noise is also injected in the embedding space. At the reconstruction stage, a homotopy method is employed to pursue the density prior, such as to boost the reconstruction performance. Experiment results imply the remarkable performance of HGGDP in terms of high reconstruction accuracy; only 10% of the k-space data can still generate images of high quality as effectively as standard MRI reconstruction with the fully sampled data.

## Training
```bash
python3 separate_siat.py --exe SIAT_TRAIN --config siat_config.yml --checkpoint your save path
```

## Test
```bash
python3 separate_siat.py --exe SIAT_MULTICHANNEL --config siat_config.yml --model hggdp --test
```
## Compare_MoDL
```bash
python3 separate_siat.py --exe SIAT_MULTICHANNEL_MODL --config siat_config.yml --model hggdp --test
```
## Compare_DDP
```bash
python3 separate_siat.py --exe SIAT_MULTICHANNEL_DDP --config siat_config.yml --model hggdp --test
```
In order to verify the fairness of the experiment, in the MoDL experiment comparison, we chose the test data, coil sensitivity maps and undersampling mask shared by Aggarwal et.al. Orignal MoDL available code[<font size=5>**[Code]**</font>](https://github.com/hkaggarwal/modl)  
In the DDP experiment comparison, we chose the test data, coil sensitivity maps, undersampleing patterns and undersampling mask shared by Tezcan et.al. Orignal DDP available code[<font size=5>**[Code]**</font>](https://github.com/kctezcan/ddp_recon)  

## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig6.png" width = "400" height = "450">  </div>
 
Performance exhibition of “multi-view noise” strategy. (a) Training sliced score matching (SSM) loss and validation loss for each iteration. (b) Image quality comparison on the brain dataset at 15% radial sampling: Reconstruction images, error maps (Red) and zoom-in results (Green).

 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>

Pipeline of sampling from the high-dimensional noisy data distribution with multi-view noise and intermediate samples. (a) Conceptual dia-gram of the sampling on high-dimensional noisy data distribution with multi-view noise. (b) Intermediate samples of annealed Langevin dynamics.


## Reconstruction Results by Various Methods at 85% 2D Random Undersampling.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig11.png"> </div>

Reconstruction comparison on pseudo radial sampling at acceleration factor 6.7 . Top: Reference, reconstruction by DLMRI, PANO, FDLCP; Bottom: Reconstruction by NLR-CS, DC-CNN, EDAEPRec, HGGDPRec. Green and red boxes illustrate the zoom in results and error maps, respectively.

## Reconstruction Results by Various Methods at various 1D Cartesian undersampling percentages.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/compare_DDP.PNG"> </div>

Complex-valued reconstruction results on brain image at various 1D Cartesian undersampling percentages (R=2, 3). From left to right: Ground-truth, various 1D Cartesian undersampling masks, reconstruction by Zero-Filled, DDP and HGGDPRec.

## Reconstruction Results by Various Methods at 6-fold 2D Random undersampling mask.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/Compare_MoDL.png"> </div>

Complex-valued reconstruction results on brain image at 16.7% 2D random sampling. From left to right: Ground-truth, 6-fold 2D random undersample mask, reconstruction by Zero-Filled, MoDL and HGGDPRec.

## Table
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/table1.png"> </div>
RECONSTRUCTION PSNR, SSIM AND HFEN VALUES OF THREE TEST IMAGES AT VARIOUS SAMPLING TRAJECTORIES AND UNDERSAMPLING PER-CENTAGES. 

## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Baidu Drive](https://pan.baidu.com/s/1QIjU8kRUQ3i2pT6PvROlKQ). 
key number is "awn0" 

## Test Data
In file './test_data_31', 31 complex-valued MRI data with size of 256x256 were acquired by using a 3D fast-spin-echo (FSE) sequence with T2 weighting from a 3.0T whole body MR system (SIEMENS MAGNETOM TrioTim).

## Other Related Projects
  * Multi-Channel and Multi-Model-Based Autoencoding Prior for Grayscale Image Restoration  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide) [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * High-dimensional Embedding Network Derived Prior for Compressive Sensing MRI Reconstruction  
 [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300815?via%3Dihub)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDMSPRec)
 
  * Denoising Auto-encoding Priors in Undecimated Wavelet Domain for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S0925231221000990) [<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1909/1909.01108.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WDAEPRec)

  * Complex-valued MRI data from SIAT--test31 [<font size=5>**[Data]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/test_data_31)
  * More explanations with regard to the MoDL test datasets, we use some data from the test dataset in "dataset.hdf5" file, where the image slice numbers are 40,48,56,64,72,80,88,96,104,112(https://drive.google.com/file/d/1qp-l9kJbRfQU1W5wCjOQZi7I3T6jwA37/view)
  * DDP Method Link [<font size=5>**[DDP Code]**</font>](https://github.com/kctezcan/ddp_recon)
  * MoDL Method Link [<font size=5>**[MoDL code]**</font>](https://github.com/hkaggarwal/modl)
  * Complex-valued MRI data from SIAT--SIAT_MRIdata200 [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT_MRIdata200)  
  * Complex-valued MRI data from SIAT--SIAT_MRIdata500-singlecoil [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT500data-singlecoil)   
  * Complex-valued MRI data from SIAT--SIAT_MRIdata500-12coils [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT500data-12coils)    
 
  * Learning Multi-Denoising Autoencoding Priors for Image Super-Resolution  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320318302700)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MDAEP-SR)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Iterative Reconstruction for Low-Dose CT using Deep Gradient Priors of Generative Model  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9703672)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EASEL)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)

  * Universal Generative Modeling for Calibration-free Parallel MR Imaging  
[<font size=5>**[Paper]**</font>](https://biomedicalimaging.org/2022/)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/UGM-PI)   [<font size=5>**[Poster]**</font>](https://github.com/yqx7150/UGM-PI/blob/main/paper%20%23160-Poster.pdf)

* Progressive Colorization via Interative Generative Models  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9258392)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Joint Intensity-Gradient Guided Generative Modeling for Colorization
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2012.14130)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/JGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  [<font size=5>**[数学图像联盟会议交流PPT]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

* Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)  
