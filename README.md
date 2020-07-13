# HGGDP

**Paper**: HGGDP: Homotopic Gradients of Generative Density Priors for MR Image Reconstruction

**Authors**: Cong Quan, Jinjie Zhou, Yuanzheng Zhu, Yang Chen, Shanshan Wang, Dong Liang*, Qiegen Liu*

Date : 6/2020  
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

## Graphical representation
 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig6.png" width = "400" height = "450">  </div>
 
Performance exhibition of “multi-view noise” strategy. (a) Training sliced score matching (SSM) loss and validation loss for each iteration. (b) Image quality comparison on the brain dataset at 15% radial sampling: Reconstruction images, error maps (Red) and zoom-in results (Green).

 <div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig7.png"> </div>

Pipeline of sampling from the high-dimensional noisy data distribution with multi-view noise and intermediate samples. (a) Conceptual dia-gram of the sampling on high-dimensional noisy data distribution with multi-view noise. (b) Intermediate samples of annealed Langevin dynamics.


## Reconstruction Results by Various Methods at 85% 2D Random Undersampling.
<div align="center"><img src="https://github.com/yqx7150/HGGDP/blob/master/hggdp_rec/sample/fig11.png"> </div>

Reconstruction comparison on pseudo radial sampling at acceleration factor 6.7 . Top: Reference, reconstruction by DLMRI, PANO, FDLCP; Bottom: Reconstruction by NLR-CS, DC-CNN, EDAEPRec, HGGDPRec. Green and red boxes illustrate the zoom in results and error maps, respectively.

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
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8782831)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MEDAEP)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors  
[<font size=5>**[Paper]**</font>](https://cardiacmr.hms.harvard.edu/files/cardiacmr/files/liu2019.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDAEPRec)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)

  * High-dimensional Embedding Network Derived Prior for Compressive Sensing MRI Reconstruction  
 [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1361841520300815?via%3Dihub)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/EDMSPRec)
 
  * Denoising Auto-encoding Priors in Undecimated Wavelet Domain for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/ftp/arxiv/papers/1909/1909.01108.pdf)  [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WDAEPRec)

  * Complex-valued MRI data from SIAT--test31 [<font size=5>**[Data]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/test_data_31)

  * Complex-valued MRI data from SIAT--SIAT_MRIdata200 [<font size=5>**[Data]**</font>](https://github.com/yqx7150/SIAT_MRIdata200)
 
  * Learning Multi-Denoising Autoencoding Priors for Image Super-Resolution  
[<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/pii/S1047320318302700)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/MDAEP-SR)

  * REDAEP: Robust and Enhanced Denoising Autoencoding Prior for Sparse-View CT Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/document/9076295)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/REDAEP)
