# The Code is created based on the method described in the following paper:
# Highly Undersampled Magnetic Resonance Imaging Reconstruction using Autoencoding Priors, Submitted to Magnetic Resonance in Medicine, 2018.
# Author: Q. Liu, Q. Yang, H. Cheng, S. Wang, M. Zhang, D. Liang.
# Date : 7/2020
# Version : 1.0
# The code and the algorithm are for non-comercial use only.
# Copyright 2018, Department of Electronic Information Engineering, Nanchang University.
# Paul C. Lauterbur Research Center for Biomedical Imaging, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen 518055, China
# Medical AI research center, Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences, Shenzhen 518055, China

# EDAEPRec - Enhanced Denoising Autoencoder Prior for Reconstruction

import sys,argparse

import hggdp.main_siat as hggdp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hggdp', help='The model to use as a prior')

    parsed,sys.argv = parser.parse_known_args()
    sys.argv.insert(0,parsed.model)
    # main function
    hggdp.main()
