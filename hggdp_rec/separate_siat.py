# The Code is created based on the method described in the following paper:
# HGGDP: Homotopic Gradients of Generative Density Priors for MR Image Reconstruction
# Authors: Cong Quan, Jinjie Zhou, Yuanzheng Zhu, Yang Chen, Shanshan Wang, Dong Liang*, Qiegen Liu*
# Date : 6/2020
# Version : 1.0
# The code and the algorithm are for non-comercial use only.
# Copyright 2020, Department of Electronic Information Engineering, Nanchang University.

import sys,argparse

import hggdp.main_siat as hggdp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='hggdp', help='The model to use as a prior')

    parsed,sys.argv = parser.parse_known_args()
    sys.argv.insert(0,parsed.model)
    # main function
    hggdp.main()
