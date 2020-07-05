import cv2
import numpy as np
from scipy import io
from scipy.misc import imread,imsave
from skimage.measure import compare_psnr,compare_ssim



# cal hfen
def compare_hfen(ori,rec):
    operation = np.array(io.loadmat("./hggdp/siat/loglvbo.mat")['h1'],dtype=np.float32)
    ori = cv2.filter2D(ori.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    rec = cv2.filter2D(rec.astype('float32'), -1, operation,borderType=cv2.BORDER_CONSTANT)
    hfen = np.linalg.norm(ori-rec, ord = 'fro')
    return hfen

if __name__ == "__main__":

    rec = imread('network_output.png')
    ori = imread('ori.png')
    print(np.max(rec))
    psnr = compare_psnr(ori,rec)
    ssim = compare_ssim(ori/255,rec/255)
    hfen = compare_hfen(ori/255,rec/255,operation)
    print(psnr,ssim,hfen)
