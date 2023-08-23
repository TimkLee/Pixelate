from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv




img = cv.imread(cv.samples.findFile("../assets/lena_std.tif"))

cv.imshow("Display window", img)
cv.waitKey(0)


img_ver = img.shape[0]
img_hor = img.shape[1]
img_chan = img.shape[2]


# Pixelate window
ver_stride = 10
hor_stride = 20


pix_init_ver = np.ceil(img_ver/ver_stride).astype(int)
pix_init_hor = np.ceil(img_hor/hor_stride).astype(int)


pix_img = (np.ones((pix_init_ver,pix_init_hor,img_chan))*255).astype(img.dtype)
pix_img[:,:,:] = img[0:img_ver:ver_stride,0:img_hor:hor_stride,:]


mid_img = np.repeat(pix_img, hor_stride, axis=1)
out_img = np.repeat(mid_img, ver_stride,axis=0)[:img_ver,:img_hor,:img_chan]


cv.imwrite("../assets/pixelated.png", out_img)