import cv2
import numpy as np


# for i in range(1,35):
#     inp = cv2.imread(f'/root/workplace/supple/test34_attn_old/360_glass_{str(i).zfill(2)}_IN.png')
#     out = cv2.imread(f'/root/workplace/supple/test34_attn_old/360_glass_{str(i).zfill(2)}_OUT.png')

#     res = (inp - out)**2
#     res = np.mean(res, axis = 2)

#     cv2.imwrite(f'/root/workplace/supple/test34_attn_old/360_glass_{str(i).zfill(2)}_RES.png',res)

inp = cv2.imread('/root/workplace/supple2/tmp/360_glass_09_IN.png')
out = cv2.imread('/root/workplace/supple2/tmp/360_glass_09_OUT.png')

res = (inp - out)**2
res = np.mean(res, axis = 2)

cv2.imwrite('/root/workplace/supple2/tmp/360_glass_09_RES.png',res)