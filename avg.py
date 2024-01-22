import cv2
import numpy as np
import glob



for i in range(1,35):
    attns = []
    for j in range(0,8):
        path = f'/root/workplace/supple/test34_attn_old/360_glass_{str(i).zfill(2)}/attn/block_7/head_{j}.png'
        attns.append(cv2.imread(path))
        # if i == 0: 
        #     attn = cv2.imread(path[i])
        # else:
        #     attn += cv2.imread(path[i])
    attns = np.array(attns)
    print(attns.shape)
    attns = np.mean(attns, 0)
    print(attns.shape)
    attns = np.mean(attns, 2)
    attns = (attns/np.max(attns))*256
    cv2.imwrite(f'/root/workplace/supple/test34_attn_old/360_glass_{str(i).zfill(2)}/attn/block_7/avg.png', attns)


# path = glob.glob('/root/workplace/abl2_res/df_prop_360_selected/0500/*.png')

# attns = []
# for i in range(len(path)):
#     attns.append(cv2.imread(path[i]))
#     # if i == 0: 
#     #     attn = cv2.imread(path[i])
#     # else:
#     #     attn += cv2.imread(path[i])
# attns = np.array(attns)
# print(attns.shape)
# attns = np.mean(attns, 0)
# print(attns.shape)
# attns = np.mean(attns, 2)
# attns = (attns/np.max(attns))*256
# cv2.imwrite('/root/workplace/abl2_res/df_prop_360_selected/0500/avg.png',attns)