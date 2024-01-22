import cv2
import numpy as np
import glob


for path in glob.glob('/root/workplace/supple2/in/*.png'):
    filename = path.split('/')[-1]
    img = cv2.imread(path)
    img[:, 256:] = 0
    cv2.imwrite(f'/root/workplace/supple2/out/{filename}', img)