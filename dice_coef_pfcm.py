# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:51:12 2021

@author: eduardo
"""

import os
import numpy as np
from PIL import Image
def dice(pred, true):
    overlap = np.logical_and(true, pred)
    dice = np.sum(overlap)*2 / (np.sum(true)+ np.sum(pred))
    return dice
dir_images = "test/images_test_pfcm"
dir_expected = "test/images_test_pfcm_expected"
f = open (dir_expected + '/evaluation_dice_pfcm.txt','wt')
images = os.listdir(dir_images)
for img in images:
    imgpred = Image.open(dir_images + "/" + img)
    imgpred = np.array(imgpred)
    imgpred = imgpred / 65533
    name = img.replace("_prediction.png", "_mask.png")
    imgtrue = Image.open(dir_expected + "/" + name)
    imgtrue = np.array(imgtrue)
    imgtrue = imgtrue / 65533
    result = 'image ' + img + ' ' + str(dice(imgpred, imgtrue))
    print(result)
    f.write(result + "\n")
f.close()