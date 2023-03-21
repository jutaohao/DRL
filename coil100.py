import os

import torch
from PIL import Image
import numpy as np
import torchvision.utils as vutils




# img=image.read_image(path)
# print(type(img))
# print(img.shape)
filepath=r"C:/Users/haojutao/Downloads/coil-100/"
pathDir = os.listdir(filepath)
import cv2
import torchvision.transforms.functional as F
import torchvision.transforms as T
lt=[]
for allDir in pathDir:
    # print(allDir)
    # if allDir.endswith("_gt"):
    for i in range(6,7):
        if allDir.startswith("obj%s"%i):
            print(allDir)
            image = cv2.imread(filepath+ allDir)
            img1 = F.to_tensor(image)
            if len(lt)<8:
                lt.append(img1)
k=0
skip = 0
for allDir in pathDir:
    # print(allDir)
    # if allDir.endswith("_gt"):
    skip +=1
    if skip%5!=0:
        continue
    for i in range(32,33):
        if allDir.startswith("obj%s"%i):
            print(allDir)
            image = cv2.imread(filepath+ allDir)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img1 = F.to_tensor(image)
            if k<4:
                lt[2*k]=img1
                k=k+1

k=0
for allDir in pathDir:
    # print(allDir)
    # if allDir.endswith("_gt"):
    for i in range(24,25):
        if allDir.startswith("obj%s"%i):
            print(allDir)
            image = cv2.imread(filepath+ allDir)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img1 = F.to_tensor(image)
            if k<4:
                lt[2 * (k-1) +1] = img1
                k+=1
vutils.save_image(lt, 'coil2.png', normalize=True)
