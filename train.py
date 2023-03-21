"""
TRAIN Dlrmodel                                 \
"""

##
# LIBRARIES
import os
import random
import shutil
import sys

import numpy as np
from cv2 import cv2

from options import Options
from lib.data.dataloader import load_data
from lib.models import load_model

##
def train_cifar10():
    """ Training
    """
    opt = Options().parse()
    opt.dataset = "cifar10"
    t = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8,'truck': 9}
    opt.nc = 3
    for key in t:

        opt.normal_class = key
        data = load_data(opt)
        # for w in [1,10,20,30,40,50,60,70,80,90,100]:
                # opt.abnomal_class = abn_class
            # opt.w_adv = w
            # opt.nz = z

        model = load_model(opt, data)
        model.train()

def train_MNIST():
    """ Training
    """
    opt = Options().parse()
    opt.dataset = "mnist"
    opt.nc = 1
    t = {'0': 0,'1':1,'2': 2,'3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9}
    for key in t:
        # for i in range(0,10):
            opt.normal_class=t[key]
            data = load_data(opt)
            model = load_model(opt, data)
            model.train()
def train_GTSRB():
    """ Training
    """
    opt = Options().parse()
    opt.dataset = "GTSRB"
    opt.nc = 3

    opt.normal_class="stop_sign"
    data = load_data(opt)
    model = load_model(opt, data)
    model.train()

def train_FASION():
    opt = Options().parse()
    opt.dataset = "FashionMNIST"
    opt.nc = 1
    t= {'T-shirt/top': 0, 'Trouser': 1, 'Pullover': 2, 'Dress': 3, 'Coat': 4, 'Sandal': 5, 'Shirt': 6, 'Sneaker': 7,'Bag': 8, 'Ankle boot': 9}
    for key in t:
        # for i in range(0, 10):
            opt.normal_class = t[key]
            data = load_data(opt)
            model = load_model(opt, data)
            model.train()
def train_COIL():
    opt = Options().parse()
    opt.dataset = "coil100"
    opt.nc = 3
    filepath = r"C:/Users/haojutao/Downloads/coil-100/"

    pathDir = os.listdir(filepath)
    for obj in range(1,101):
        train = r"data/COIL100/train/0.normal/"
        valid_0 = r"data/COIL100/valid/0.normal/"
        valid_1 = r"data/COIL100/valid/1.abnormal/"
        test_0 = r"data/COIL100/test/0.normal/"
        test_1 = r"data/COIL100/test/1.abnormal/"
        emplist = [train,valid_0,valid_1,test_0,test_1]
        for ddir in emplist:
            ff = os.listdir(ddir)
            for f in ff:
                os.remove(ddir+f)
        files = []
        for allDir in pathDir:
            if allDir.startswith("obj%s__"%obj):
                files.append(filepath+allDir)
        print(len(files))
        nrm_trn_val_file = random.sample(files, int(len(files) * 0.8))
        print(len(nrm_trn_val_file))
        nrm_tst_file = [i for i in files if not i in nrm_trn_val_file]
        #测试集正常
        for f in nrm_tst_file:
            shutil.copy(f, test_0)
        print(len(nrm_tst_file))
        #测试集异常
        test_abn_files = []
        for allDir in pathDir:
            if (not allDir.startswith("obj%s__"%obj)) and allDir.endswith("png"):
                test_abn_files.append(filepath+allDir)
        abn_test_files = random.sample(test_abn_files, int(len(test_abn_files) * 0.5))
        for f in abn_test_files:
            shutil.copy(f, test_1)

        nrm_trn_file = nrm_trn_val_file[:int(len(nrm_trn_val_file) * 0.9)]
        #训练集数据
        for f in nrm_trn_file:
            shutil.copy(f, train)
        nrm_val_file = [i for i in nrm_trn_val_file if not i in nrm_trn_file]
        #验证集正常数据
        for f in nrm_val_file:
            shutil.copy(f, valid_0)
        for abnum in range(0,len(nrm_val_file)):
            print("-------------------")
            abn_valp_file = random.sample(nrm_val_file, int(len(nrm_val_file) * 0.8))
            img = cv2.imread(abn_valp_file[0])
            # for f in abn_valp_file:
            #     print(f)
            #     img2 = cv2.imread(f)
            #     img += img2
            # cv2.imwrite(valid_1+"%d.png"%abnum, img)
            for i in range(0, 10):
                # 随机中心点
                center_x = np.random.randint(0, high=128)
                center_y = np.random.randint(0, high=128)

                # 随机半径与颜色
                radius = np.random.randint(3, high=30)
                # color = np.random.randint(0, high=256, size=(3,)).tolist()
                color = np.random.randint(0, high=256, size=(3,)).tolist()

                cv2.circle(img, (center_x, center_y), radius, color, -1)
            cv2.imwrite(valid_1 + "%d.png" % abnum, img)
        opt = Options().parse()
        opt.dataset = "coil100"
        opt.nc = 3
        opt.normal_class = obj
        data = load_data(opt)
        model = load_model(opt, data)
        model.train()


if __name__ == '__main__':
    train_cifar10()