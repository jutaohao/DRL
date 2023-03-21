"""
CREATE DATASETS
"""

# pylint: disable=C0301,E1101,W0622,C0103,R0902,R0915
import cv2
import torch.utils.data as data
import torch
from random import shuffle

from torch.utils.data.sampler import RandomSampler
from torchvision.datasets import DatasetFolder

from pathlib import Path
from PIL import Image
import numpy as np
import os
import os.path
import random
import imageio
import numpy as np

# pylint: disable=E1101

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, nz=100, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.noise = torch.FloatTensor(len(self.imgs), nz, 1, 1).normal_(0, 1)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        latentz = self.noise[index]

        # TODO: Return these variables in a dict.
        # return img, latentz, index, target
        return {'image': img, 'latentz': latentz, 'index': index, 'frame_gt': target}

    def __setitem__(self, index, value):
        self.noise[index] = value

    def __len__(self):
        return len(self.imgs)

# TODO: refactor cifar-mnist anomaly dataset functions into one generic function.

def get_cifar_nomaly_train_dataset(train_ds,valid_ds,test_ds, abn_cls_idx=0,abnomal_class=0):
    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.targets)

    nrm_idx = np.where(trn_lbl == abn_cls_idx)[0]
    nrm_trn_idx = nrm_idx[0:int(len(nrm_idx)*0.9)]
    nrm_val_idx = nrm_idx[int(len(nrm_idx)*0.9):]

    abn_val_idx = np.where(trn_lbl != abn_cls_idx)[0]
    # abn_val_idx = np.where(trn_lbl == abnomal_class)[0]
    abn_val_idx = list(abn_val_idx)
    abn_val_idx = random.sample(abn_val_idx, 500)

    tst_img, tst_lbl = test_ds.data, np.array(test_ds.targets)
    nrm_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]

    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    nrm_val_img = trn_img[nrm_val_idx]
    abn_val_img = trn_img[abn_val_idx]

    for rcnt in range(0,500):
        img = np.random.random((32, 32, 3))
        for i in range(0, 5):
            # 随机中心点
            center_x = np.random.randint(0, high=32)
            center_y = np.random.randint(0, high=32)
            # 随机半径与颜色
            radius = np.random.randint(3, high=32 / 5)
            # color = np.random.randint(0, high=256, size=(3,)).tolist()
            color = np.random.randint(0, high=256, size=(3,)).tolist()
            cv2.circle(img, (center_x, center_y), radius, color, -1)
        abn_val_img[rcnt] = nrm_val_img[rcnt] + img
    # cv2.imshow("img", abn_val_img[499])
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    nrm_tst_img = tst_img[nrm_tst_idx]
    abn_tst_img = tst_img[abn_tst_idx]


    nrm_trn_lbl = trn_lbl[nrm_trn_idx]
    nrm_val_lbl = trn_lbl[nrm_val_idx]
    abn_val_lbl = trn_lbl[abn_val_idx]
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]
    abn_tst_lbl = tst_lbl[abn_tst_idx]

    nrm_trn_lbl[:] =  0
    nrm_val_lbl[:] =  0
    abn_val_lbl[:] =  1
    nrm_tst_lbl[:] =  0
    abn_tst_lbl[:] =  1

    train_ds.data = np.copy(nrm_trn_img)
    # valid_ds.data = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    valid_ds.data = np.concatenate((nrm_val_img, abn_val_img), axis=0)
    test_ds.data =  np.concatenate((nrm_tst_img, abn_tst_img), axis=0)

    train_ds.targets = np.copy(nrm_trn_lbl)
    # valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    valid_ds.targets = np.concatenate((nrm_val_lbl, abn_val_lbl), axis=0)
    test_ds.targets = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)

    return train_ds, valid_ds,test_ds

def get_FashionMNIST_nomaly_train_dataset(train_ds,valid_ds,test_ds, norm_cls_idx=0,abnomal_class=0):
    # Get images and labels.
    #p1 训练集里面选80% 20%测试  新奇从测试集选取50%
    trn_img, trn_lbl = train_ds.data, train_ds.targets
    print("trn_img",trn_img.shape)
    # nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_idx = np.where(trn_lbl.numpy() == norm_cls_idx)[0]
    nrm_idx = list(nrm_idx)
    nrm_trn_val_idx = random.sample(nrm_idx,int(len(nrm_idx)*0.8))

    nrm_tst_idx = [i for i in nrm_idx if not i in nrm_trn_val_idx]
    nrm_trn_idx = nrm_trn_val_idx[:int(len(nrm_trn_val_idx)*0.9)]
    nrm_val_idx = [i for i in nrm_trn_val_idx if not i in nrm_trn_idx]

    nrm_trn_img = trn_img[nrm_trn_idx]
    nrm_val_img = trn_img[nrm_val_idx]
    nrm_tst_img = trn_img[nrm_tst_idx]

    abn_val_img =  trn_img[nrm_val_idx]
    for rcnt in range(0, len(nrm_val_idx)):
        abn_val_idx2 = list(nrm_val_idx)
        abn_val_idx2 = random.sample(abn_val_idx2, 10)
        for id in abn_val_idx2:
            abn_val_img[rcnt] = trn_img[rcnt] + trn_img[id]


    tst_img, tst_lbl = test_ds.data, test_ds.targets
    abn_tst_idx = np.where(tst_lbl.numpy() != norm_cls_idx)[0]
    abn_tst_idx = list(abn_tst_idx)
    abn_tst_idx = random.sample(abn_tst_idx, int(len(abn_tst_idx) * 0.5))

    abn_tst_img = tst_img[abn_tst_idx]
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]
    nrm_val_lbl = trn_lbl[nrm_val_idx]
    abn_val_lbl = trn_lbl[nrm_val_idx]
    nrm_tst_lbl = trn_lbl[nrm_tst_idx]
    abn_tst_lbl = tst_lbl[abn_tst_idx]

    # --
    # Assign labels to normal (0) and novel (1)
    nrm_trn_lbl[:] = 0
    nrm_val_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_val_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    train_ds.data = nrm_trn_img.clone()
    test_ds.data = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    train_ds.targets = nrm_trn_lbl.clone()
    test_ds.targets = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)
    valid_ds.data = torch.cat((nrm_val_img, abn_val_img), dim=0)
    valid_ds.targets = torch.cat((nrm_val_lbl, abn_val_lbl), dim=0)
    return train_ds, valid_ds,test_ds

def get_MINISTP1_nomaly_train_dataset(train_ds,valid_ds,test_ds, norm_cls_idx=0,abnomal_class=0):
    # Get images and labels.
    #p1 训练集里面选80% 20%测试  新奇从测试集选取50%
    trn_img, trn_lbl = train_ds.data, train_ds.targets
    # nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_idx = np.where(trn_lbl.numpy() == norm_cls_idx)[0]
    nrm_idx = list(nrm_idx)
    nrm_trn_val_idx = random.sample(nrm_idx,int(len(nrm_idx)*0.8))


    nrm_tst_idx = [i for i in nrm_idx if not i in nrm_trn_val_idx]
    nrm_trn_idx = nrm_trn_val_idx[:int(len(nrm_trn_val_idx)*0.9)]
    nrm_val_idx = [i for i in nrm_trn_val_idx if not i in nrm_trn_idx]
    nrm_trn_img = trn_img[nrm_trn_idx]
    nrm_val_img = trn_img[nrm_val_idx]
    nrm_tst_img = trn_img[nrm_tst_idx]

    abn_val_img =  trn_img[nrm_val_idx]
    for rcnt in range(0, len(nrm_val_idx)):
        abn_val_idx2 = list(nrm_val_idx)
        abn_val_idx2 = random.sample(abn_val_idx2, 10)
        for id in abn_val_idx2:
            abn_val_img[rcnt] = trn_img[rcnt] + trn_img[id]


    tst_img, tst_lbl = test_ds.data, test_ds.targets
    abn_tst_idx = np.where(tst_lbl.numpy() != norm_cls_idx)[0]
    abn_tst_idx = list(abn_tst_idx)
    abn_tst_idx = random.sample(abn_tst_idx, int(len(abn_tst_idx) * 0.5))


    abn_tst_img = tst_img[abn_tst_idx]
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]
    nrm_val_lbl = trn_lbl[nrm_val_idx]
    abn_val_lbl = trn_lbl[nrm_val_idx]
    nrm_tst_lbl = trn_lbl[nrm_tst_idx]
    abn_tst_lbl = tst_lbl[abn_tst_idx]

    # --
    # Assign labels to normal (0) and novel (1)
    nrm_trn_lbl[:] = 0
    nrm_val_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_val_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    train_ds.data = nrm_trn_img.clone()
    test_ds.data = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    train_ds.targets = nrm_trn_lbl.clone()
    test_ds.targets = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)
    valid_ds.data = torch.cat((nrm_val_img, abn_val_img), dim=0)
    valid_ds.targets = torch.cat((nrm_val_lbl, abn_val_lbl), dim=0)
    return train_ds, valid_ds,test_ds
##
def get_cifar_anomaly_dataset(train_ds, valid_ds, abn_cls_idx=0):
    """[summary]
    Arguments:
        train_ds {Dataset - CIFAR10} -- Training dataset
        valid_ds {Dataset - CIFAR10} -- Validation dataset.
    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})
    Returns:
        [np.array] -- New training-test images and labels.
    """

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, np.array(train_ds.targets)
    tst_img, tst_lbl = valid_ds.data, np.array(valid_ds.targets)

    # --
    # Find idx, img, lbl for abnormal and normal on org dataset.
    #等于正常类ID的为正常类数据
    nrm_trn_idx = np.where(trn_lbl == abn_cls_idx)[0]
    abn_trn_idx = np.where(trn_lbl != abn_cls_idx)[0]

    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.

    nrm_tst_idx = np.where(tst_lbl == abn_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl != abn_cls_idx)[0]
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    # - anomaly dataset
    #   . -> train
    #        . -> normal
    #   . -> test
    #        . -> normal
    #        . -> abnormal
    # train_ds.data = np.copy(nrm_trn_img)
    # valid_ds.data = np.concatenate((abn_trn_img, abn_tst_img), axis=0)
    # train_ds.targets = np.copy(nrm_trn_lbl)
    # valid_ds.targets = np.concatenate((abn_trn_lbl, abn_tst_lbl), axis=0)

    train_ds.data = np.copy(nrm_trn_img)
    # valid_ds.data = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    valid_ds.data = np.concatenate((nrm_tst_img, abn_tst_img), axis=0)

    train_ds.targets = np.copy(nrm_trn_lbl)
    # valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)
    return train_ds, valid_ds

##
def get_mnist_anomaly_dataset(train_ds, valid_ds, abn_cls_idx=0):
    """[summary]
    Arguments:
        train_ds {Dataset - MNIST} -- Training dataset
        valid_ds {Dataset - MNIST} -- Validation dataset.
    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})
    Returns:
        [np.array] -- New training-test images and labels.
    """

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, train_ds.targets
    tst_img, tst_lbl = valid_ds.data, valid_ds.targets

    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    print("nrm_trn_idx",nrm_trn_idx)
    print(nrm_trn_idx.shape)
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    train_ds.data = nrm_trn_img.clone()
    valid_ds.data = torch.cat((nrm_tst_img, abn_tst_img), dim=0)
    train_ds.targets = nrm_trn_lbl.clone()
    valid_ds.targets = torch.cat((nrm_tst_lbl, abn_tst_lbl), dim=0)

    return train_ds, valid_ds

def get_MNIST_nomaly_train_dataset(train_ds,valid_ds,test_ds, norm_cls_idx=0,abnomal_class=0):
    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, train_ds.targets
    # Get images and labels.
    nrm_idx = np.where(trn_lbl == norm_cls_idx)[0]

    nrm_trn_idx = nrm_idx[0:int(len(nrm_idx) * 0.9)]
    nrm_val_idx = nrm_idx[int(len(nrm_idx) * 0.9):]

    abn_val_idx = np.where(trn_lbl == norm_cls_idx)[0]
    # abn_val_idx = np.where(trn_lbl == abnomal_class)[0]
    abn_val_idx = list(abn_val_idx)
    abn_val_idx = random.sample(abn_val_idx, 500)

    tst_img, tst_lbl = test_ds.data, np.array(test_ds.targets)
    nrm_tst_idx = np.where(tst_lbl == norm_cls_idx)[0]
    abn_tst_idx = np.where(tst_lbl != norm_cls_idx)[0]

    nrm_trn_img = trn_img[nrm_trn_idx]  # Normal training images
    nrm_val_img = trn_img[nrm_val_idx]
    abn_val_img = trn_img[abn_val_idx]
    for rcnt in range(0, 500):
        abn_val_idx2 = list(abn_val_idx)
        abn_val_idx2 = random.sample(abn_val_idx2, 10)
        for id in abn_val_idx2:
            abn_val_img[rcnt] = trn_img[rcnt] + trn_img[id]

    nrm_tst_img = tst_img[nrm_tst_idx]
    abn_tst_img = tst_img[abn_tst_idx]

    nrm_trn_lbl = trn_lbl[nrm_trn_idx]
    nrm_val_lbl = trn_lbl[nrm_val_idx]
    abn_val_lbl = trn_lbl[abn_val_idx]
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]
    abn_tst_lbl = tst_lbl[abn_tst_idx]

    nrm_trn_lbl[:] = 0
    nrm_val_lbl[:] = 0
    abn_val_lbl[:] = 1
    nrm_tst_lbl[:] = 0
    abn_tst_lbl[:] = 1

    train_ds.data = np.copy(nrm_trn_img)
    # valid_ds.data = np.concatenate((nrm_tst_img, abn_trn_img, abn_tst_img), axis=0)
    valid_ds.data = np.concatenate((nrm_val_img, abn_val_img), axis=0)
    test_ds.data = np.concatenate((nrm_tst_img, abn_tst_img), axis=0)

    train_ds.targets = np.copy(nrm_trn_lbl)
    # valid_ds.targets = np.concatenate((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), axis=0)
    valid_ds.targets = np.concatenate((nrm_val_lbl, abn_val_lbl), axis=0)
    test_ds.targets = np.concatenate((nrm_tst_lbl, abn_tst_lbl), axis=0)

    return train_ds, valid_ds, test_ds
##
def make_anomaly_dataset(train_ds, valid_ds, abn_cls_idx=0):
    """[summary]

    Arguments:
        train_ds {Dataset - MNIST} -- Training dataset
        valid_ds {Dataset - MNIST} -- Validation dataset.

    Keyword Arguments:
        abn_cls_idx {int} -- Anomalous class index (default: {0})

    Returns:
        [np.array] -- New training-test images and labels.
    """

    # Check the input type.
    if isinstance(train_ds.data, np.ndarray):
        train_ds.data = torch.from_numpy(train_ds.data)
        valid_ds.data = torch.from_numpy(valid_ds.data)
        train_ds.targets = torch.Tensor(train_ds.targets)
        valid_ds.targets = torch.Tensor(valid_ds.targets)

    # Get images and labels.
    trn_img, trn_lbl = train_ds.data, train_ds.targets
    tst_img, tst_lbl = valid_ds.data, valid_ds.targets

    # --
    # Find normal abnormal indexes.
    # TODO: PyTorch v0.4 has torch.where function
    nrm_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() != abn_cls_idx)[0])
    abn_trn_idx = torch.from_numpy(np.where(trn_lbl.numpy() == abn_cls_idx)[0])
    nrm_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() != abn_cls_idx)[0])
    abn_tst_idx = torch.from_numpy(np.where(tst_lbl.numpy() == abn_cls_idx)[0])

    # --
    # Find normal and abnormal images
    nrm_trn_img = trn_img[nrm_trn_idx]    # Normal training images
    abn_trn_img = trn_img[abn_trn_idx]    # Abnormal training images.
    nrm_tst_img = tst_img[nrm_tst_idx]    # Normal training images
    abn_tst_img = tst_img[abn_tst_idx]    # Abnormal training images.

    # --
    # Find normal and abnormal labels.
    nrm_trn_lbl = trn_lbl[nrm_trn_idx]    # Normal training labels
    abn_trn_lbl = trn_lbl[abn_trn_idx]    # Abnormal training labels.
    nrm_tst_lbl = tst_lbl[nrm_tst_idx]    # Normal training labels
    abn_tst_lbl = tst_lbl[abn_tst_idx]    # Abnormal training labels.

    # --
    # Assign labels to normal (0) and abnormals (1)
    nrm_trn_lbl[:] = 0
    nrm_tst_lbl[:] = 0
    abn_trn_lbl[:] = 1
    abn_tst_lbl[:] = 1

    # Create new anomaly dataset based on the following data structure:
    train_ds.data = nrm_trn_img.clone()
    valid_ds.data = torch.cat((nrm_tst_img, abn_trn_img, abn_tst_img), dim=0)
    train_ds.targets = nrm_trn_lbl.clone()
    valid_ds.targets = torch.cat((nrm_tst_lbl, abn_trn_lbl, abn_tst_lbl), dim=0)

    return train_ds, valid_ds