from __future__ import division
import os
import numpy as np
import glob
import torch
import random
# import cv2
from torch.utils.data import Dataset
import pandas as pd
import h5py

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.transforms.functional as tfunc

from config.config import IMAGENET_PATH, DATASET_BASE_PATH
from config.config import COCO_2017_TRAIN_IMGS, COCO_2017_VAL_IMGS, COCO_2017_TRAIN_ANN, COCO_2017_VAL_ANN, VOC_2012_ROOT, PLACES365_ROOT
from dataset_utils.voc0712 import VOCDetection

#import utils.utils_backdoor as utils_backdoor
#DATA_DIR = 'data'  # data folder
#DATA_FILE = 'cifar_dataset.h5'  # dataset file

def get_data_specs(pretrained_dataset):
    if pretrained_dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000
        input_size = 224
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == "cifar10":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 10
        input_size = 224#32
        num_channels = 3
    elif pretrained_dataset == "cifar100":
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 100
        input_size = 32
        num_channels = 3
    else:
        raise ValueError
    return num_classes, (mean, std), input_size, num_channels


def get_data(dataset, pretrained_dataset):

    num_classes, (mean, std), input_size, num_channels = get_data_specs(pretrained_dataset)

    if dataset == 'cifar10':
        #return get_data_class(data_file=('%s/%s' % (DATA_DIR, DATA_FILE)), cur_class=3)
        train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.Resize(size=(224, 224)),
                 transforms.RandomCrop(input_size, padding=4),
                 transforms.ToTensor(),
                 #transforms.Normalize(mean, std),
                 transforms.Normalize(
                     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                 )
                 ])
        

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Resize(size=(224, 224)),
                 #transforms.Normalize(mean, std),
                 transforms.Normalize(
                     (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                 )
                 ])


        train_data = dset.CIFAR10(DATASET_BASE_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(DATASET_BASE_PATH, train=False, transform=test_transform, download=True)
    
    elif dataset == 'cifar100':
        train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(input_size, padding=4),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])

        test_transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = dset.CIFAR100(DATASET_BASE_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(DATASET_BASE_PATH, train=False, transform=test_transform, download=True)
    
    elif dataset == "imagenet":
        #use imagenet 2012 validation set as uap training set
        #use imagenet DEV 1000 sample dataset as the test set
        #traindir = os.path.join(IMAGENET_PATH, 'train')
        #valdir = os.path.join(IMAGENET_PATH, 'val')
        traindir = os.path.join(IMAGENET_PATH, 'validation')
        #traindir = IMAGENET_PATH
        valdir = os.path.join(IMAGENET_PATH, 'ImageNet1k')

        train_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=valdir, transform=test_transform)

    elif dataset == "coco":
        train_transform = transforms.Compose([
                transforms.Resize(int(input_size * 1.143)),
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.Resize(int(input_size * 1.143)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = dset.CocoDetection(root=COCO_2017_TRAIN_IMGS,
                                        annFile=COCO_2017_TRAIN_ANN,
                                        transform=train_transform)
        test_data = dset.CocoDetection(root=COCO_2017_VAL_IMGS,
                                        annFile=COCO_2017_VAL_ANN,
                                        transform=test_transform)
    
    elif dataset == "voc":
        train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(input_size * 1.143)),
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(int(input_size * 1.143)),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        train_data = VOCDetection(root=VOC_2012_ROOT,
                                year="2012",
                                image_set='train',
                                transform=train_transform)
        test_data = VOCDetection(root=VOC_2012_ROOT,
                                year="2012",
                                image_set='val',
                                transform=test_transform)
    
    elif dataset == "places365":
        traindir = os.path.join(PLACES365_ROOT, "train")
        testdir = os.path.join(PLACES365_ROOT, "train")
        # Places365 downloaded as 224x224 images

        train_transform = transforms.Compose([
                transforms.Resize(input_size), # Places images downloaded as 224
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        
        train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=testdir, transform=test_transform)
    
    return train_data, test_data


'''
def get_data_perturbed(pretrained_dataset, uap):

    if pretrained_dataset == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)),
             transforms.Lambda(lambda y: (y + uap)),
             transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(224, padding=4),
             transforms.ToTensor(),
             # transforms.Normalize(mean, std),
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
             )
             ])

        test_transform = transforms.Compose(
            [transforms.Resize(size=(224, 224)),
             transforms.Lambda(lambda y: (y + uap)),
             transforms.ToTensor(),
             # transforms.Normalize(mean, std),
             transforms.Normalize(
                 (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
             )
             ])

        train_data = dset.CIFAR10(DATASET_BASE_PATH, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(DATASET_BASE_PATH, train=False, transform=test_transform, download=True)

    return train_data, test_data
'''

def get_data_class(dataset, cur_class=1):
    #num_classes, (mean, std), input_size, num_channels = get_data_specs(pretrained_dataset)
    if dataset == 'cifar10':

        data_file = DATASET_BASE_PATH + '/cifar.h5'
        train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        ])

        train_data = CustomCifarClassDataSet(data_file, is_train=True, cur_class=cur_class, transform=train_transform)
        test_data = CustomCifarClassDataSet(data_file, is_train=False, cur_class=cur_class, transform=test_transform)
    elif dataset == 'imagenet':
        num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)
        #use imagenet 2012 validation set as uap training set
        #use imagenet DEV 1000 sample dataset as the test set
        #traindir = os.path.join(IMAGENET_PATH, 'validation')
        valdir = os.path.join(IMAGENET_PATH, 'ImageNet1k')

        train_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        test_transform = transforms.Compose([
                transforms.Resize(256),
                # transforms.Resize(299), # inception_v3
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

        #train_data = dset.ImageFolder(root=traindir, transform=train_transform)
        test_data = dset.ImageFolder(root=valdir, transform=test_transform)

        #train_data = fix_labels(train_data)
        test_data = fix_labels_nips_class(test_data, pytorch=True)
        train_data = None

    else:
        return None
    return train_data, test_data


class CustomCifarClassDataSet(Dataset):
    def __init__(self, data_file, cur_class, transform=False, is_train=False):
        self.transform = transform

        dataset = load_dataset_h5(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
        if is_train:
            xs = dataset['X_train'].astype("uint8")
            ys = dataset['Y_train'].T[0]
        else:
            xs = dataset['X_test'].astype("uint8")
            ys = dataset['Y_test'].T[0]

        idxes = (ys == cur_class)
        self.x = xs[idxes]
        self.y = ys[idxes]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.y[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


#############################################################
# This will fix labels for NIPS ImageNet
def fix_labels_nips(test_set, pytorch=False, target_flag=False):
    '''
    :param pytorch: pytorch models have 1000 labels as compared to tensorflow models with 1001 labels
    '''

    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    # Load provided files and get image labels and names
    image_classes = pd.read_csv(os.path.join(IMAGENET_PATH, "ImageNet1k/images.csv"))
    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()
    val_dict = {}
    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = [true_classes[i], target_classes[i]]

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        if target_flag:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][1]
        else:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][0]
        if pytorch:
            new_data_samples.append((test_set.samples[i][0], org_label - 1))
        else:
            new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def fix_labels_nips_class(test_set, pytorch=False, target_flag=False, cur_class=1):
    '''
    :param pytorch: pytorch models have 1000 labels as compared to tensorflow models with 1001 labels
    '''

    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    # Load provided files and get image labels and names
    image_classes = pd.read_csv(os.path.join(IMAGENET_PATH, "ImageNet1k/images.csv"))
    image_metadata = pd.DataFrame({"ImageId": [f[:-4] for f in filenames]}).merge(image_classes, on="ImageId")
    true_classes = image_metadata["TrueLabel"].tolist()
    target_classes = image_metadata["TargetClass"].tolist()
    val_dict = {}
    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = [true_classes[i], target_classes[i]]

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        if target_flag:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][1]
        else:
            org_label = val_dict[test_set.samples[i][0].split('/')[-1]][0]
        if pytorch:
            if (org_label - 1) == cur_class:
                new_data_samples.append((test_set.samples[i][0], org_label - 1))
        else:
            if org_label == cur_class:
                new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def fix_ground_truth(test_set):
    filenames = [i.split('/')[-1] for i, j in test_set.samples]
    val_dict = {}
    val = []
    groudtruth = os.path.join(IMAGENET_PATH, 'validation/ILSVRC2012_validation_ground_truth.txt')

    with open(groudtruth) as file:
        for line in file:
            val.append(int(line))

    for f, i in zip(filenames, range(len(filenames))):
        val_dict[f] = val[i]

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        org_label = val_dict[test_set.samples[i][0].split('/')[-1]]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def fix_labels(test_set):
    val_dict = {}
    groudtruth = os.path.join(IMAGENET_PATH, 'validation/classes.txt')

    i = 0
    with open(groudtruth) as file:
        for line in file:
            (key, class_name) = line.split(':')
            val_dict[key] = i
            i = i + 1

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        class_id = test_set.samples[i][0].split('/')[-1].split('.')[0].split('_')[-1]
        org_label = val_dict[class_id]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def load_dataset_h5(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset