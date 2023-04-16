from __future__ import division
import os
import numpy as np
import glob
import torch
import random
# import cv2
from torch.utils.data import Dataset
import pandas as pd

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
        '''
        train_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        ])

        test_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        ])
        '''
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
        #traindir = os.path.join(IMAGENET_PATH, 'train')
        #valdir = os.path.join(IMAGENET_PATH, 'val')

        traindir = IMAGENET_PATH
        valdir = IMAGENET_PATH

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
'''
def get_data_class(data_file, cur_class=3):
    #num_classes, (mean, std), input_size, num_channels = get_data_specs(pretrained_dataset)
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

    train_data = CustomCifarDataset(data_file, is_train=True, cur_class=cur_class, transform=train_transform)
    test_data = CustomCifarDataset(data_file, is_train=False, cur_class=cur_class, transform=test_transform)

    return train_data, test_data

class CustomCifarDataset(Dataset):
    def __init__(self, data_file, is_train=False, cur_class=3, transform=False):
        self.is_train = is_train
        self.cur_class = cur_class
        self.data_file = data_file
        self.transform = transform
        dataset = utils_backdoor.load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])
        #trig_mask = np.load(RESULT_DIR + "uap_trig_0.08.npy") * 255
        x_train = dataset['X_train'].astype("float32")# / 255
        y_train = dataset['Y_train'].T[0]#self.to_categorical(dataset['Y_train'], 10)
        #y_train = self.to_categorical(dataset['Y_train'], 10)
        x_test = dataset['X_test'].astype("float32") #/ 255
        y_test = dataset['Y_test'].T[0]#self.to_categorical(dataset['Y_test'], 10)
        #y_test = self.to_categorical(dataset['Y_test'], 10)

        x_out = []
        y_out = []
        for i in range(0, len(x_test)):
            #if np.argmax(y_test[i], axis=1) == cur_class:
            if y_test[i] == cur_class:
                x_out.append(x_test[i])# + trig_mask)
                y_out.append(y_test[i])
        self.X_test = np.uint8(np.array(x_out))
        self.Y_test = np.uint8(np.squeeze(np.array(y_out)))

        x_out = []
        y_out = []
        for i in range(0, len(x_train)):
            #if np.argmax(y_train[i], axis=1) == cur_class:
            if y_train[i] == cur_class:
                x_out.append(x_train[i])# + trig_mask)
                y_out.append(y_train[i])
        self.X_train = np.uint8(np.array(x_out))
        self.Y_train = np.uint8(np.squeeze(np.array(y_out)))

    def __len__(self):
        if self.is_train:
            return len(self.X_train)
        else:
            return len(self.X_test)

    def __getitem__(self, idx):
        if self.is_train:
            image = self.X_train[idx]
            label = self.Y_train[idx]
        else:
            image = self.X_test[idx]
            label = self.Y_test[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]
'''


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
    image_classes = pd.read_csv(os.path.join(IMAGENET_PATH, "images.csv"))
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