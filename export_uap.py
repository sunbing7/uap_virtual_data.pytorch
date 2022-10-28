from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict

from networks.uap import UAP
from utils.data import get_data_specs, get_data
from utils.utils import get_model_path, get_result_path, get_uap_path
from utils.utils import print_log
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import train, save_checkpoint, metrics_evaluate, eval_uap
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg

from matplotlib import pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Causality Analysis')
    # dataset used to train UAP
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='Used dataset to generate UAP (default: cifar10)')
    # dataset used to train UAP model
    parser.add_argument('--pretrained_dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Used dataset to train the initial model (default: cifar10)')
    # model used to train UAP
    parser.add_argument('--pretrained_arch', default='alexnet', choices=['vgg16_cifar', 'vgg19_cifar', 'resnet20', 'resnet56',
                                                                       'alexnet', 'googlenet', 'vgg16', 'vgg19',
                                                                       'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                                       'inception_v3'],
                        help='Used model architecture: (default: alexnet)')
    parser.add_argument('--model_name', type=str, default='alexnet_cifar10.pth',
                        help='model name (default: alexnet_cifar10.pth)')
    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--uap_model', type=str, default='checkpoint.pth.tar',
                        help='uap model name (default: checkpoint.pth.tar)')
    parser.add_argument('--uap_name', type=str, default='uap.npy',
                        help='uap file name (default: uap.npy)')

    # Parameters regarding UAP
    parser.add_argument('--result_subfolder', default='result', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')
    parser.add_argument('--targeted',  action='store_true', default='True',
                        help='Target a specific class (default: True)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Target class (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)
    return args


def main():
    args = parse_arguments()

    random.seed(args.pretrained_seed)
    torch.manual_seed(args.pretrained_seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.pretrained_seed)
    cudnn.benchmark = True

    # print_log("=> Inserting Generator", log)
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)
    generator = UAP(shape=(input_size, input_size),
                    num_channels=num_channels,
                    mean=mean,
                    std=std,
                    use_cuda=args.use_cuda)

    # load perturbed network data
    # get a path for loading the model to be attacked
    model_path = get_uap_path(uap_data=args.dataset,
                              model_data=args.pretrained_dataset,
                              network_arch=args.pretrained_arch,
                              random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, args.uap_model)

    network_data = torch.load(model_weights_path, map_location=torch.device('cpu'))
    generator.load_state_dict(network_data['state_dict'])

    #plot uap
    #'''
    tuap = torch.unsqueeze(generator.uap, dim=0)
    plot_tuap = tuap[0].cpu().detach().numpy()
    plot_tuap = np.transpose(plot_tuap, (1, 2, 0))
    plot_tuap_normal = plot_tuap + 0.5
    plot_tuap_amp = plot_tuap / 2 + 0.5
    tuap_range = np.max(plot_tuap_amp) - np.min(plot_tuap_amp)
    plot_tuap_amp = plot_tuap_amp / tuap_range + 0.5
    plot_tuap_amp -= np.min(plot_tuap_amp)

    imgplot = plt.imshow(plot_tuap_amp)
    plt.savefig(model_path + '/uap.png')
    plt.show()
    np.save(model_path + '/' + args.uap_name, tuap.cpu().detach().numpy())

    print('uap saved!')



if __name__ == '__main__':
    main()
