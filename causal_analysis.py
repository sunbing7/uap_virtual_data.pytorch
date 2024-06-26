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
from utils.utils import get_model_path, get_result_path, get_uap_path, get_neuron_path, get_neuron_name
from utils.utils import print_log
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import train, save_checkpoint, metrics_evaluate, solve_causal
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg
import skimage.measure
import scipy
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import cv2


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Causality Analysis')
    parser.add_argument('--causal_type', default='logit', choices=['logit', 'act', 'slogit', 'sact', 'uap_act', 'inact', 'be_act'],
                        help='Causality analysis type (default: logit)')
    # pretrained
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='Used dataset to generate UAP (default: cifar10)')

    # candidate model
    parser.add_argument('--pretrained_dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Used dataset to train the initial model (default: cifar10)')
    parser.add_argument('--pretrained_arch', default='alexnet', choices=['vgg16_cifar', 'vgg19_cifar', 'resnet20', 'resnet56',
                                                                       'alexnet', 'googlenet', 'vgg16', 'vgg19',
                                                                       'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                                       'inception_v3'],
                        help='Used model architecture: (default: alexnet)')
    # filter model
    parser.add_argument('--filter_arch', default='vgg19', choices=['vgg16_cifar', 'vgg19_cifar', 'resnet20', 'resnet56',
                                                                       'alexnet', 'googlenet', 'vgg16', 'vgg19',
                                                                       'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                                       'inception_v3'])
    parser.add_argument('--filter_dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='dataset to train the filter model (default: cifar10)')
    parser.add_argument('--filter_name', type=str, default='vgg19_cifar10.pth',
                        help='filter model name (default: vgg19_cifar10.pth)')


    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')

    parser.add_argument('--split_layer', type=int, default=43,
                        help='causality analysis layer (default: 43)')
    # Parameters regarding UAP
    parser.add_argument('--num_iterations', type=int, default=32,
                        help='Number of iterations for causality analysis (default: 32)')
    parser.add_argument('--result_subfolder', default='result', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')

    parser.add_argument('--targeted',  type=bool, default='',
                        help='Target a specific class (default: False)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Target class (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--is_nips', action='store_true', default=True, help='Evaluation on NIPS data')

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

    # get the result path to store the results
    result_path = get_result_path(dataset_name=args.dataset,
                                network_arch=args.pretrained_arch,
                                random_seed=args.pretrained_seed,
                                result_subfolder=args.result_subfolder,
                                postfix=args.postfix)

    # Init logger
    log_file_name = os.path.join(result_path, 'log.txt')
    print("Log file: {}".format(log_file_name))
    log = open(log_file_name, 'w')
    print_log('save path : {}'.format(result_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print_log("{} : {}".format(key, value), log)
    print_log("Random Seed: {}".format(args.pretrained_seed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("Torch  version : {}".format(torch.__version__), log)
    print_log("Cudnn  version : {}".format(torch.backends.cudnn.version()), log)

    _, pretrained_data_test = get_data(args.pretrained_dataset, args.pretrained_dataset)

    pretrained_data_test_loader = torch.utils.data.DataLoader(pretrained_data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)

    ##### Dataloader for training: perturbed data ####    #load uap
    uap = None
    if args.causal_type != 'inact':
        uap_path = get_uap_path(uap_data=args.dataset,
                                model_data=args.pretrained_dataset,
                                network_arch=args.pretrained_arch,
                                random_seed=args.pretrained_seed)
        uap_fn = os.path.join(uap_path, 'uap.npy')
        uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
        uap = torch.from_numpy(uap)

    data_train, _ = get_data(args.filter_dataset, args.filter_dataset)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    ####################################
    # Init model, criterion, and optimizer
    print_log("=> Creating model '{}'".format(args.filter_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.filter_dataset,
                                network_arch=args.filter_arch,
                                random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, args.filter_name)

    filter_network = get_network(args.filter_arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    print_log("=> Network :\n {}".format(filter_network), log)
    #filter_network = torch.nn.DataParallel(filter_network, device_ids=list(range(args.ngpu)))
    # Set the target model into evaluation mode
    filter_network.eval()
    # Imagenet models use the pretrained pytorch weights
    if args.pretrained_dataset != "imagenet":
        #network_data = torch.load(model_weights_path, map_location=torch.device('cpu'))
        #target_network.load_state_dict(network_data['state_dict'])
        #target_network.load_state_dict(network_data.state_dict())
        filter_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    #set_parameter_requires_grad(target_network, requires_grad=False)
    total_params = get_num_parameters(filter_network)

    print_log("Filter Network Total # parameters: {}".format(total_params), log)

    if args.use_cuda:
        filter_network.cuda()

    # perform causality analysis
    neuron_ranking = solve_causal(data_train_loader, filter_network, uap, args.filter_arch,
                                  split_layer=args.split_layer,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  log=log,
                                  use_cuda=args.use_cuda)

    if args.causal_type == 'inact':
        outstanding_neuron = neuron_ranking
    else:
        # find outstanding neuron neuron_ranking shape: 4096x2
        temp = neuron_ranking
        ind = np.argsort(temp[:, 1])[::-1]
        temp = temp[ind]
        top = outlier_detection(temp[:, 1], max(temp[:, 1]), verbose=False)
        outstanding_neuron = temp[0: len(top)][:, 0]
    print('total:{}, top:{}'.format(len(neuron_ranking), len(outstanding_neuron)))

    neuron_path = get_neuron_path()

    neuron_fn = get_neuron_name(uap_data=args.dataset,
                                filter_data=args.filter_dataset,
                                uap_arch=args.pretrained_arch,
                                filter_arch=args.filter_arch,
                                random_seed=args.pretrained_seed)

    uap_fn = os.path.join(neuron_path, neuron_fn)
    np.save(uap_fn, outstanding_neuron)
    #neuron_fn = os.path.join(neuron_path, 'ranking_' + args.pretrained_arch + '.npy')
    #np.save(neuron_fn, temp)
    log.close()
    return

def outlier_detection(cmp_list, max_val, verbose=False, th=2):
    cmp_list = list(np.array(cmp_list) / max_val)
    consistency_constant = 1.4826  # if normal distribution
    median = np.median(cmp_list)
    mad = consistency_constant * np.median(np.abs(cmp_list - median))  # median of the deviation
    if mad != 0:

        #min_mad = np.abs(np.min(cmp_list) - median) / mad

        # print('median: %f, MAD: %f' % (median, mad))
        # print('anomaly index: %f' % min_mad)

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] < median:
                i = i + 1
                continue
            if np.abs(cmp_list[i] - median) / mad > th:
                flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
    else:

        flag_list = []
        i = 0
        for cmp in cmp_list:
            if cmp_list[i] <= median:
                i = i + 1
                continue
            flag_list.append((i, cmp_list[i]))
            i = i + 1

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])
            if verbose:
                print('flagged label list: %s' %
                      ', '.join(['%d: %2f' % (idx, val)
                                 for idx, val in flag_list]))
    return flag_list


def calculate_ssim(before, after):
    before = np.array(before * 255).astype('uint8')
    after = np.array(after * 255).astype('uint8')

    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)
    # Compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, full=True)

    return score

def calculate_shannon_entropy(x, size):
    """
    calculate information entropy
    Returns:
    H
    """
    x = np.array(x * 255).astype('uint8')
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    #p = np.histogram(x, bins=size, density=True)[0]
    #h = scipy.stats.entropy(p, base=2)

    h = skimage.measure.shannon_entropy(x)

    '''
    x = x.flatten(order='C')
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    h = -np.sum(probabilities * np.log2(probabilities))
    '''
    return h


def calculate_shannon_entropy_array(x):
    """
    calculate information entropy
    Returns:
    H
    """
    x = np.array(x)

    x = x.flatten(order='C')
    _, counts = np.unique(x, return_counts=True)
    probabilities = counts / len(x)
    h = -np.sum(probabilities * np.log2(probabilities))

    return h


def calculate_shannon_entropy_batch(x):
    """
    calculate information entropy
    Returns:
    H
    """
    x = np.array(x)
    out = np.zeros(x.shape)
    for index, x_i in enumerate(x):
        _, counts = np.unique(x_i, return_counts=True)
        probabilities = counts / x.shape[-1]
        product = probabilities * np.log2(probabilities)
        out[index][:len(counts)] = product

    h = -np.sum(out, axis=-1)
    return h


def calc_hloss(x):
    x = np.array(x)
    b = my_softmax((x)) * my_logsoftmax(x)
    b = -1.0 * b.sum()
    return b


def my_softmax(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def my_logsoftmax(x):
    c = x.max()
    logsumexp = np.log(np.exp(x - c).sum())
    return x - c - logsumexp

if __name__ == '__main__':
    main()


