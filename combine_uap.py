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
from utils.utils import get_model_path, get_result_path, get_uaps_path
from utils.utils import print_log
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import train, save_checkpoint, metrics_evaluate, eval_uap
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg

from matplotlib import pyplot as plt

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate general UAP from candidate UAPs')
    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

    return args


def main():
    args = parse_arguments()
    #test
    #for input, gt in pretrained_data_test_loader:
    #    clean_output = target_network(input)
    #    attack_output = target_network(input + tuap)

    # evaluate uap
    #load uap
    uaps_path = get_uaps_path()

    uap_list = []
    for fn in os.listdir(uaps_path):
        each_uap_path = os.path.join(uaps_path, fn)
        each_uap = np.load(each_uap_path)
        uap_list.append(each_uap)

    uap_list = np.array(uap_list)
    uap_list = np.sum(uap_list, axis=0) * 10

    np.save(uaps_path + '/uap_sum.npy', uap_list)

    plot_tuap = uap_list[0]
    plot_tuap = np.transpose(plot_tuap, (1, 2, 0))
    plot_tuap_normal = plot_tuap + 0.5
    plot_tuap_amp = plot_tuap / 2 + 0.5
    tuap_range = np.max(plot_tuap_amp) - np.min(plot_tuap_amp)
    plot_tuap_amp = plot_tuap_amp / tuap_range + 0.5
    plot_tuap_amp -= np.min(plot_tuap_amp)

    imgplot = plt.imshow(plot_tuap_amp)
    plt.savefig(uaps_path + '/uap_sum.png')


if __name__ == '__main__':
    main()
