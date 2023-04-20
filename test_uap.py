from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict

from networks.uap import UAP
from utils.data import get_data_specs, get_data, fix_labels_nips, fix_labels
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

    # model to test
    parser.add_argument('--test_dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='Test model training set (default: cifar10)')
    parser.add_argument('--test_arch', default='vgg19', choices=['vgg16_cifar', 'vgg19_cifar', 'resnet20', 'resnet56',
                                                                   'alexnet', 'googlenet', 'vgg16', 'vgg19',
                                                                   'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                                   'inception_v3'],
                        help='Test model architecture: (default: vgg19)')
    parser.add_argument('--test_name', type=str, default='vgg19_cifar10.pth',
                        help='Test model name (default: vgg19_cifar10.pth)')

    # Parameters regarding UAP
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

    _, data_test = get_data(args.test_dataset, args.test_dataset)
    # Fix labels if needed
    if args.is_nips:
        print('is_nips')
        data_test = fix_labels_nips(data_test, pytorch=True)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)

    #data_train, _ = get_data(args.dataset, args.pretrained_dataset)
    #if args.dataset == "imagenet":
    #    data_train = fix_labels(data_train)
    #data_train_loader = torch.utils.data.DataLoader(data_train,
    #                                                batch_size=args.batch_size,
    #                                                shuffle=True,
    #                                                num_workers=args.workers,
    #                                                pin_memory=True)

    ####################################
    # Init model, criterion, and optimizer
    print_log("=> Creating model '{}'".format(args.test_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.test_dataset,
                                network_arch=args.test_arch,
                                random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, args.test_name)

    target_network = get_network(args.test_arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    print_log("=> Network :\n {}".format(target_network), log)
    #target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))
    # Set the target model into evaluation mode
    target_network.eval()
    if args.test_dataset != "imagenet":
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    #set_parameter_requires_grad(target_network, requires_grad=False)

    total_params = get_num_parameters(target_network)
    print_log("Target Network Total # parameters: {}".format(total_params), log)

    if args.use_cuda:
        target_network.cuda()

    #test
    #for input, gt in pretrained_data_test_loader:
    #    clean_output = target_network(input)
    #    attack_output = target_network(input + tuap)

    # evaluate uap
    #load uap
    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.pretrained_dataset,
                            network_arch=args.pretrained_arch,
                            random_seed=args.pretrained_seed)
    uap_fn = os.path.join(uap_path, args.uap_name)
    uap = np.load(uap_fn)
    tuap = torch.from_numpy(uap)

    test_sr, nt_sr, clean_test_acc, _test_sr, _nt_sr = eval_uap(data_test_loader, target_network, tuap,
                                                                 target_class=args.target_class, log=log, use_cuda=args.use_cuda, targeted=args.targeted)
    print('All samples: UAP targeted attack testing set SR: %.2f' % (test_sr))
    print('All samples: UAP non-targeted attack testing set SR: %.2f' % (nt_sr))
    print('UAP targeted attack testing set SR: %.2f' % (_test_sr))
    print('UAP non-targeted attack testing set SR: %.2f' % (_nt_sr))
    print('Clean sample test accuracy: %.2f' % clean_test_acc)
    '''
    metrics_evaluate(data_loader=pretrained_data_test_loader,
                    target_model=target_network,
                    perturbed_model=perturbed_net,
                    targeted=args.targeted,
                    target_class=args.target_class,
                    log=log,
                    use_cuda=args.use_cuda)

    save_checkpoint({
      'arch'        : args.pretrained_arch,
      # 'state_dict'  : perturbed_net.state_dict(),
      'state_dict'  : perturbed_net.module.generator.state_dict(),
      #'optimizer'   : optimizer.state_dict(),
      'args'        : copy.deepcopy(args),
    }, result_path, 'checkpoint_cifar10.pth.tar')
    '''
    log.close()


def main_net():
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

    _, data_test = get_data(args.test_dataset, args.test_dataset)
    # Fix labels if needed
    if args.is_nips:
        print('is_nips')
        data_test = fix_labels_nips(data_test, pytorch=True)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)

    #data_train, _ = get_data(args.dataset, args.pretrained_dataset)
    #if args.dataset == "imagenet":
    #    data_train = fix_labels(data_train)
    #data_train_loader = torch.utils.data.DataLoader(data_train,
    #                                                batch_size=args.batch_size,
    #                                                shuffle=True,
    #                                                num_workers=args.workers,
    #                                                pin_memory=True)

    ####################################
    # Init model, criterion, and optimizer
    print_log("=> Creating model '{}'".format(args.test_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.test_dataset,
                                network_arch=args.test_arch,
                                random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, args.test_name)

    target_network = get_network(args.test_arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    print_log("=> Network :\n {}".format(target_network), log)
    #target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))
    # Set the target model into evaluation mode
    target_network.eval()
    if args.test_dataset != "imagenet":
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    #set_parameter_requires_grad(target_network, requires_grad=False)

    total_params = get_num_parameters(target_network)
    print_log("Target Network Total # parameters: {}".format(total_params), log)

    if args.use_cuda:
        target_network.cuda()

    #test
    #for input, gt in pretrained_data_test_loader:
    #    clean_output = target_network(input)
    #    attack_output = target_network(input + tuap)

    # evaluate uap
    #load uap
    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.pretrained_dataset,
                            network_arch=args.pretrained_arch,
                            random_seed=args.pretrained_seed)
    uap_fn = os.path.join(uap_path, args.uap_name)

    uap_pert_model = torch.load(uap_fn, map_location=torch.device('cpu'))
    if args.use_cuda:
        uap_pert_model.cuda()

    #'''
    metrics_evaluate(data_loader=data_test_loader,
                    target_model=target_network,
                    perturbed_model=uap_pert_model,
                    targeted=args.targeted,
                    target_class=args.target_class,
                    log=log,
                    use_cuda=args.use_cuda)

    #'''
    log.close()

if __name__ == '__main__':
    #main()
    main_net()