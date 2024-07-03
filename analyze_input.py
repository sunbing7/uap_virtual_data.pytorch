from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn

from utils.data import get_data_specs, get_data, get_data_class
from utils.utils import get_model_path, get_result_path, get_uap_path, get_attribution_path, get_attribution_name
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import solve_input_attribution, solve_input_attribution_single, solve_causal, solve_causal_single, \
    my_test, my_test_uap, gen_low_entropy_sample, replace_model, adv_ae_train
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg
from causal_analysis import (calculate_shannon_entropy, calculate_ssim, calculate_shannon_entropy_array,
                             calculate_shannon_entropy_batch, calc_hloss)
from collections import OrderedDict
from activation_analysis import outlier_detection
from utils.training import *

import warnings
warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Perform Causality Analysis on Input')
    parser.add_argument('--option', default='analyze_inputs', choices=['analyze_inputs', 'calc_entropy',
                                                                       'analyze_layers', 'calc_pcc', 'analyze_clean',
                                                                       'test', 'pcc', 'entropy', 'classify',
                                                                       'repair_ae', 'analyze_entropy',
                                                                       'repair', 'repair_uap', 'gen_en_sample',
                                                                       'repair_enpool', 'repair_enrep'],
                        help='Run options')
    parser.add_argument('--causal_type', default='logit', choices=['logit', 'act', 'slogit', 'sact',
                                                                   'uap_act', 'inact', 'be_act'],
                        help='Causality analysis type (default: logit)')

    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet',
                                                                 'coco', 'voc', 'places365', 'caltech', 'asl',
                                                                 'eurosat'],
                        help='Used dataset to generate UAP (default: cifar10)')
    parser.add_argument('--is_train', type=int, default=0)
    parser.add_argument('--arch', default='alexnet', choices=['vgg16_cifar', 'vgg19_cifar', 'resnet20',
                                                           'resnet56', 'alexnet', 'googlenet', 'vgg16', 'vgg19',
                                                           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                           'inception_v3', 'shufflenetv2', 'mobilenet'])
    parser.add_argument('--model_name', type=str, default='vgg19_cifar10.pth',
                        help='model name (default: vgg19_cifar10.pth)')
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')

    parser.add_argument('--split_layer', type=int, default=43,
                        help='causality analysis layer (default: 43)')
    parser.add_argument('--split_layers', type=int, nargs="*", default=[43])
    # Parameters regarding UAP
    parser.add_argument('--num_iterations', type=int, default=32,
                        help='Number of iterations for causality analysis (default: 32)')
    parser.add_argument('--num_batches', type=int, default=1500)
    parser.add_argument('--result_subfolder', default='result', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')

    parser.add_argument('--idx', type=int, default=0)

    parser.add_argument('--targeted',  type=bool, default='',
                        help='Target a specific class (default: False)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Target class (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')

    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    parser.add_argument('--analyze_clean', type=int, default=0)

    parser.add_argument('--th', type=float, default=2)

    parser.add_argument('--alpha', type=float, default=0.1)

    parser.add_argument('--ae_alpha', type=float, default=0.5)
    parser.add_argument('--ae_iter', type=int, default=10)

    parser.add_argument('--is_nips', default=1, type=int,
                        help='Evaluation on NIPS data')

    parser.add_argument('--loss_function', default='ce', choices=['ce', 'neg_ce', 'logit', 'bounded_logit',
                                                                  'bounded_logit_fixed_ref', 'bounded_logit_neg'],
                        help='Used loss function for source classes: (default: bounded_logit_fixed_ref)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate (default: 0.001)')
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--epsilon', type=float, default=0.03922,
                        help='Norm restriction of UAP (default: 10/255)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
    print('use_cuda: {}'.format(args.use_cuda))
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    return args


def analyze_inputs(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # get the result path to store the results
    result_path = get_result_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed,
                                result_subfolder=args.result_subfolder,
                                postfix=args.postfix)

    print('save path : {}'.format(result_path))
    print("Random Seed: {}".format(args.seed))
    print("Python version : {}".format(sys.version.replace('\n', ' ')))
    print("Torch  version : {}".format(torch.__version__))
    print("Cudnn  version : {}".format(torch.backends.cudnn.version()))

    _, data_test = get_data(args.dataset, args.dataset)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    #load uap
    uap = None
    if args.causal_type != 'inact':
        uap_path = get_uap_path(uap_data=args.dataset,
                                model_data=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
        uap_fn = os.path.join(uap_path, 'uap.npy')
        uap = np.load(uap_fn) / np.array(std).reshape(1,3,1,1)
        uap = torch.from_numpy(uap)

    '''
    data_train, _ = get_data(args.dataset, args.dataset)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)
    '''
    ####################################
    # Init model, criterion, and optimizer
    #print("=> Creating model '{}'".format(args.arch))
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    network = get_network(args.arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    print("=> Network :\n {}".format(network))

    # Set the target model into evaluation mode
    network.eval()

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            network.load_state_dict(new_state_dict)

    elif args.dataset == 'eurosat':
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)
    total_params = get_num_parameters(network)

    print("Filter Network Total # parameters: {}".format(total_params))

    if args.use_cuda:
        network.cuda()

    # perform causality analysis
    attribution_map = solve_input_attribution_single(data_test_loader, network, uap,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  use_cuda=args.use_cuda)

    #save multiple maps
    attribution_path = get_attribution_path()
    for i in range(0, len(attribution_map)):
        attribution_map_ = attribution_map[i]
        uap_fn = os.path.join(attribution_path, "uap_attribution_single_" + str(i) + ".npy")
        np.save(uap_fn, attribution_map_)

    _, data_test = get_data_class(args.dataset, 1)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    attribution_map = solve_input_attribution(data_test_loader, network, None,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  use_cuda=args.use_cuda)
    uap_fn = os.path.join(attribution_path, "clean_attribution.npy")
    np.save(uap_fn, attribution_map)

    # single attribution on clean samples
    attribution_map = solve_input_attribution_single(data_test_loader, network, None,
                                  targeted=args.targeted,
                                  target_class=args.target_class,
                                  num_sample=args.num_iterations,
                                  causal_type=args.causal_type,
                                  use_cuda=args.use_cuda)

    #save multiple maps
    attribution_path = get_attribution_path()
    for i in range (0, len(attribution_map)):
        attribution_map_ = attribution_map[i]
        uap_fn = os.path.join(attribution_path, "clean_attribution_single_" + str(i) + ".npy")
        np.save(uap_fn, attribution_map_)
    return


def analyze_entropy(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
    ####################################
    # Init model
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    network = get_network(args.arch,
                          input_size=input_size,
                          num_classes=num_classes,
                          finetune=False)

    # Set the target model into evaluation mode
    network.eval()

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            network.load_state_dict(new_state_dict)

    elif args.dataset == 'eurosat':
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)

    if args.use_cuda:
        network.cuda()

    uap = None

    # load dataset
    _, data_test = get_data(args.dataset, args.dataset)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    ##################################################################################
    # anlayze clean inputs
    '''
    _, data_class_test = get_data_class(args.dataset, args.target_class)
    if len(data_class_test) == 0:
        print('No sample from class {}'.format(args.target_class))
        return
    data_test_class_loader = torch.utils.data.DataLoader(data_class_test,
                                                         batch_size=args.batch_size,
                                                         shuffle=True,
                                                         num_workers=args.workers,
                                                         pin_memory=True)
    '''

    ##################################################################################
    # anlayze perturbed inputs
    # load uap
    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=args.arch,
                            random_seed=args.seed)
    uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
    uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
    uap = torch.from_numpy(uap)

    attribution_map_clean, attribution_map_pert = solve_causal_single_both(data_test_loader,
                                                                           network,
                                                                           uap,
                                                                           args.arch,
                                                                           split_layer=args.split_layer,
                                                                           targeted=args.targeted,
                                                                           target_class=args.target_class,
                                                                           num_sample=args.num_iterations,
                                                                           causal_type=args.causal_type,
                                                                           log=None,
                                                                           use_cuda=args.use_cuda)


    #save multiple maps
    attribution_path = get_attribution_path()
    for i in range(0, len(attribution_map_pert)):
        attribution_map_ = attribution_map_pert[i]
        uap_fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + "_s" + str(i)
                              + ".npy")
        np.save(uap_fn, attribution_map_)

    for i in range(0, len(attribution_map_clean)):
        attribution_map_ = attribution_map_clean[i]
        uap_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + "_s" + str(i)
                              + ".npy")
        np.save(uap_fn, attribution_map_)

    ##################################################################################
    # anlayze entropy
    clean_hs = []
    uap_hs = []
    diff = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        args.analyze_clean = 1
        clean_h = calc_entropy_i(i, args)
        clean_hs.append(clean_h)
        args.analyze_clean = 0
        uap_h = calc_entropy_i(i, args)
        uap_hs.append(uap_h)
        diff.append(abs(clean_h - uap_h))
        #print('{}: {}'.format(clean_h, uap_h))

    clean_hs_avg = np.mean(np.array(clean_hs))
    uap_hs_avg = np.mean(np.array(uap_hs))
    print('clean_hs_avg,  uap_hs_avg: {} : {}'.format(clean_hs_avg, uap_hs_avg))

    # calculate quartiles
    clean_q1 = np.quantile(np.array(clean_hs), 0.25)
    clean_q2 = np.quantile(np.array(clean_hs), 0.5)
    clean_q3 = np.quantile(np.array(clean_hs), 0.75)
    clean_min = np.min(np.array(clean_hs))
    clean_max = np.max(np.array(clean_hs))

    uap_q1 = np.quantile(np.array(uap_hs), 0.25)
    uap_q2 = np.quantile(np.array(uap_hs), 0.5)
    uap_q3 = np.quantile(np.array(uap_hs), 0.75)
    uap_min = np.min(np.array(uap_hs))
    uap_max = np.max(np.array(uap_hs))

    diff_q1 = np.quantile(np.array(diff), 0.25)
    diff_q2 = np.quantile(np.array(diff), 0.5)
    diff_q3 = np.quantile(np.array(diff), 0.75)
    diff_min = np.min(np.array(diff))
    diff_max = np.max(np.array(diff))

    #print('clean min, q1, q2, q3, max: {} {} {} {} {}'.format(clean_min, clean_q1, clean_q2, clean_q3, clean_max))
    #print('uap min, q1, q2, q3, max: {} {} {} {} {}'.format(uap_min, uap_q1, uap_q2, uap_q3, uap_max))

    print('clean:')
    print('lower whisker={:.2f},'.format(clean_min))
    print('lower quartile={:.2f},'.format(clean_q1))
    print('median={:.2f},'.format(clean_q2))
    print('upper quartile={:.2f},'.format(clean_q3))
    print('upper whisker={:.2f},'.format(clean_max))

    print('uap:')
    print('lower whisker={:.2f},'.format(uap_min))
    print('lower quartile={:.2f},'.format(uap_q1))
    print('median={:.2f},'.format(uap_q2))
    print('upper quartile={:.2f},'.format(uap_q3))
    print('upper whisker={:.2f},'.format(uap_max))
    '''
    print('diff:')
    print('lower whisker={:.2f},'.format(diff_min))
    print('lower quartile={:.2f},'.format(diff_q1))
    print('median={:.2f},'.format(diff_q2))
    print('upper quartile={:.2f},'.format(diff_q3))
    print('upper whisker={:.2f},'.format(diff_max))
    '''

def analyze_layers(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
    ####################################
    # Init model, criterion, and optimizer
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    network = get_network(args.arch,
                          input_size=input_size,
                          num_classes=num_classes,
                          finetune=False)

    # Set the target model into evaluation mode
    network.eval()

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            network.load_state_dict(new_state_dict)

    elif args.dataset == 'eurosat':
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)
    total_params = get_num_parameters(network)

    if args.use_cuda:
        network.cuda()

    uap = None
    if args.analyze_clean == 0:
        _, data_test = get_data(args.dataset, args.dataset)

        data_test_loader = torch.utils.data.DataLoader(data_test,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=args.workers,
                                                       pin_memory=True)
        #load uap
        if args.causal_type != 'inact':
            uap_path = get_uap_path(uap_data=args.dataset,
                                    model_data=args.dataset,
                                    network_arch=args.arch,
                                    random_seed=args.seed)
            uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
            uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
            uap = torch.from_numpy(uap)

        # perform causality analysis
        attribution_map, outputs, clean_outputs = solve_causal_single(data_test_loader, network, uap, args.arch,
                                                                      split_layer=args.split_layer,
                                                                      targeted=args.targeted,
                                                                      target_class=args.target_class,
                                                                      num_sample=args.num_iterations,
                                                                      causal_type=args.causal_type,
                                                                      log=None,
                                                                      use_cuda=args.use_cuda)

        #save multiple maps
        attribution_path = get_attribution_path()
        for i in range(0, len(attribution_map)):
            attribution_map_ = attribution_map[i]
            uap_fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + "_s" + str(i)
                                  + ".npy")#'_' + str(outputs[i]) + ".npy")
            np.save(uap_fn, attribution_map_)
        output_fn = os.path.join(attribution_path, "uap_clean_outputs_" + str(args.split_layer) + ".npy")
        np.save(output_fn, clean_outputs)
    elif args.analyze_clean == 1:
        _, data_test = get_data_class(args.dataset, args.target_class)
        if len(data_test) == 0:
            print('No sample from class {}'.format(args.target_class))
            return
        data_test_loader = torch.utils.data.DataLoader(data_test,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=args.workers,
                                                       pin_memory=True)

        attribution_map, outputs, clean_outputs = solve_causal_single(data_test_loader, network, None, args.arch,
                                                                      split_layer=args.split_layer,
                                                                      targeted=args.targeted,
                                                                      target_class=args.target_class,
                                                                      num_sample=args.num_iterations,
                                                                      causal_type=args.causal_type,
                                                                      log=None,
                                                                      use_cuda=args.use_cuda)

        attribution_path = get_attribution_path()
        for i in range(0, len(attribution_map)):
            attribution_map_ = attribution_map[i]
            uap_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + "_s" + str(i)
                                  + ".npy")#'_' + str(outputs[i]) + ".npy")
            np.save(uap_fn, attribution_map_)
        output_fn = os.path.join(attribution_path, "clean_outputs_" + str(args.split_layer) + ".npy")
        np.save(output_fn, clean_outputs)
    elif args.analyze_clean == 2: #analyze uap only
        uap_path = get_uap_path(uap_data=args.dataset,
                                model_data=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
        uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
        uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
        uap = torch.from_numpy(uap)
        attribution_map, outputs = solve_causal_uap(network, uap, args.arch,
                                                    split_layer=args.split_layer,
                                                    causal_type=args.causal_type,
                                                    use_cuda=args.use_cuda)

        attribution_path = get_attribution_path()
        for i in range(0, len(attribution_map)):
            attribution_map_ = attribution_map[i]
            uap_fn = os.path.join(attribution_path, "uaponly_attribution_" + str(args.split_layer) + ".npy")
            np.save(uap_fn, attribution_map_)
        calc_entropy_uap(args)

    return


def analyze_layers_clean(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    #load uap
    uap = None

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    ####################################
    # Init model, criterion, and optimizer
    #print("=> Creating model '{}'".format(args.arch))
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    network = get_network(args.arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    #print("=> Network :\n {}".format(network))

    # Set the target model into evaluation mode
    network.eval()

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            network.load_state_dict(new_state_dict)

    elif args.dataset == 'eurosat':
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)
    total_params = get_num_parameters(network)

    #print("Filter Network Total # parameters: {}".format(total_params))

    if args.use_cuda:
        network.cuda()

    _, data_test = get_data_class(args.dataset, args.target_class)
    #print('Number of training samples in this class: {}'.format(len(data_train)))

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    attribution_map = solve_causal(data_test_loader, network, None, args.arch,
                                   split_layer=args.split_layer,
                                   targeted=args.targeted,
                                   target_class=args.target_class,
                                   num_sample=args.num_iterations,
                                   causal_type=args.causal_type,
                                   log=None,
                                   use_cuda=args.use_cuda)

    attribution_path = get_attribution_path()
    file_name = "clean_attribution_" + str(args.split_layer) + '_' + str(args.target_class) + "_avg.npy"
    uap_fn = os.path.join(attribution_path, file_name)
    np.save(uap_fn, attribution_map)

    return


def calc_entropy(args):
    print('idx is {}'.format(args.idx))
    attribution_path = get_attribution_path()
    clean_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer)
                            + '_' + str(args.target_class) + "_avg.npy")
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    if args.analyze_clean == 0:
        fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                          str(args.idx) + ".npy")#'_' + str(args.target_class) + ".npy")
    elif args.analyze_clean == 1:
        fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + '_s' +
                          str(args.idx) + ".npy")#'_' + str(args.target_class) + ".npy")

    loaded = np.load(fn)

    if args.causal_type == 'logit':
        uap_ca = loaded[:, 1]
    elif args.causal_type == 'act':
        uap_ca = loaded.transpose()

    clean_h = calculate_shannon_entropy_array(clean_ca)
    uap_h = calculate_shannon_entropy_array(uap_ca)

    print('entropy avg, single: {} {}'.format(clean_h, uap_h))


def calc_entropy_i(i, args):
    attribution_path = get_attribution_path()

    if args.analyze_clean == 0:
        if 'repaired' in args.model_name:
            fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                              str(i) + ".npy")
        else:
            fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                              str(i) + ".npy")#'_' + str(args.target_class) + ".npy")

    elif args.analyze_clean == 1:
        fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + '_s' +
                          str(i) + ".npy")#'_' + str(args.target_class) + ".npy")
    elif args.analyze_clean == 2:
        fn = os.path.join(attribution_path, "uaponly_attribution_" + str(args.split_layer) + ".npy")

    if os.path.exists(fn):
        loaded = np.load(fn)
    else:
        return

    if args.causal_type == 'logit':
        ca = loaded[:, 1]
    elif args.causal_type == 'act':
        ca = loaded.transpose()

    #uap_h = calculate_shannon_entropy_array(ca)
    uap_h = calc_hloss(ca)

    #print('entropy {}: {}'.format(i, uap_h))
    #print('uap_hloss {}: {}'.format(i, uap_hloss))
    return uap_h


def calc_entropy_uap(args):
    attribution_path = get_attribution_path()

    fn = os.path.join(attribution_path, "uaponly_attribution_" + str(args.split_layer) + ".npy")
    if os.path.exists(fn):
        loaded = np.load(fn)
    else:
        return

    if args.causal_type == 'logit':
        ca = loaded[:, 1]
    elif args.causal_type == 'act':
        ca = loaded.transpose()

    # uap_h = calculate_shannon_entropy_array(ca)
    uap_h = calc_hloss(ca)

    print('entropy uap: {}'.format(uap_h))
    # print('uap_hloss {}: {}'.format(i, uap_hloss))
    return uap_h


def calc_entropy_layer(i):
    attribution_path = get_attribution_path()
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
    uap_fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + "_s" + str(i) + ".npy")
    loaded = np.load(uap_fn) / np.array(std).reshape(1,3,1,1)
    uap_ca = loaded[:, 1]

    uap_h = calculate_shannon_entropy_array(uap_ca)

    clean1_fn = os.path.join(attribution_path,
                             "clean_attribution_" + str(args.split_layer) + "_s" + str(i) + ".npy")
    loaded = np.load(clean1_fn)
    clean1_ca = loaded[:, 1]
    clean1_h = calculate_shannon_entropy_array(clean1_ca)
    print('uap_h: {}, clean1_h: {}'.format(uap_h, clean1_h))


def calc_pcc(args):
    #print('idx is {}'.format(args.idx))
    attribution_path = get_attribution_path()
    clean_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer)
                            + '_' + str(args.target_class) + "_avg.npy")
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    if args.analyze_clean == 0:
        fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                          str(args.idx) + ".npy")#'_' + str(args.target_class) + ".npy")
        prefix = 'uap'
    elif args.analyze_clean == 1:
        fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + '_s' +
                          str(args.idx) + ".npy")#'_' + str(args.target_class) + ".npy")
        prefix = 'clean'

    if os.path.exists(fn):
        loaded = np.load(fn)
    else:
        return

    if args.causal_type == 'logit':
        uap_ca = loaded[:, 1]
    elif args.causal_type == 'act':
        uap_ca = loaded.transpose()

    uap_pcc = np.corrcoef(uap_ca, clean_ca)[0, 1]

    print('{} pcc: {}'.format(prefix, uap_pcc))

    return uap_pcc


def calc_pcc_i(i, args):
    attribution_path = get_attribution_path()
    clean_fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer)
                            + '_' + str(args.target_class) + "_avg.npy")
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]

    if args.analyze_clean == 0:
        fn = os.path.join(attribution_path, "uap_attribution_" + str(args.split_layer) + '_s' +
                          str(i) + ".npy")#'_' + str(args.target_class) + ".npy")
    elif args.analyze_clean == 1:
        fn = os.path.join(attribution_path, "clean_attribution_" + str(args.split_layer) + '_s' +
                          str(i) + ".npy")#'_' + str(args.target_class) + ".npy")
    if os.path.exists(fn):
        loaded = np.load(fn)
    else:
        return

    if args.causal_type == 'logit':
        ca = loaded[:, 1]
    elif args.causal_type == 'act':
        ca = loaded.transpose()

    uap_pcc = np.corrcoef(ca, clean_ca)[0, 1]

    #print('pcc {}: {}'.format(i, uap_pcc))
    return uap_pcc


def calc_pcc_i_old(i, args):
    attribution_path = get_attribution_path()

    clean_fn = os.path.join(attribution_path, args.avg_ca_name)
    loaded = np.load(clean_fn)
    clean_ca = loaded[:, -1]
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
    uap_fn = os.path.join(attribution_path, args.uap_ca_name)
    loaded = np.load(uap_fn) / np.array(std).reshape(1,3,1,1)
    uap_ca = loaded[:, 1]

    uap_pcc = np.corrcoef(uap_ca, clean_ca)[0, 1]

    clean1_fn = os.path.join(attribution_path, args.clean_ca_name)
    loaded = np.load(clean1_fn)
    clean1_ca = loaded[:, 1]
    clean1_pcc = np.corrcoef(clean1_ca, clean1_ca)[0, 1]
    print('{}: {}, {}'.format(i, uap_pcc, clean1_pcc))


def test(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    #load uap
    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=args.arch,
                            random_seed=args.seed)
    uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
    uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
    uap = torch.from_numpy(uap)

    ####################################
    # Init model, criterion, and optimizer
    #print("=> Creating model '{}'".format(args.arch))
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    network = get_network(args.arch,
                          input_size=input_size,
                          num_classes=num_classes,
                          finetune=False)

    #print("=> Network :\n {}".format(network))

    # Set the target model into evaluation mode
    network.eval()

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            network.load_state_dict(new_state_dict)

    elif args.dataset == 'eurosat':
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)

    if args.use_cuda:
        network.cuda()

    data_train, data_test = get_data(args.dataset, args.dataset)

    if args.is_train:
        dataset = data_train
    else:
        dataset = data_test
    print('[DEBUG] test length: {}'.format(len(dataset)))

    data_test_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    _, acc, _, fr, _, asr = my_test_uap(data_test_loader, network, uap, args.target_class, args.num_iterations,
                      use_cuda=args.use_cuda)
    print('overall acc {}'.format(acc))
    print('overall fooling ratio {}'.format(fr))
    print('overall asr {}'.format(asr))
    '''
    tot_correct = 0
    tot_num = 0
    for cur_class in range(0, 1000):
        data_train, data_test = get_data_class(args.dataset, cur_class)
        if len(data_test) == 0:
            continue
        data_test_loader = torch.utils.data.DataLoader(data_test,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.workers,
                                                        pin_memory=True)

        corr, _, fool, _, num = my_test_uap(data_test_loader, network, uap, args.batch_size, args.num_iterations
                               use_cuda=args.use_cuda)
        print('class {}, correct {}, fool {}, num {}'.format(cur_class, corr, fool, num))
        tot_correct += corr
        tot_num += num
    print('Model accuracy: {}%'.format(tot_correct / tot_num * 100))
    '''
    return


def process_pcc(args):
    uap_pccs = []
    clean_pccs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_pcc = calc_pcc_i(i, args)
        if uap_pcc is not None:
            uap_pccs.append(uap_pcc)
            print('uap_pcc: {}'.format(uap_pcc))

    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_pcc = calc_pcc_i(i, args)
        if clean_pcc is not None:
            clean_pccs.append(clean_pcc)
            print('clean_pcc: {}'.format(clean_pcc))
    return


def process_entropy(args):
    uap_hs = []
    clean_hs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_h = calc_entropy_i(i, args)
        if uap_h is not None:
            uap_hs.append(uap_h)
            print('uap_h: {}'.format(uap_h))

    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_h = calc_entropy_i(i, args)
        if clean_h is not None:
            clean_hs.append(clean_h)
            print('clean_h: {}'.format(clean_h))
    return


def uap_classification_avg(args):
    #get average entropy of clean data
    clean_hs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_h = calc_entropy_i(i, args)
        if clean_h is not None:
            clean_hs.append(clean_h)
            #print('clean_h: {}'.format(clean_h))
    clean_hs_avg = np.mean(np.array(clean_hs))

    #get entropy of the test sample
    h_result = []
    uap_hs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_h = calc_entropy_i(i, args)
        if uap_h is not None:
            uap_hs.append(uap_h)
            h_result.append(int(uap_h > clean_hs_avg))
            #print('uap_h: {}'.format(uap_h))
    print('Layer {} entropy result[{}]: {}'.format(args.split_layer, len(h_result), h_result))

    #get average pcc of clean data
    clean_pccs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_pcc = calc_pcc_i(i, args)
        if clean_pcc is not None:
            clean_pccs.append(clean_pcc)
            #print('clean_pcc: {}'.format(clean_pcc))
    clean_pcc_avg = np.mean(np.array(clean_pccs))

    #get pcc of the test sample
    pcc_result = []
    uap_pccs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_pcc = calc_pcc_i(i, args)
        if uap_pcc is not None:
            uap_pccs.append(uap_pcc)
            pcc_result.append(int(uap_pcc < clean_pcc_avg))
            #print('uap_pcc: {}'.format(uap_pcc))
    print('Layer {} pcc result[{}]    : {}'.format(args.split_layer, len(pcc_result), pcc_result))
    return np.sum(np.logical_and(np.array(h_result) == 1, np.array(pcc_result) == 1)) / len(pcc_result) * 100


def uap_classification(args):
    #get average entropy of clean data
    clean_hs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_h = calc_entropy_i(i, args)
        if clean_h is not None:
            clean_hs.append(clean_h)
            print('clean_h: {}'.format(clean_h))
    #print('clean_hs : {}'.format(clean_hs))
    clean_hs_avg = np.mean(np.array(clean_hs))
    print('clean_hs_avg : {}'.format(clean_hs_avg))

    #get entropy of the test sample
    h_result = []
    uap_hs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_h = calc_entropy_i(i, args)
        if uap_h is not None:
            uap_hs.append(uap_h)
            print('uap_h: {}'.format(uap_h))
            top = outlier_detection((clean_hs + [uap_h]), max(clean_hs + [uap_h]), verbose=False, th=args.th)
            outliers = [x[0] for x in top]
            h_result.append(int((len(clean_hs + [uap_h]) - 1) in outliers))
            #print('Outliers: {}, uap index: {}'.format(top, len(clean_hs + [uap_h]) - 1))
    #print('uap_hs : {}'.format(uap_hs))
    uap_hs_avg = np.mean(np.array(uap_hs))
    print('uap_hs_avg : {}'.format(uap_hs_avg))
    #print('Layer {} entropy result[{}]: {}'.format(args.split_layer, len(h_result), h_result))

    #get average pcc of clean data
    clean_pccs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_pcc = calc_pcc_i(i, args)
        if clean_pcc is not None:
            clean_pccs.append(clean_pcc)
            #print('clean_pcc: {}'.format(clean_pcc))
    clean_pcc_avg = np.mean(np.array(clean_pccs))

    #get pcc of the test sample
    pcc_result = []
    uap_pccs = []
    args.analyze_clean = 0
    for i in range(0, args.num_iterations):
        uap_pcc = calc_pcc_i(i, args)
        if uap_pcc is not None:
            uap_pccs.append(uap_pcc)
            reversed_list = max(clean_pccs + [uap_pcc]) - np.array(clean_pccs + [uap_pcc])
            top = outlier_detection(reversed_list, max(clean_pccs + [uap_pcc]), verbose=False, th=args.th)
            outliers = [x[0] for x in top]
            pcc_result.append(int((len(clean_pccs + [uap_pcc]) - 1) in outliers))
            #print('Outliers: {}, uap index: {}'.format(top, len(clean_pccs + [uap_pcc]) - 1))
    #print('Layer {} pcc result[{}]    : {}'.format(args.split_layer, len(pcc_result), pcc_result))
    #return np.sum(np.logical_and(np.array(h_result) == 1, np.array(pcc_result) == 1)) / len(pcc_result) * 100
    return 0


def uap_repair(args):
    _, data_test = get_data(args.dataset, args.dataset)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    data_train, _ = get_data(args.dataset, args.dataset)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=args.arch,
                            random_seed=args.seed)
    uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
    uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
    uap = torch.from_numpy(uap)

    ####################################
    # Init model, criterion, and optimizer
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    target_network = get_network(args.arch,
                                 input_size=input_size,
                                 num_classes=num_classes,
                                 finetune=False)

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in target_network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            target_network.load_state_dict(new_state_dict)

    elif args.dataset == 'eurosat':
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    #non_trainale_params = get_num_non_trainable_parameters(target_network)
    trainale_params = get_num_trainable_parameters(target_network)
    total_params = get_num_parameters(target_network)
    print("Target Network Trainable parameters: {}".format(trainale_params))
    print("Target Network Total # parameters: {}".format(total_params))

    target_network.train()

    if args.loss_function == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_function == "neg_ce":
        criterion = NegativeCrossEntropy()
    elif args.loss_function == "logit":
        criterion = LogitLoss(num_classes=num_classes, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit":
        criterion = BoundedLogitLoss(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit_fixed_ref":
        criterion = BoundedLogitLossFixedRef(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit_neg":
        criterion = BoundedLogitLoss_neg(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    else:
        raise ValueError

    print('Criteria: {}'.format(criterion))

    if args.use_cuda:
        target_network.cuda()
        criterion.cuda()

    optimizer = torch.optim.SGD(target_network.parameters(), lr=args.learning_rate, momentum=0.9)
    #'''
    # Measure the time needed for the UAP generation
    start = time.time()

    if 'ae' in args.option:
        repaired_network = adv_train(data_train_loader,
                                     target_network,
                                     args.arch,
                                     criterion,
                                     optimizer,
                                     args.num_iterations,
                                     args.split_layers,
                                     uap=uap,
                                     num_batches=args.num_batches,
                                     alpha=args.alpha,
                                     use_cuda=args.use_cuda,
                                     adv_itr=args.ae_iter,
                                     eps=args.epsilon,
                                     mean=mean,
                                     std=std)
        post_fix = 'ae'
        '''
        repaired_network = adv_train(data_train_loader,
                             target_network,
                             args.target_class,
                             criterion,
                             optimizer,
                             args.num_iterations,
                             args.split_layers,
                             uap=uap,
                             num_batches=args.num_batches,
                             alpha=args.alpha,
                             use_cuda=args.use_cuda,
                             adv_itr=args.ae_iter,
                             eps=args.epsilon,
                             mean=mean,
                             std=std)
        '''
        #post_fix = 'pgd_' + str(args.target_class)
    elif 'uap' in args.option:
        train_uaps = None
        for target_i in [755,743,804,700,922,174,547,369]:
            for idx in range(0, 10):
                train_uap = 'uap_train_' + str(target_i) + '_' + str(idx) + '.npy'
                uap_fn = os.path.join(uap_path, train_uap)
                if train_uaps == None:
                    train_uaps = torch.from_numpy(np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1))
                else:
                    train_uaps = torch.cat((train_uaps, torch.from_numpy(np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1))))

        repaired_network = known_uap_train(data_train_loader,
                                           target_network,
                                           args.arch,
                                           criterion,
                                           optimizer,
                                           args.num_iterations,
                                           args.split_layers,
                                           train_uaps,
                                           alpha=args.alpha,
                                           use_cuda=args.use_cuda)
        post_fix = 'uap'# + str(args.target_class)
    elif 'enpool' in args.option:
        repaired_network = replace_model(target_network, args.arch, replace_layer=args.split_layers[0])
        print("=> repaired_network :\n {}".format(repaired_network))
        post_fix = 'enpool'
    elif 'enrep' in args.option:
        target_network = replace_model(target_network, args.arch, replace_layer=args.split_layers[0])
        repaired_network = adv_ae_train(data_train_loader,
                                        target_network,
                                        args.arch,
                                        criterion,
                                        optimizer,
                                        args.num_iterations,
                                        args.split_layers,
                                        uap=uap,
                                        std=std,
                                        alpha=args.alpha,
                                        ae_alpha=args.ae_alpha,
                                        print_freq=args.print_freq,
                                        use_cuda=args.use_cuda,
                                        adv_itr=args.ae_iter,
                                        eps=args.epsilon)
        post_fix = 'enrep'
    else:
        #fine tune with clean sample only
        repaired_network = train_repair(data_loader=data_train_loader,
                                        model=target_network,
                                        arch=args.arch,
                                        criterion=criterion,
                                        optimizer=optimizer,
                                        num_iterations=args.num_iterations,
                                        num_batches=args.num_batches,
                                        print_freq=args.print_freq,
                                        use_cuda=args.use_cuda)
        post_fix = 'finetuned'

    end = time.time()
    print("Time needed for UAP repair: {}".format(end - start))

    #eval
    if args.use_cuda:
        uap = uap.cuda()
    metrics_evaluate_test(data_loader=data_test_loader,
                          target_model=repaired_network,
                          uap=uap,
                          targeted=args.targeted,
                          target_class=args.target_class,
                          log=None,
                          use_cuda=args.use_cuda)

    model_repaired_path = os.path.join(model_path, args.arch + '_' + args.dataset + '_' + post_fix + '_repaired.pth')

    torch.save(repaired_network, model_repaired_path)
    print('repaired model saved to {}'.format(model_repaired_path))

    _, acc, _, fr, _, asr = my_test_uap(data_test_loader, repaired_network, uap, args.target_class, 2000,
                      use_cuda=args.use_cuda)
    print('overall acc {}'.format(acc))
    print('overall fooling ratio {}'.format(fr))
    print('overall asr {}'.format(asr))


def uap_gen_low_en_sample(args):
    _, data_test = get_data(args.dataset, args.dataset)

    data_test_loader = torch.utils.data.DataLoader(data_test,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.workers,
                                                   pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)

    data_train, _ = get_data(args.dataset, args.dataset)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=args.arch,
                            random_seed=args.seed)
    uap_fn = os.path.join(uap_path, 'uap_' + str(args.target_class) + '.npy')
    uap = np.load(uap_fn) / np.array(std).reshape(1, 3, 1, 1)
    uap = torch.from_numpy(uap)

    ####################################
    # Init model, criterion, and optimizer
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.dataset,
                                network_arch=args.arch,
                                random_seed=args.seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    target_network = get_network(args.arch,
                                 input_size=input_size,
                                 num_classes=num_classes,
                                 finetune=False)

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in target_network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            target_network.load_state_dict(new_state_dict)

    elif args.datasetdataset == 'eurosat':
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    #non_trainale_params = get_num_non_trainable_parameters(target_network)
    trainale_params = get_num_trainable_parameters(target_network)
    total_params = get_num_parameters(target_network)
    print("Target Network Trainable parameters: {}".format(trainale_params))
    print("Target Network Total # parameters: {}".format(total_params))

    target_network.train()

    if args.loss_function == "ce":
        criterion = torch.nn.CrossEntropyLoss()
    elif args.loss_function == "neg_ce":
        criterion = NegativeCrossEntropy()
    elif args.loss_function == "logit":
        criterion = LogitLoss(num_classes=num_classes, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit":
        criterion = BoundedLogitLoss(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit_fixed_ref":
        criterion = BoundedLogitLossFixedRef(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    elif args.loss_function == "bounded_logit_neg":
        criterion = BoundedLogitLoss_neg(num_classes=num_classes, confidence=args.confidence, use_cuda=args.use_cuda)
    else:
        raise ValueError

    print('Criteria: {}'.format(criterion))

    if args.use_cuda:
        target_network.cuda()
        criterion.cuda()

    optimizer = torch.optim.SGD(target_network.parameters(), lr=args.learning_rate, momentum=0.9)
    #'''
    delta = gen_low_entropy_sample(data_train_loader,
                                   target_network,
                                   args.arch,
                                   criterion,
                                   args.split_layers,
                                   use_cuda=args.use_cuda,
                                   adv_itr=args.ae_iter,
                                   eps=args.epsilon)

    np.save(model_path + '/' + 'ae', delta.cpu().detach().numpy())

def clean_classification(args):
    #get average entropy of clean data
    clean_hs = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        clean_h = calc_entropy_i(i, args)
        if clean_h is not None:
            clean_hs.append(clean_h)
            #print('clean_h: {}'.format(clean_h))
    clean_hs_avg = np.mean(np.array(clean_hs))

    #get entropy of the test sample
    h_result = []
    uap_hs = []
    for i in range(0, args.num_iterations):
        uap_h = calc_entropy_i(i, args)
        if uap_h is not None:
            uap_hs.append(uap_h)
            reversed_list = max(clean_hs + [uap_h]) - np.array(clean_hs + [uap_h])
            top = outlier_detection(reversed_list, max(clean_hs + [uap_h]), verbose=False, th=args.th)
            outliers = [x[0] for x in top]
            h_result.append(int((len(clean_hs + [uap_h]) - 1) in outliers))
            #print('Outliers: {}, uap index: {}'.format(top, len(clean_hs + [uap_h]) - 1))
    #print('Layer {} entropy result[{}]: {}'.format(args.split_layer, len(h_result), h_result))

    #get average pcc of clean data
    clean_pccs = []
    for i in range(0, args.num_iterations):
        clean_pcc = calc_pcc_i(i, args)
        if clean_pcc is not None:
            clean_pccs.append(clean_pcc)
            #print('clean_pcc: {}'.format(clean_pcc))
    clean_pcc_avg = np.mean(np.array(clean_pccs))

    #get pcc of the test sample
    pcc_result = []
    uap_pccs = []
    for i in range(0, args.num_iterations):
        uap_pcc = calc_pcc_i(i, args)
        if uap_pcc is not None:
            uap_pccs.append(uap_pcc)
            top = outlier_detection((clean_pccs + [uap_pcc]), max(clean_pccs + [uap_pcc]), verbose=False, th=args.th)
            outliers = [x[0] for x in top]
            pcc_result.append(int((len(clean_pccs + [uap_pcc]) - 1) in outliers))
            #print('Outliers: {}, uap index: {}'.format(top, len(clean_pccs + [uap_pcc]) - 1))
    #print('Layer {} pcc result[{}]    : {}'.format(args.split_layer, len(pcc_result), pcc_result))
    return np.sum(np.logical_and(np.array(h_result) == 1, np.array(pcc_result) == 1) / len(pcc_result)) * 100


if __name__ == '__main__':
    args = parse_arguments()
    state = {k: v for k, v in args._get_kwargs()}
    start = time.time()
    '''
    attribution_path = get_attribution_path()
    output_fn = os.path.join(attribution_path, "uap_clean_outputs_" + str(args.split_layer) + ".npy")
    clean_outputs = np.load(output_fn)
    print(clean_outputs)


    for key, value in state.items():
        print("{} : {}".format(key, value))
    '''
    if args.option == 'analyze_inputs':
        analyze_inputs(args)
    elif args.option == 'calc_entropy':
        if args.num_iterations != 0:
            for i in range(0, args.num_iterations):
                calc_entropy_i(i, args)
        else:
            calc_entropy(args)
    elif args.option == 'calc_pcc':
        if args.num_iterations != 0:
            for i in range(0, args.num_iterations):
                calc_pcc_i(i, args)
        else:
            calc_pcc(args)
    elif args.option == 'analyze_layers':
        analyze_layers(args)
    elif args.option == 'analyze_clean':
        analyze_layers_clean(args)
    elif args.option == 'test':
        test(args)
    elif args.option == 'pcc':
        process_pcc(args)
    elif args.option == 'entropy':
        process_entropy(args)
    elif args.option == 'classify':
        tpr = uap_classification(args)
        fpr = clean_classification(args)
        #print('TPR: {}, FPR: {}'.format(tpr, fpr))
    elif 'analyze_entropy' in args.option:
        analyze_entropy(args)
    elif 'repair' in args.option:
        uap_repair(args)
    elif args.option == 'gen_en_sample':
        uap_gen_low_en_sample(args)
    end = time.time()
    #print('Process time: {}'.format(end - start))

