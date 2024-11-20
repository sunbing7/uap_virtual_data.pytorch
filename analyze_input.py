from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn

from utils.data import get_data_specs, get_data, get_data_class, Normalizer
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
    parser.add_argument('--option', default='analyze_entropy', choices=['repair_ae', 'analyze_entropy',
                                                                                     'repair', 'repair_uap'],
                        help='Run options')
    parser.add_argument('--causal_type', default='act', choices=['act'],
                        help='Causality analysis type (default: act)')
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'caltech', 'asl', 'eurosat', 'cifar10'],
                        help='Used dataset to generate UAP (default: imagenet)')
    parser.add_argument('--arch', default='resnet50',
                        choices=['googlenet', 'vgg19', 'resnet50', 'shufflenetv2', 'mobilenet', 'wideresnet', 'resnet110'])
    parser.add_argument('--model_name', type=str, default='vgg19.pth',
                        help='model name')
    parser.add_argument('--seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--split_layer', type=int, default=43,
                        help='causality analysis layer (default: 43)')
    parser.add_argument('--split_layers', type=int, nargs="*", default=[43])
    # Parameters regarding UAP
    parser.add_argument('--num_iterations', type=int, default=32,
                        help='Number of iterations (default: 32)')
    parser.add_argument('--num_batches', type=int, default=1500)
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

    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of used GPUs (0 = CPU) (default: 1)')

    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')

    parser.add_argument('--analyze_clean', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=0.1)

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
        if 'repaired' in args.model_name:
            adaptive = '_adaptive'
    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))
    elif args.pretrained_dataset == "cifar10":
        if 'repaired' in args.model_name:
            network = torch.load(model_weights_path, map_location=torch.device('cpu'))
            adaptive = '_adaptive'
        else:
            if args.arch == 'resnet110':
                sd0 = torch.load(model_weights_path)['state_dict']
                network.load_state_dict(sd0, strict=True)
            else:
                network.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    # Set all weights to not trainable
    set_parameter_requires_grad(network, requires_grad=False)

    if args.arch == 'resnet110':
        # Normalization wrapper, so that we don't have to normalize adversarial perturbations
        normalize = Normalizer(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        target_network = nn.Sequential(normalize, network)

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
    if args.targeted:
        target_name = str(args.target_class)
    else:
        target_name = 'nontarget'

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.dataset,
                            network_arch=args.arch,
                            random_seed=args.seed)
    uap_fn = os.path.join(uap_path, 'uap_' + target_name + '.npy')
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

    attribution_map, outputs = solve_causal_uap(network, uap, args.arch,
                                                split_layer=args.split_layer,
                                                causal_type=args.causal_type,
                                                use_cuda=args.use_cuda)

    for i in range(0, len(attribution_map)):
        attribution_map_ = attribution_map[i]
        uap_fn = os.path.join(attribution_path, "uaponly_attribution_" + str(args.split_layer) + ".npy")
        np.save(uap_fn, attribution_map_)

    ##################################################################################
    # anlayze entropy
    clean_hs = []
    uap_hs = []
    #diff = []
    args.analyze_clean = 1
    for i in range(0, args.num_iterations):
        args.analyze_clean = 1
        clean_h = calc_entropy_i(i, args)
        clean_hs.append(clean_h)
        #args.analyze_clean = 0
        #uap_h = calc_entropy_i(i, args)
        #uap_hs.append(uap_h)
        #diff.append(abs(clean_h - uap_h))
        print('({}, {})'.format(i, clean_h))

    for i in range(0, args.num_iterations):
        args.analyze_clean = 0
        uap_h = calc_entropy_i(i, args)
        uap_hs.append(uap_h)
        print('({}, {})'.format(i, uap_h))

    args.analyze_clean = 2
    uaponly_h = calc_entropy_i(0, args)
    print('uaponly_h: {}'.format(uaponly_h))

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
    '''
    diff_q1 = np.quantile(np.array(diff), 0.25)
    diff_q2 = np.quantile(np.array(diff), 0.5)
    diff_q3 = np.quantile(np.array(diff), 0.75)
    diff_min = np.min(np.array(diff))
    diff_max = np.max(np.array(diff))
    '''
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

    ca = loaded.transpose()

    #uap_h = calculate_shannon_entropy_array(ca)
    uap_h = calc_hloss(ca)

    #print('entropy {}: {}'.format(i, uap_h))
    #print('uap_hloss {}: {}'.format(i, uap_hloss))
    return uap_h


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
        if 'repaired' in args.model_name:
            adaptive = '_adaptive'
    # Imagenet models use the pretrained pytorch weights
    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
    elif args.dataset == "cifar10":
        if 'repaired' in args.model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
            adaptive = '_adaptive'
        else:
            if args.arch == 'resnet110':
                sd0 = torch.load(model_weights_path)['state_dict']
                target_network.load_state_dict(sd0, strict=True)
            else:
                target_network.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))

    if args.arch == 'resnet110':
        target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))
        # Normalization wrapper, so that we don't have to normalize adversarial perturbations
        normalize = Normalizer(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        target_network = nn.Sequential(normalize, target_network)

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


if __name__ == '__main__':
    args = parse_arguments()
    state = {k: v for k, v in args._get_kwargs()}
    start = time.time()
    if 'analyze_entropy' in args.option:
        analyze_entropy(args)
    elif 'repair' in args.option:
        uap_repair(args)
    end = time.time()
    #print('Process time: {}'.format(end - start))

