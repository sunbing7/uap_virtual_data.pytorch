from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict
from utils.training import ae_training_pgd_tgt

from networks.uap import UAP
from utils.data import get_data_specs, get_data
from utils.utils import get_model_path, get_result_path, get_uap_path
from utils.utils import print_log
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import train, save_checkpoint, metrics_evaluate, eval_uap
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


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


def main(args):
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
                                                    shuffle=False,
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

    #non_trainale_params = get_num_non_trainable_parameters(target_network)
    trainale_params = get_num_trainable_parameters(target_network)
    total_params = get_num_parameters(target_network)
    print("Target Network Trainable parameters: {}".format(trainale_params))
    print("Target Network Total # parameters: {}".format(total_params))

    target_network.eval()

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

    num_batch = 0
    for input, target in data_train_loader:
        if num_batch > args.num_batches:
            break

        if args.use_cuda:
            target = target.cuda()
            input = input.cuda()

        '''
        #untarget
        delta = ae_training_pgd(model,
                                input,
                                target,
                                criterion,
                                adv_itr,
                                eps,
                                True,
                                mean,
                                std,
                                use_cuda)
        '''
        #targeted
        target_class = torch.ones_like(target) * args.target_class

        delta = ae_training_pgd_tgt(target_network,
                                    input,
                                    target_class,
                                    criterion,
                                    args.ae_iter,
                                    args.epsilon,
                                    True,
                                    mean,
                                    std,
                                    args.use_cuda)
        x_adv = input + delta
        delta_output = target_network(x_adv)
        adv_pred = torch.argmax(delta_output, dim=-1)
        #if(adv_pred == target_class):
        print('AE generated for data point {}; with original class {}; prediction class {}'.format(
            num_batch, target.cpu().detach().numpy(), adv_pred.cpu().detach().numpy()))
        np.save(uap_path + '/ae_tgt_' + str(args.target_class) +
                '_' + str(num_batch) + '.npy', delta.cpu().detach().numpy())
        num_batch = num_batch + 1
    end = time.time()
    print("Time needed for generating AE: {}".format(end - start))

    #eval



if __name__ == '__main__':
    args = parse_arguments()
    main(args)
