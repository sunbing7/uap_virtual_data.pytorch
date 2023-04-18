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
from utils.utils import get_model_path, get_result_path, get_neuron_path, get_uap_path, get_neuron_name
from utils.utils import print_log
from utils.network import get_network, set_parameter_requires_grad
from utils.network import get_num_parameters, get_num_non_trainable_parameters, get_num_trainable_parameters
from utils.training import train, save_checkpoint, metrics_evaluate, reconstruct_model, train_hidden, split_model
from utils.custom_loss import LogitLoss, BoundedLogitLoss, NegativeCrossEntropy, BoundedLogitLossFixedRef, BoundedLogitLoss_neg, HiddenMSELoss

from matplotlib import pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a UAP')
    # dataset used to train UAP
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='Used dataset to generate UAP (default: cifar10)')
    # model used to train UAP
    parser.add_argument('--pretrained_dataset', default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Used dataset to train the initial model (default: cifar10)')
    parser.add_argument('--pretrained_arch', default='alexnet', choices=['vgg16_cifar', 'vgg19_cifar', 'resnet20', 'resnet56',
                                                                       'alexnet', 'googlenet', 'vgg16', 'vgg19',
                                                                       'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                                       'inception_v3'],
                        help='Used model architecture: (default: alexnet)')
    parser.add_argument('--model_name', type=str, default='alexnet_cifar10.pth',
                        help='model name (default: alexnet_cifar10.pth)')
    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--split_layer', type=int, default=43,
                        help='causality analysis layer (default: 43)')
    parser.add_argument('--split_num_n', type=int, default=4096,
                        help='causality analysis layer number of neurons (default: 4096)')
    parser.add_argument('--do_val', type=int, default=1,
                        help='hidden neuron intervention value')
    parser.add_argument('--rec_type', type=str, default='mask',
                        help='reconstruct model type (default: mask)')
    # Parameters regarding UAP
    parser.add_argument('--epsilon', type=float, default=0.03922,
                        help='Norm restriction of UAP (default: 10/255)')
    parser.add_argument('--num_iterations', type=int, default=2000,
                        help='Number of iterations (default: 2000)')
    parser.add_argument('--result_subfolder', default='result', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')
    parser.add_argument('--uap_model', type=str, default='checkpoint.pth.tar',
                        help='uap model name (default: checkpoint.pth.tar)')
    parser.add_argument('--uap_name', type=str, default='uap_general.npy',
                        help='uap file name (default: uap_general.npy)')
    # Optimization options
    parser.add_argument('--loss_function', default='bounded_logit_fixed_ref', choices=['ce', 'neg_ce', 'logit', 'bounded_logit',
                                                                  'bounded_logit_fixed_ref', 'bounded_logit_neg', 'mse'],
                        help='Used loss function for source classes: (default: bounded_logit_fixed_ref)')
    parser.add_argument('--confidence', default=0., type=float,
                        help='Confidence value for C&W losses (default: 0.0)')
    parser.add_argument('--targeted',  default='True',
                        help='Target a specific class (default: True)')
    parser.add_argument('--target_class', type=int, default=7,
                        help='Target class (default: 7)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning Rate (default: 0.001)')
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
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
    # Fix labels if needed
    if args.is_nips:
        print('is_nips')
        pretrained_data_test = fix_labels_nips(pretrained_data_test, pytorch=True)
    pretrained_data_test_loader = torch.utils.data.DataLoader(pretrained_data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)

    data_train, _ = get_data(args.dataset, args.pretrained_dataset)

    if args.dataset == "imagenet":
        data_train = fix_labels(data_train)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    ####################################
    # Init model, criterion, and optimizer
    print_log("=> Creating model '{}'".format(args.pretrained_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.pretrained_dataset,
                                network_arch=args.pretrained_arch,
                                random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    target_network = get_network(args.pretrained_arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    print_log("=> Network :\n {}".format(target_network), log)
    target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))
    # Set the target model into evaluation mode
    target_network.eval()
    # Imagenet models use the pretrained pytorch weights
    if args.pretrained_dataset != "imagenet":
        #network_data = torch.load(model_weights_path, map_location=torch.device('cpu'))
        #target_network.load_state_dict(network_data['state_dict'])
        #target_network.load_state_dict(network_data.state_dict())
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    # add mask based on outstanding neuron
    neuron_path = get_neuron_path()
    neu_idx_list = []
    for fn in os.listdir(neuron_path):
        neu_idx = np.zeros(args.split_num_n)
        each_neuron_path = os.path.join(neuron_path, fn)
        each_neuron = np.load(each_neuron_path)
        neu_idx[each_neuron.astype(int)] = 1
        neu_idx_list.append(neu_idx)

    neu_idx = np.ones(args.split_num_n)
    for idx in neu_idx_list:
        neu_idx = list_and(neu_idx, np.array(idx))
    print('Number of common outstanding neurons: {}'.format(np.sum(neu_idx == 1)))
    neu_idx = torch.from_numpy(neu_idx)
    if args.use_cuda:
        neu_idx = neu_idx.cuda()
    #update the network
    target_network, num_classes = reconstruct_model(target_network, args.pretrained_arch, neu_idx, split_layer=args.split_layer, rec_type=args.rec_type)


    #test
    #for input, gt in pretrained_data_test_loader:
    #    res_output = reconstructed_network(input)
    #    clean_output = target_network(input)

    # Set all weights to not trainable
    set_parameter_requires_grad(target_network, requires_grad=False)

    non_trainale_params = get_num_non_trainable_parameters(target_network)
    trainale_params = get_num_trainable_parameters(target_network)
    total_params = get_num_parameters(target_network)
    print_log("Target Network Trainable parameters: {}".format(trainale_params), log)
    print_log("Target Network Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Target Network Total # parameters: {}".format(total_params), log)

    print_log("=> Inserting Generator", log)

    generator = UAP(shape=(input_size, input_size),
                num_channels=num_channels,
                mean=mean,
                std=std,
                use_cuda=args.use_cuda)

    print_log("=> Generator :\n {}".format(generator), log)
    non_trainale_params = get_num_non_trainable_parameters(generator)
    trainale_params = get_num_trainable_parameters(generator)
    total_params = get_num_parameters(generator)
    print_log("Generator Trainable parameters: {}".format(trainale_params), log)
    print_log("Generator Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Generator Total # parameters: {}".format(total_params), log)

    perturbed_net = nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_network)]))
    perturbed_net = torch.nn.DataParallel(perturbed_net, device_ids=list(range(args.ngpu)))

    non_trainale_params = get_num_non_trainable_parameters(perturbed_net)
    trainale_params = get_num_trainable_parameters(perturbed_net)
    total_params = get_num_parameters(perturbed_net)
    print_log("Perturbed Net Trainable parameters: {}".format(trainale_params), log)
    print_log("Perturbed Net Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Perturbed Net Total # parameters: {}".format(total_params), log)

    # Set the target model into evaluation mode
    perturbed_net.module.target_model.eval()
    perturbed_net.module.generator.train()

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

    if args.use_cuda:
        target_network.cuda()
        generator.cuda()
        perturbed_net.cuda()
        criterion.cuda()

    optimizer = torch.optim.Adam(perturbed_net.parameters(), lr=state['learning_rate'])
    #'''
    # Measure the time needed for the UAP generation
    start = time.time()
    train(data_loader=data_train_loader,
            model=perturbed_net,
            criterion=criterion,
            optimizer=optimizer,
            epsilon=args.epsilon,
            num_iterations=args.num_iterations,
            targeted=args.targeted,
            target_class=args.target_class,
            log=log,
            print_freq=args.print_freq,
            use_cuda=args.use_cuda)
    end = time.time()
    print_log("Time needed for UAP generation: {}".format(end - start), log)
    # evaluate
    print_log("Final evaluation:", log)
    #'''
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
      'optimizer'   : optimizer.state_dict(),
      'args'        : copy.deepcopy(args),
    }, result_path, args.uap_model)

    #export uap and save it
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

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.pretrained_dataset,
                            network_arch=args.pretrained_arch,
                            random_seed=args.pretrained_seed)

    np.save(uap_path + '/' + args.uap_name, tuap.cpu().detach().numpy())
    plt.savefig(uap_path + '/uap_general.png')
    #plt.show()
    print('uap saved!')

    log.close()


def main_hidden():
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
    # Fix labels if needed
    if args.is_nips:
        print('is_nips')
        pretrained_data_test = fix_labels_nips(pretrained_data_test, pytorch=True)

    pretrained_data_test_loader = torch.utils.data.DataLoader(pretrained_data_test,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)

    data_train, _ = get_data(args.dataset, args.pretrained_dataset)

    if args.dataset == "imagenet":
        data_train = fix_labels(data_train)

    data_train_loader = torch.utils.data.DataLoader(data_train,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.workers,
                                                    pin_memory=True)

    ####################################
    # Init model, criterion, and optimizer
    print_log("=> Creating model '{}'".format(args.pretrained_arch), log)
    # get a path for loading the model to be attacked
    model_path = get_model_path(dataset_name=args.pretrained_dataset,
                                network_arch=args.pretrained_arch,
                                random_seed=args.pretrained_seed)
    model_weights_path = os.path.join(model_path, args.model_name)

    target_network = get_network(args.pretrained_arch,
                                input_size=input_size,
                                num_classes=num_classes,
                                finetune=False)

    print_log("=> Network :\n {}".format(target_network), log)
    #target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))
    # Set the target model into evaluation mode
    target_network.eval()
    # Imagenet models use the pretrained pytorch weights
    if args.pretrained_dataset != "imagenet":
        #network_data = torch.load(model_weights_path, map_location=torch.device('cpu'))
        #target_network.load_state_dict(network_data['state_dict'])
        #target_network.load_state_dict(network_data.state_dict())
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    model1, model2 = split_model(target_network, args.pretrained_arch, split_layer=args.split_layer)
    target_network = model1

    # add mask based on outstanding neuron
    neuron_path = get_neuron_path()
    neu_idx_list = []
    for fn in os.listdir(neuron_path):
        neu_idx = np.zeros(args.split_num_n)
        each_neuron_path = os.path.join(neuron_path, fn)
        each_neuron = np.load(each_neuron_path)
        neu_idx[each_neuron.astype(int)] = 1
        neu_idx_list.append(neu_idx)

    neu_idx = np.ones(args.split_num_n)
    for idx in neu_idx_list:
        neu_idx = list_and(neu_idx, np.array(idx))
    print('Number of common outstanding neurons: {}'.format(np.sum(neu_idx == 1)))
    neu_idx = torch.from_numpy(neu_idx)
    #if args.use_cuda:
    #    neu_idx = neu_idx.cuda()
    #update the network
    #target_network, num_classes = reconstruct_model(target_network, args.pretrained_arch, neu_idx, split_layer=args.split_layer, rec_type=args.rec_type)


    #test
    #for input, gt in pretrained_data_test_loader:
    #    res_output = reconstructed_network(input)
    #    clean_output = target_network(input)

    # Set all weights to not trainable
    set_parameter_requires_grad(target_network, requires_grad=False)

    non_trainale_params = get_num_non_trainable_parameters(target_network)
    trainale_params = get_num_trainable_parameters(target_network)
    total_params = get_num_parameters(target_network)
    print_log("Target Network Trainable parameters: {}".format(trainale_params), log)
    print_log("Target Network Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Target Network Total # parameters: {}".format(total_params), log)

    print_log("=> Inserting Generator", log)

    generator = UAP(shape=(input_size, input_size),
                num_channels=num_channels,
                mean=mean,
                std=std,
                use_cuda=args.use_cuda)

    print_log("=> Generator :\n {}".format(generator), log)
    non_trainale_params = get_num_non_trainable_parameters(generator)
    trainale_params = get_num_trainable_parameters(generator)
    total_params = get_num_parameters(generator)
    print_log("Generator Trainable parameters: {}".format(trainale_params), log)
    print_log("Generator Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Generator Total # parameters: {}".format(total_params), log)

    perturbed_net = nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_network)]))
    perturbed_net = torch.nn.DataParallel(perturbed_net, device_ids=list(range(args.ngpu)))

    non_trainale_params = get_num_non_trainable_parameters(perturbed_net)
    trainale_params = get_num_trainable_parameters(perturbed_net)
    total_params = get_num_parameters(perturbed_net)
    print_log("Perturbed Net Trainable parameters: {}".format(trainale_params), log)
    print_log("Perturbed Net Non Trainable parameters: {}".format(non_trainale_params), log)
    print_log("Perturbed Net Total # parameters: {}".format(total_params), log)

    # Set the target model into evaluation mode
    perturbed_net.module.target_model.eval()
    perturbed_net.module.generator.train()

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
    elif args.loss_function == 'mse':
        criterion = HiddenMSELoss(num_classes=num_classes, use_cuda=args.use_cuda)
    else:
        raise ValueError

    if args.use_cuda:
        target_network.cuda()
        model2.cuda()
        generator.cuda()
        perturbed_net.cuda()
        criterion.cuda()

    optimizer = torch.optim.Adam(perturbed_net.parameters(), lr=state['learning_rate'])
    #'''
    # Measure the time needed for the UAP generation
    start = time.time()
    train_hidden(data_loader=data_train_loader,
            model=perturbed_net,
            model2=model2,
            criterion=criterion,
            optimizer=optimizer,
            epsilon=args.epsilon,
            num_iterations=args.num_iterations,
            targeted=args.targeted,
            target_class=args.target_class,
            log=log,
            arch=args.pretrained_arch,
            mask=neu_idx,
            split_layer=args.split_layer,
            do_val=args.do_val,
            num_hidden_neu=args.split_num_n,
            print_freq=args.print_freq,
            use_cuda=args.use_cuda)
    end = time.time()
    print_log("Time needed for UAP generation: {}".format(end - start), log)
    # evaluate
    print_log("Final evaluation:", log)
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
      'optimizer'   : optimizer.state_dict(),
      'args'        : copy.deepcopy(args),
    }, result_path, args.uap_model)

    #export uap and save it
    '''
    tuap = torch.unsqueeze(generator.uap, dim=0)
    plot_tuap = tuap[0].cpu().detach().numpy()
    plot_tuap = np.transpose(plot_tuap, (1, 2, 0))
    plot_tuap_normal = plot_tuap + 0.5
    plot_tuap_amp = plot_tuap / 2 + 0.5
    tuap_range = np.max(plot_tuap_amp) - np.min(plot_tuap_amp)
    plot_tuap_amp = plot_tuap_amp / tuap_range + 0.5
    plot_tuap_amp -= np.min(plot_tuap_amp)

    imgplot = plt.imshow(plot_tuap_amp)

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.pretrained_dataset,
                            network_arch=args.pretrained_arch,
                            random_seed=args.pretrained_seed)

    np.save(uap_path + '/' + args.uap_name, tuap.cpu().detach().numpy())
    plt.savefig(uap_path + '/uap_general.png')
    #plt.show()
    print('uap saved!')

    log.close()


def list_and(l1,l2):
    result = np.zeros(len(l1))
    for i in range(len(l1)):
        if l1[i] == 1  and l2[i] == 1:
            result[i] = 1
    return result


if __name__ == '__main__':
    #main
    main_hidden()
