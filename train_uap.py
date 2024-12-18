from __future__ import division

import os, sys, time, random, copy
import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.nn as nn
from collections import OrderedDict

from networks.uap import UAP
from utils.data import get_data_specs, get_data, Normalizer
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
    parser = argparse.ArgumentParser(description='Trains a UAP')
    # pretrained
    parser.add_argument('--dataset', default='imagenet',
                        choices=['imagenet', 'caltech', 'asl', 'eurosat', 'cifar10'],
                        help='Used dataset to generate UAP (default: imagenet)')

    parser.add_argument('--pretrained_dataset', default='imagenet',
                        choices=['imagenet', 'caltech', 'asl', 'eurosat', 'cifar10'],
                        help='Used dataset to train the initial model (default: imagenet)')

    parser.add_argument('--pretrained_arch', default='resnet50',
                        choices=['googlenet', 'vgg19', 'resnet50', 'shufflenetv2', 'mobilenet', 'wideresnet', 'resnet110'],
                        help='Used model architecture: (default: resnet50)')

    parser.add_argument('--model_name', type=str, default='vgg19.pth',
                        help='model name (default: vgg19.pth)')

    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    # Parameters regarding UAP
    parser.add_argument('--epsilon', type=float, default=0.03922,
                        help='Norm restriction of UAP (default: 10/255)')
    parser.add_argument('--num_iterations', type=int, default=2000,
                        help='Number of iterations (default: 2000)')
    parser.add_argument('--result_subfolder', default='result', type=str,
                        help='result subfolder name')
    parser.add_argument('--postfix', default='',
                        help='Postfix to attach to result folder')

    # Optimization options
    parser.add_argument('--loss_function', default='bounded_logit_fixed_ref', choices=['ce', 'neg_ce', 'logit', 'bounded_logit',
                                                                  'bounded_logit_fixed_ref', 'bounded_logit_neg'],
                        help='Used loss function for source classes: (default: bounded_logit_fixed_ref)')
    parser.add_argument('--confidence', default=0., type=float,
                        help='Confidence value for C&W losses (default: 0.0)')
    parser.add_argument('--targeted', default='', type=bool,
                        help='Target a specific class (default: False)')
    parser.add_argument('--target_class', type=int, default=1,
                        help='Target class (default: 1)')
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
    parser.add_argument('--is_nips', default=1, type=int,
                        help='Evaluation on NIPS data')
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

    _, pretrained_data_test = get_data(args.pretrained_dataset, args.pretrained_dataset, is_attack=True)

    pretrained_data_test_loader = torch.utils.data.DataLoader(pretrained_data_test,
                                                              batch_size=args.batch_size,
                                                              shuffle=False,
                                                              num_workers=args.workers,
                                                              pin_memory=True)

    ##### Dataloader for training ####
    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.pretrained_dataset)

    data_train, _ = get_data(args.dataset, args.pretrained_dataset, is_attack=True)

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

    #print_log("=> Network :\n {}".format(target_network), log)

    adaptive = ''
    if args.pretrained_dataset == "caltech" or args.pretrained_dataset == 'asl':
        if 'repaired' in args.model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
            adaptive = '_adaptive'
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in target_network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            target_network.load_state_dict(new_state_dict)

    elif args.pretrained_dataset == 'eurosat':
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        if 'repaired' in args.model_name:
            adaptive = '_adaptive'
    elif args.pretrained_dataset == "imagenet" and 'repaired' in args.model_name:
        target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        adaptive = '_adaptive'
    elif args.pretrained_dataset == "cifar10":
        if 'repaired' in args.model_name:
            target_network = torch.load(model_weights_path, map_location=torch.device('cpu'))
            adaptive = '_adaptive'
        else:
            if args.pretrained_arch == 'resnet110':
                sd0 = torch.load(model_weights_path)['state_dict']
                target_network.load_state_dict(sd0, strict=True)
            else:
                target_network.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
            if 'trades' in args.model_name:
                adaptive = '_trades'

    target_network = torch.nn.DataParallel(target_network, device_ids=list(range(args.ngpu)))

    if args.pretrained_arch == 'resnet110':
        # Normalization wrapper, so that we don't have to normalize adversarial perturbations
        normalize = Normalizer(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        target_network = nn.Sequential(normalize, target_network)

    # Set the target model into evaluation mode
    target_network.eval()

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

    uap_path = get_uap_path(uap_data=args.dataset,
                            model_data=args.pretrained_dataset,
                            network_arch=args.pretrained_arch,
                            random_seed=args.pretrained_seed)

    save_checkpoint({
      'arch'        : args.pretrained_arch,
      # 'state_dict'  : perturbed_net.state_dict(),
      'state_dict'  : perturbed_net.module.generator.state_dict(),
      'optimizer'   : optimizer.state_dict(),
      'args'        : copy.deepcopy(args),
    }, uap_path, 'perturbed_checkpoint_' + str(args.target_class) + '.pth')

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

    if args.targeted:
        target_name = str(args.target_class)
    else:
        target_name = 'nontarget'

    np.save(uap_path + '/uap_' + target_name + adaptive + '.npy', tuap.cpu().detach().numpy())
    plt.savefig(model_path + '/uap_' + target_name + adaptive + '.png')
    plt.show()
    torch.save(perturbed_net, uap_path + '/perturbed_net_' + target_name + adaptive + '.pth')
    print('uap saved!')
    log.close()


if __name__ == '__main__':
   main()
