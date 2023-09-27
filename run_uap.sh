#!/bin/bash

for TARGET_CLASS in {150,214,39,527,65,639,771}
do
  echo "Analyzing target class:" $TARGET_CLASS
  #python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

  for LAYER in {28,19}
  do
    #echo $LAYER
    python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    python analyze_input.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4
    python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4

    python analyze_input.py --option=classify --causal_type=act --target_class=$TARGET_CLASS --num_iterations=32 --split_layer=$LAYER --th=1
    rm attribution/uap*.npy
    rm attribution/clean*.npy
  done
done