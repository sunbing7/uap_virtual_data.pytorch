#!/bin/bash

TARGET_CLASS=150
LAYER=43

#python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4


for IDX in {0..31}
do
    #echo $IDX
    python analyze_input.py --option=calc_pcc --analyze_clean=1 --causal_type=act --idx=$IDX --target_class=$TARGET_CLASS --num_iterations=0 --split_layer=$LAYER

done

for IDX in {0..31}
do
    #echo $IDX
    python analyze_input.py --option=calc_pcc --analyze_clean=0 --causal_type=act --idx=$IDX --target_class=$TARGET_CLASS --num_iterations=0 --split_layer=$LAYER
done
