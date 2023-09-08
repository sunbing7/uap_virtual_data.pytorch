#!/bin/bash

TARGET_CLASSES=(51 582 820 637 49 560 703 160 259 755 945 498 480 214 609 212 471 805 415 692 988 754 560 557 705 973 94 321 304 825 498 402 )
IDX=0

for tgt in ${TARGET_CLASSES[@]}; do

    python analyze_input.py --option=analyze_clean --causal_type=logit --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$tgt --split_layer=43 --batch_size=32 --ngpu=1 --workers=4
    python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=logit --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$tgt --split_layer=43 --batch_size=32 --ngpu=1 --workers=4

    python analyze_input.py --option=calc_pcc --idx=$IDX --target_class=$tgt --num_iterations=0 --split_layer=43
    python analyze_input.py --option=calc_pcc --analyze_clean=1 --num_iterations=50 --target_class=$tgt --split_layer=43
    ((IDX++))
    #echo $IDX
    #echo $tgt
done
