#!/bin/bash

# Fixed Params
PRETRAINED_DATASET="imagenet"
DATASET="places365"
EPSILON=0.03922
LOSS_FN="bounded_logit_fixed_ref"
CONFIDENCE=10
BATCH_SIZE=32
TARGET_CLASS=150
LEARNING_RATE=0.005
NUM_ITERATIONS=2000
WORKERS=4
NGPU=1
SUBF="imagenet_targeted"

TARGET_NETS="alexnet googlenet vgg16 vgg19 resnet152"

TARGET_CLASSES = [51,582,820,637,49,560,703,160,259,755,945,498,480,214,609,212,471,805,415,692,988,754,560,557,705,973,94,321,304,825,498,402]

for target_class in $TARGET_CLASSES; do
    python analyze_input.py --option=analyze_clean --causal_type=logit --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=target_class --batch_size=32 --ngpu=1 --workers=4
    python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=logit --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=target_class --batch_size=32 --ngpu=1 --workers=4

    python analyze_input.py --option=calc_pcc --avg_ca_name=clean_attribution_43_214_avg.npy --ca_name=uap_attribution_43_s13_214.npy --num_iterations=0
    python analyze_input.py --option=calc_pcc --analyze_clean=1 --avg_ca_name=clean_attribution_43_214_avg.npy --num_iterations=50 --target_class=214





    python3 train_uap.py \
      --dataset $DATASET \
      --pretrained_dataset $PRETRAINED_DATASET --pretrained_arch $target_net \
      --target_class $TARGET_CLASS --targeted \
      --epsilon $EPSILON \
      --loss_function $LOSS_FN --confidence $CONFIDENCE \
      --num_iterations $NUM_ITERATIONS \
      --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE \
      --workers $WORKERS --ngpu $NGPU \
      --result_subfolder $SUBF
done
