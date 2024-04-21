#!/bin/bash
################################################################################################################################################
#vgg19
#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#  echo "Analyzing target class:" $TARGET_CLASS

#  for LAYER in {43,28,19}
#  do
    #echo $LAYER
#    python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
#    python analyze_input.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4
#    python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4

#    python analyze_input.py --option=classify --causal_type=act --target_class=$TARGET_CLASS --num_iterations=32 --split_layer=$LAYER --th=1
#    rm attribution/uap*.npy
#    rm attribution/clean*.npy
#  done
#done
################################################################################################################################################
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#  echo "Analyzing target class:" $TARGET_CLASS

#  for LAYER in {9,7,4}
#  do
    #echo $LAYER
#    python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
#    python analyze_input.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4
#    python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4

#    python analyze_input.py --option=classify --causal_type=act --target_class=$TARGET_CLASS --num_iterations=32 --split_layer=$LAYER --th=1
#    rm attribution/uap*.npy
#    rm attribution/clean*.npy
#  done
#done
################################################################################################################################################
#googlenet
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#  echo "Analyzing target class:" $TARGET_CLASS

#  for LAYER in {8,14,17}
#  do
    #echo $LAYER
#    python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
#    python analyze_input.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4
#    python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4

#    python analyze_input.py --option=classify --causal_type=act --target_class=$TARGET_CLASS --num_iterations=32 --split_layer=$LAYER --th=1
#    rm attribution/uap*.npy
#    rm attribution/clean*.npy
#  done
#done
################################################################################################################################################
#googlenet caffenet
for TARGET_CLASS in {573,807,541,240,475,753,762,505}
do
  echo "Analyzing target class:" $TARGET_CLASS

  for LAYER in {19,22}
  do
    #echo $LAYER
    python analyze_input_caffe.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet_caffe --arch=googlenet --model_name=googlenet_imagenet_caffe.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    python analyze_input_caffe.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet_caffe --arch=googlenet --model_name=googlenet_imagenet_caffe.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4
    python analyze_input_caffe.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet_caffe --arch=googlenet --model_name=googlenet_imagenet_caffe.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4

    python analyze_input_caffe.py --option=classify --causal_type=act --target_class=$TARGET_CLASS --num_iterations=32 --split_layer=$LAYER --th=1
    rm attribution/uap*.npy
    rm attribution/clean*.npy
  done
done