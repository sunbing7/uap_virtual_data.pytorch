################################################################################################################################################
#resnet50
#sPGD
UAPNAME=spgd
TARGET_CLASS=611
  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    echo "UAP:" $UAPNAME ", Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=$UAPNAME --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done

#python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=9 --seed=123 --num_iterations=128 --result_subfolder=result --target_class=547 --batch_size=128 --ngpu=1 --workers=4

  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    #echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=$UAPNAME --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done

#lavan
UAPNAME=lavan
TARGET_CLASS=391
  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    echo "UAP:" $UAPNAME ", Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=$UAPNAME --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done

#python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=9 --seed=123 --num_iterations=128 --result_subfolder=result --target_class=547 --batch_size=128 --ngpu=1 --workers=4

  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    #echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=$UAPNAME --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done
#sga
UAPNAME=sga
TARGET_CLASS=174
  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    echo "UAP:" $UAPNAME ", Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=$UAPNAME --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done

#python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=9 --seed=123 --num_iterations=128 --result_subfolder=result --target_class=547 --batch_size=128 --ngpu=1 --workers=4

  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    #echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=$UAPNAME --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done
#gap
TARGET_CLASS=463
UAPNAME=gap
  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    echo "UAP:" $UAPNAME ", Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=$UAPNAME --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done

  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=$UAPNAME --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done

