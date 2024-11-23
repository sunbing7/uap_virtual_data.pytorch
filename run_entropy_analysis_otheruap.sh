################################################################################################################################################
#resnet50
#sPGD

TARGET_CLASS=611
  #for LAYER in {9,7,4}
  for LAYER in {4,5,6,7,8,9}
  #LAYER=9
  do
    echo "UAP:" $UAPNAME ", Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=spgd --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
  done

#python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=spgd --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=9 --seed=123 --num_iterations=128 --result_subfolder=result --target_class=611 --batch_size=128 --ngpu=1 --workers=4

  #for LAYER in {9,7,4}
  for LAYER in {4,5,6,7,8,9}
  #LAYER=9
  do
    #echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=spgd --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
  done

#lavan

TARGET_CLASS=391
  #for LAYER in {9,7,4}
  for LAYER in {4,5,6,7,8,9}
  #LAYER=9
  do
    echo "UAP:" $UAPNAME ", Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=lavan --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
  done

#python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=9 --seed=123 --num_iterations=128 --result_subfolder=result --target_class=547 --batch_size=128 --ngpu=1 --workers=4

  #for LAYER in {9,7,4}
  for LAYER in {4,5,6,7,8,9}
  #LAYER=9
  do
    #echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=lavan --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
  done
#sga

TARGET_CLASS=174
  #for LAYER in {9,7,4}
  for LAYER in {4,5,6,7,8,9}
  #LAYER=9
  do
    echo "UAP:" $UAPNAME ", Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=sga --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
  done

#python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=9 --seed=123 --num_iterations=128 --result_subfolder=result --target_class=547 --batch_size=128 --ngpu=1 --workers=4

  #for LAYER in {9,7,4}
  for LAYER in {4,5,6,7,8,9}
  #LAYER=9
#  do
    #echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=sga --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
#  done
#gap
TARGET_CLASS=463

  #for LAYER in {9,7,4}
  for LAYER in {4,5,6,7,8,9}
  #LAYER=9
  do
    echo "UAP:" $UAPNAME ", Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=gap --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
  done

  #for LAYER in {9,7,4}
  for LAYER in {4,5,6,7,8,9}
  #LAYER=9
  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=gap --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
  done

