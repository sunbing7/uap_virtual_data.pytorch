################################################################################################################################################
#resnet50

  #for LAYER in {9,7,4}
  #for LAYER in {4,5,6,7,8,9}
  #do
  #  echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
  #  python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --batch_size=128 --ngpu=1 --workers=4
  #  rm attribution/uap*.npy
  #  rm attribution/clean*.npy
  #done

#python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=9 --seed=123 --num_iterations=128 --result_subfolder=result --batch_size=128 --ngpu=1 --workers=4

  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  #do
  #  echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
  #  python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --batch_size=128 --ngpu=1 --workers=4
  #  rm attribution/uap*.npy
  #  rm attribution/clean*.npy
  #done

################################################################################################################################################
#vgg19

  #for LAYER in {43,28,19}
  #for LAYER in {4,10,19,28,37,43}
  #do
  #  echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
  #  python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --batch_size=128 --ngpu=1 --workers=4
  #  rm attribution/uap*.npy
  #  rm attribution/clean*.npy
  #done


  #for LAYER in {43,28,19}
  #do
  #  echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
  #  python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --batch_size=128 --ngpu=1 --workers=4
  #  rm attribution/uap*.npy
  #  rm attribution/clean*.npy
  #done

################################################################################################################################################
#googlenet

  for LAYER in {8,14,17}
  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
  done

  #for LAYER in {8,14,17}
  #do
  #  echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
  #  python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4
  #  rm attribution/uap*.npy
  #  rm attribution/clean*.npy
  #done

################################################################################################################################################
#mobilenet asl

  for LAYER in {1,3}
  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
  done


  #for LAYER in {1,3}
  #do
  #  echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
  #  python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4
  #  rm attribution/uap*.npy
  #  rm attribution/clean*.npy
  #done

################################################################################################################################################
#shufflenetv2 caltech

  for LAYER in {1,6}
  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
  done

  #for LAYER in {1,6}
  #do
  #  echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
  #  python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4
  #  rm attribution/uap*.npy
  #  rm attribution/clean*.npy
  #done

################################################################################################################################################
#resnet50 eurosat
  for LAYER in {4,5,6,7,8,9}
  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --split_layer=$LAYER --seed=123 --num_iterations=1000 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4

    rm attribution/uap*.npy
    rm attribution/clean*.npy
  done

  #for LAYER in {9,7,4}
  #do
  #  echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
  #  python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4

  #  rm attribution/uap*.npy
  #  rm attribution/clean*.npy
  #done
