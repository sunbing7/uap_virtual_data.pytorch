################################################################################################################################################
#resnet50
for TARGET_CLASS in {755,743,804,700,922,174,547,369}
do
  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done

for TARGET_CLASS in {755,743,804,700,922,174,547,369}
do
  #for LAYER in {9,7,4}
  #for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done

################################################################################################################################################
#vgg19
for TARGET_CLASS in {150,214,39,527,65,639,771,412}
do
#  for LAYER in {43,28,19}
#  for LAYER in {43,37,28,19,10,4}
    LAYER=43
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done

for TARGET_CLASS in {150,214,39,527,65,639,771,412}
do

#  for LAYER in {43,28,19}
#  for LAYER in {43,37,28,19,10,4}
    LAYER=43
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done
################################################################################################################################################
#googlenet
for TARGET_CLASS in {573,807,541,240,475,753,762,505}
do

    LAYER=17
#  for LAYER in {8,14,17}
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done
for TARGET_CLASS in {573,807,541,240,475,753,762,505}
do

    LAYER=17
#  for LAYER in {8,14,17}
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done
################################################################################################################################################
#shufflenetv2 caltech
for TARGET_CLASS in {37,85,55,79,21,9,4,6}
do

   LAYER=6
#  for LAYER in {1,6}
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done

for TARGET_CLASS in {37,85,55,79,21,9,4,6}
do

   LAYER=6
#  for LAYER in {1,6}
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done
#python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --split_layer=6 --seed=123 --num_iterations=32 --result_subfolder=result --target_class=37 --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_clean --causal_type=act --targeted=True --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=37 --split_layer=6 --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=37 --split_layer=6 --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=classify --causal_type=act --target_class=37 --num_iterations=32 --split_layer=6 --th=1

################################################################################################################################################
#mobilenet asl
for TARGET_CLASS in {19,17,8,21,2,9,23,6}
do

   LAYER=3
#  for LAYER in {1,3}
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done

for TARGET_CLASS in {19,17,8,21,2,9,23,6}
do

   LAYER=3
#  for LAYER in {1,3}
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done
#python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --split_layer=1 --seed=123 --num_iterations=32 --result_subfolder=result --target_class=19 --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_clean --causal_type=act --targeted=True --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=19 --split_layer=3 --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=19 --split_layer=3 --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=classify --causal_type=act --target_class=19 --num_iterations=32 --split_layer=3 --th=1

################################################################################################################################################
#resnet50 eurosat
for TARGET_CLASS in {9,1,8,2,3,7,4,6}
do

   LAYER=9
#  for LAYER in {9,7,4}
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --split_layer=$LAYER --seed=123 --num_iterations=1000 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4

    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done

for TARGET_CLASS in {9,1,8,2,3,7,4,6}
do

   LAYER=9
#  for LAYER in {9,7,4}
#  do
    echo "Target class:" $TARGET_CLASS ", Layer:" $LAYER
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --causal_type=act --targeted=True --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat_ae_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4

    rm attribution/uap*.npy
    rm attribution/clean*.npy
#  done
done