################################################################################################################################################
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_ae.py --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.001 --seed=123 --targeted=True --result_subfolder=result --batch_size=1 --num_batches=110 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=$TARGET_CLASS
#done

#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#  echo "Analyzing target class:" $TARGET_CLASS
  #for LAYER in {9,7,4}
#  for LAYER in {9,8,7,6,5,4}
#  LAYER=9
#  do
#    echo $LAYER
#    python analyze_ae.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_batches=100 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
#    python analyze_ae.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --seed=123 --num_batches=100 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4
#    python analyze_ae.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --seed=123 --num_batches=100 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4

#    python analyze_ae.py --option=classify --causal_type=act --target_class=$TARGET_CLASS --num_batches=100 --split_layer=$LAYER --th=1

    #python analyze_input.py --option=analyze_layers --analyze_clean=2 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4

#    rm attribution/ae*.npy
#  done
#done


#for TARGET_CLASS in {214,39,527,65,639,771,412}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
################################################################################################################################################
#vgg19
for TARGET_CLASS in {214,39,527,65,639,771,412}
do
    echo "Analyzing target class:" $TARGET_CLASS
    python train_uap_multiple.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
done

#resent50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap_multiple.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

#googlenet
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python train_uap_multiple.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

#shufflenetv2 caltech
#for TARGET_CLASS in {37,85,55,79,21,9,4,6}
#do
#    python train_uap_multiple.py --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

#mobilenet asl
#for TARGET_CLASS in {19,17,8,21,2,9,23,6}
#do
#    python train_uap_multiple.py --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

#resnet50 eurosat
#for TARGET_CLASS in {9,1,8,2,3,7,4,6}
#do
#    python train_uap_multiple.py --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done