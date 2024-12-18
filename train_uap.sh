#vgg19
#for TARGET_CLASS in {214,39,527,65,639,771,412}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=150 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
################################################################################################################################################
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=0 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50  --model_name=resnet50_imagenet_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=174 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0784 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50  --model_name=resnet50_imagenet_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
################################################################################################################################################
#googlenet

#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python ython train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=753 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet  --model_name=googlenet_imagenet_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
################################################################################################################################################
#googlenet_caffe

#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python train_uap_caffe.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet  --model_name=googlenet_imagenet_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

################################################################################################################################################
#shufflenetv2 caltech

#for TARGET_CLASS in {37,85,55,79,21,9,4,6}
#do
#    python train_uap.py --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=37 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#for TARGET_CLASS in {37,85,55,79,21,9,4,6}
#do
#    python train_uap.py --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=85 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

################################################################################################################################################
#mobilenet asl

#for TARGET_CLASS in {19,17,8,21,2,9,23,6}
#do
#    python train_uap.py --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=19 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#for TARGET_CLASS in {19,17,8,21,2,9,23,6}
#do
#    python train_uap.py --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl_ae_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=23 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

################################################################################################################################################
#resnet50 eurosat

#for TARGET_CLASS in {9,1,8,2,3,7,4,6}
#do
#    python train_uap.py --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=3 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#for TARGET_CLASS in {9,1,8,2,3,7,4,6}
#do
#    python train_uap.py --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=3 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
################################################################################################################################################
#wideresnet cifar10

#for TARGET_CLASS in {0,1,2,3,4,5,6,7,8,9}
#do
#    python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10_trades.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=3 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#for TARGET_CLASS in {0,1,2,3,4,5,6,7,8,9}
#do
#    python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=0 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

#for TARGET_CLASS in {0,1,2,3,4,5,6,7,8,9}
#do
#    python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10_trades.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
################################################################################################################################################
#resnet110 cifar10

for TARGET_CLASS in {1,2,3,4,5,6,7,8,9}
do
    python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=resnet110 --model_name=resnet110_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
done
#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=resnet110 --model_name=resnet110_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=0 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#for TARGET_CLASS in {0,1,2,3,4,5,6,7,8,9}
#do
#    python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=0 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

#for TARGET_CLASS in {0,1,2,3,4,5,6,7,8,9}
#do
#    python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10_trades.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done