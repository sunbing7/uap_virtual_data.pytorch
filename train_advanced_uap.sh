
#resnet50
#TARGET_CLASS=256

#for EN_WEIGHT in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
#do
#    python train_advanced_uap.py --split_layer 9 --en_weight=$EN_WEIGHT --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python train_advanced_uap.py --split_layer 9 --en_weight=0.1 --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=0 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

################################################################################################################################################
#vgg19
#TARGET_CLASS=522

#for EN_WEIGHT in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
#do
#    python train_advanced_uap.py --split_layer 43 --en_weight=$EN_WEIGHT --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python train_advanced_uap.py --split_layer 43 --en_weight=0.9 --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=522 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
################################################################################################################################################
#googlenet
#TARGET_CLASS=365

#for EN_WEIGHT in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
#do
#    python train_advanced_uap.py --split_layer 17 --en_weight=$EN_WEIGHT --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python train_advanced_uap.py --split_layer 17 --en_weight=0.1 --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=365 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

################################################################################################################################################
#mobilenet asl
#TARGET_CLASS=7

#for EN_WEIGHT in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
#do
#    python train_advanced_uap.py --split_layer 3 --en_weight=$EN_WEIGHT --adjust=100.0 --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#    python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl_ae_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python train_advanced_uap.py --split_layer 3 --en_weight=0.1 --adjust=100.0 --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=7 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
################################################################################################################################################
#shufflenetv2 caltech
TARGET_CLASS=44

for EN_WEIGHT in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
do
    python train_advanced_uap.py --split_layer 6 --en_weight=$EN_WEIGHT --adjust=1000.0 --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_finetuned_repaired.pth --uap_name=perturbed_net_37.pth --test_arch=shufflenetv2  --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python train_advanced_uap.py --split_layer 6 --en_weight=0.1 --adjust=1000.0 --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=44 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
################################################################################################################################################
#resnet50 eurosat
#TARGET_CLASS=7

#for EN_WEIGHT in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
#do
#    python train_advanced_uap.py --split_layer 9 --en_weight=$EN_WEIGHT --adjust=5.0 --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#    python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python train_advanced_uap.py --split_layer 9 --en_weight=0.9 --adjust=5.0 --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=7 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
################################################################################################################################################
#wideresnet cifar10
TARGET_CLASS=9

for EN_WEIGHT in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
do
    python train_advanced_uap.py --split_layer 6 --en_weight=$EN_WEIGHT --adjust=100.0 --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
    python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --model_name=wideresnet_cifar10_finetuned_repaired.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python train_advanced_uap.py --split_layer 6 --en_weight=0.1 --adjust=100.0 --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=wideresnet --model_name=wideresnet_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=9 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
