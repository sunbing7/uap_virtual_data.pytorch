################################################################################################################################################
#sga cifar10
#for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
#do
#    python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --uap_name=sga --model_name=wideresnet_cifar10.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --uap_name=sga --model_name=wideresnet_cifar10.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=0 --ngpu=1 --workers=4
#resnet50
#for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
#do
#    python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --uap_name=sga --model_name=wideresnet_cifar10_finetuned_repaired.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --uap_name=sga --model_name=wideresnet_cifar10_finetuned_repaired.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=0 --ngpu=1 --workers=4
