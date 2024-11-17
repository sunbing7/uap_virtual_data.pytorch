################################################################################################################################################
#sga resnet50 imagenet
for TARGET_CLASS in {755,743,804,700,922,547,369} #174,
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=174 --ngpu=1 --workers=4

for TARGET_CLASS in {755,743,804,700,922,547,369} #174
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=174 --ngpu=1 --workers=4
################################################################################################################################################
#sga vgg19 imagenet
for TARGET_CLASS in {150,39,527,65,639,771,412}#214,
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=vgg19_imagenet.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=vgg19_imagenet.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=214 --ngpu=1 --workers=4

for TARGET_CLASS in {150,39,527,65,639,771,412}214,
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=214 --ngpu=1 --workers=4
################################################################################################################################################
#sga googlenet imagenet
for TARGET_CLASS in {807,541,240,475,753,762,505} #573,
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=googlenet_imagenet.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=googlenet_imagenet.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=573 --ngpu=1 --workers=4

for TARGET_CLASS in {807,541,240,475,753,762,505} #573,
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=sga --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=573 --ngpu=1 --workers=4
################################################################################################################################################
#sga mobilenet asl
for TARGET_CLASS in {17,8,21,2,9,23,6} #19,
do
    python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --uap_name=sga --model_name=mobilenet_asl.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --uap_name=sga --model_name=mobilenet_asl.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=19 --ngpu=1 --workers=4

for TARGET_CLASS in {17,8,21,2,9,23,6} #19,
do
    python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --uap_name=sga --model_name=mobilenet_asl_ae_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --uap_name=sga --model_name=mobilenet_asl_ae_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=19 --ngpu=1 --workers=4
################################################################################################################################################
#sga shufflenetv2 caltech
for TARGET_CLASS in {85,55,79,21,9,4,6} #37,
do
    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --uap_name=sga --model_name=shufflenetv2_caltech.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --uap_name=sga --model_name=shufflenetv2_caltech.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4

for TARGET_CLASS in {85,55,79,21,9,4,6} #37,
do
    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --uap_name=sga --model_name=shufflenetv2_caltech_finetuned_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --uap_name=sga --model_name=shufflenetv2_caltech_finetuned_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4
################################################################################################################################################
#sga resnet50 eurosat
for TARGET_CLASS in {1,8,2,3,7,4,6} #9
do
    python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=sga --model_name=resnet50_eurosat.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=sga --model_name=resnet50_eurosat.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=9 --ngpu=1 --workers=4

for TARGET_CLASS in {1,8,2,3,7,4,6} #9
do
    python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=sga --model_name=resnet50_eurosat_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=sga --model_name=resnet50_eurosat_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=9 --ngpu=1 --workers=4
################################################################################################################################################
#sga cifar10
for TARGET_CLASS in {1,2,3,5,4,6,7,8,9} #0
do
    python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --uap_name=sga --model_name=wideresnet_cifar10.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --uap_name=sga --model_name=wideresnet_cifar10.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=0 --ngpu=1 --workers=4

for TARGET_CLASS in {1,2,3,5,4,6,7,8,9} #0
do
    python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --uap_name=sga --model_name=wideresnet_cifar10_finetuned_repaired.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --uap_name=sga --model_name=wideresnet_cifar10_finetuned_repaired.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=0 --ngpu=1 --workers=4
