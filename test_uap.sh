
#vgg19
#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#vgg19
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=150 --ngpu=1 --workers=4
#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=150 --ngpu=1 --workers=4
################################################################################################################################################
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=174 --ngpu=1 --workers=4
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=174 --ngpu=1 --workers=4
################################################################################################################################################
#googlenet
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=573 --ngpu=1 --workers=4
################################################################################################################################################
#googlenet caffe
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python test_uap_caffe.py --targeted=True --dataset=imagenet_caffe --pretrained_dataset=imagenet_caffe --model_name=googlenet_imagenet_caffe.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet_caffe --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}#
#do
#    python test_uap_caffe.py --targeted=True --dataset=imagenet_caffe --pretrained_dataset=imagenet_caffe --model_name=googlenet_imagenet_caffe_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet_caffe --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#shufflenetv2
#for TARGET_CLASS in {37,85,55,79,21,9,4,6}
#do
#    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_ae_repaired_100_5.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4

#for TARGET_CLASS in {37,85,55,79,21,9,4,6}
#do
#    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_finetuned_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_finetuned_repaired.pth --uap_name=perturbed_net_37.pth --test_arch=shufflenetv2  --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4
################################################################################################################################################
#asl
#for TARGET_CLASS in {19,17,8,21,2,9,23,6}
#do
#    python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#for TARGET_CLASS in {19,17,8,21,2,9,23,6}
#do
#    python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl_ae_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl_ae_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=8 --ngpu=1 --workers=4
################################################################################################################################################
#eurosat resnet50
#for TARGET_CLASS in {9,1,8,2,3,7,4,6}
#do
#    python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {9,1,8,2,3,7,4,6}
#do
#    python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=9 --ngpu=1 --workers=4


################################################################################################################################################
#trades wideresnet
#for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=lavan --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --model_name=wideresnet_cifar10.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=3 --ngpu=1 --workers=4
#resnet50
#for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
#do
#    python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --model_name=wideresnet_cifar10_finetuned_repaired.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --model_name=wideresnet_cifar10_trades.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=5 --ngpu=1 --workers=4
#for TARGET_CLASS in {0,1,2,3,5,4,6,7,8,9}
#do
#    python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --model_name=wideresnet_cifar10_trades.pth --test_arch=wideresnet --pretrained_seed=123 --test_dataset=cifar10 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done



################################################################################################################################################
#lavan resnet50
#for TARGET_CLASS in {391,955,901,50,891,862,251,512}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=lavan --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {391,955,901,50,891,862,251,512}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=lavan --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done

