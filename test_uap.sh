
#vgg19
#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=vgg19_imagenet.pth --pretrained_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --test_arch=vgg19 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#vgg19
#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=vgg19_imagenet.pth --pretrained_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --test_arch=vgg19 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=resnet50_imagenet.pth --pretrained_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --test_arch=resnet50 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=resnet50_imagenet_finetuned_repaired.pth --pretrained_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --test_arch=resnet50 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#googlenet
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=googlenet_imagenet.pth --pretrained_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=googlenet_imagenet_finetuned_repaired.pth --pretrained_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#googlenet caffe
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python test_uap_caffe.py --targeted=True --dataset=imagenet_caffe --pretrained_dataset=imagenet_caffe --test_name=googlenet_imagenet_caffe.pth --pretrained_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet_caffe --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}#
#do
#    python test_uap_caffe.py --targeted=True --dataset=imagenet_caffe --pretrained_dataset=imagenet_caffe --test_name=googlenet_imagenet_caffe_finetuned_repaired.pth --pretrained_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet_caffe --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#shufflenetv2
for TARGET_CLASS in {37,99,95,79,21,9,4,6}
do
    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --test_name=shufflenetv2_caltech.pth --pretrained_arch=shufflenetv2 --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done
#for TARGET_CLASS in {37,99,95,79,21,9,4,6}
#do
#    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --test_name=shufflenetv2_caltech_finetuned_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --test_name=shufflenetv2_caltech.pth --uap_name=perturbed_net_37.pth --pretrained_arch=shufflenetv2 --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4