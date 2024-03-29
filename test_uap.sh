
#vgg19
#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=vgg19_imagenet.pth --pretrained_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --test_arch=vgg19 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done

#resnet50
for TARGET_CLASS in {755,743,804,700,922,174,547,369}
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=resnet50_imagenet_finetuned_repaired.pth --pretrained_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --test_arch=resnet50 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done