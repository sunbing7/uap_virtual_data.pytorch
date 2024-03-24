for TARGET_CLASS in {150,214,39,527,65,639,771,412}
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=vgg19_imagenet.pth --pretrained_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --test_arch=vgg19 --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done