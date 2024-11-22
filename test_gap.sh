################################################################################################################################################
#gap resnet50 imagenet
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=gap --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=463 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=gap --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=463 --ngpu=1 --workers=4

################################################################################################################################################
#gap vgg19 imagenet
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=gap --model_name=vgg19_imagenet.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=45 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=gap --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=45 --ngpu=1 --workers=4
################################################################################################################################################
#gap googlenet imagenet
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=gap --model_name=googlenet_imagenet.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=458 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=gap --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=458 --ngpu=1 --workers=4
################################################################################################################################################
#gap mobilenet asl
python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --uap_name=gap --model_name=mobilenet_asl.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=7 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --uap_name=gap --model_name=mobilenet_asl_ae_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=7 --ngpu=1 --workers=4
################################################################################################################################################
#gap shufflenetv2 caltech
python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --uap_name=gap --model_name=shufflenetv2_caltech.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=24 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --uap_name=gap --model_name=shufflenetv2_caltech_finetuned_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=24 --ngpu=1 --workers=4
################################################################################################################################################
#gap resnet50 eurosat
python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=gap --model_name=resnet50_eurosat.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=3 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=gap --model_name=resnet50_eurosat_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=3 --ngpu=1 --workers=4
