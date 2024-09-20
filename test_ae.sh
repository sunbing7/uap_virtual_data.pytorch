#resnet50
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_tgt_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=547 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_untgt_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=547 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_uap_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=547 --ngpu=1 --workers=4
################################################################################################################################################
#vgg19
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet_pgd_tgt_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=150 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet_pgd_untgt_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=150 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet_uap_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=150 --ngpu=1 --workers=4
################################################################################################################################################
#googlenet
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_pgd_tgt_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=753 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_pgd_untgt_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=753 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_uap_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=753 --ngpu=1 --workers=4
################################################################################################################################################
#asl mobilenet
python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl_pgd_tgt_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=19 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl_pgd_untgt_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=19 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl_uap_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=19 --ngpu=1 --workers=4
################################################################################################################################################
#caltech shufflenetv2
python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_pgd_tgt_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_pgd_untgt_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_uap_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4
################################################################################################################################################
#eurosat resnet50
python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat_pgd_tgt_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=3 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat_pgd_untgt_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=3 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat_uap_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=3 --ngpu=1 --workers=4


