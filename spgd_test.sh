################################################################################################################################################
#sPGD resnet50
#for TARGET_CLASS in {611,734,854,859,497,577,8,5}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=spgd --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {611,734,854,859,497,577,8,5}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=spgd --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#sPGD vgg19
#for TARGET_CLASS in {898,895,861,720,764,701,71,545}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=spgd --model_name=vgg19_imagenet.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {898,895,861,720,764,701,71,545}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=spgd --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#sPGD googlenet
#for TARGET_CLASS in {581,20,700,671,83,138,197,619}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=spgd --model_name=googlenet_imagenet.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {581,20,700,671,83,138,197,619}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --uap_name=spgd --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#sPGD mobilenet asl
#for TARGET_CLASS in {3,8,17,28,0,19,18,1}
#do
#    python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --uap_name=spgd --model_name=mobilenet_asl.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {3,8,17,28,0,19,18,1}
#do
#    python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --uap_name=spgd --model_name=mobilenet_asl_ae_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#sPGD shufflenetv2 caltech
#for TARGET_CLASS in {49,56,95,92,48,50,3,34}
#do
#    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --uap_name=spgd --model_name=shufflenetv2_caltech.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
#for TARGET_CLASS in {49,56,95,92,48,50,3,34}
#do
#    python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --uap_name=spgd --model_name=shufflenetv2_caltech_finetuned_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
################################################################################################################################################
#sPGD resnet50 eurosat
#for TARGET_CLASS in {7,1,4,5,9,8,2,6}
#do
#    python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=spgd --model_name=resnet50_eurosat.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done
#resnet50
for TARGET_CLASS in {7,1,4,5,9,8,2,6}
do
    python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=spgd --model_name=resnet50_eurosat_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done