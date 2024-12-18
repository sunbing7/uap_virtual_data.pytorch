################################################################################################################################################
#attack
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

################################################################################################################################################
#test
python test_uap.py --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet_finetuned_repaired.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl_ae_repaired.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_finetuned_repaired.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat_finetuned_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --ngpu=1 --workers=4

################################################################################################################################################
#test
python test_uap.py --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=imagenet --pretrained_dataset=imagenet --model_name=vgg19_imagenet.pth --test_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet.pth --test_arch=googlenet --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=asl --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech.pth --test_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --result_subfolder=result --ngpu=1 --workers=4
python test_uap.py --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --ngpu=1 --workers=4

