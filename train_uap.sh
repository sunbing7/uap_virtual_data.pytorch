# train multiple UAP
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=214 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=150 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=39 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=527 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=65 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=639 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=771 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

#vgg19
#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

#for TARGET_CLASS in {150,214,39,527,65,639,771,412}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet_ae_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
################################################################################################################################################
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50  --model_name=resnet50_imagenet_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

################################################################################################################################################
#googlenet

for TARGET_CLASS in {573,807,541,240,475,753,762,505}
do
    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
done

#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet  --model_name=googlenet_imagenet_finetuned_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done

################################################################################################################################################
