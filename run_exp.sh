#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=alexnet --model_name=alexnet_cifar10.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=4992 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4 --batch_size=32
#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=googlenet --model_name=googlenet_cifar10.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=4992 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4 --batch_size=32
python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=4992 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4 --batch_size=32


python causal_analysis.py --causal_type='logit' --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=alexnet --uap_name=checkpoint_cifar10.pth.tar --model_name=alexnet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
python causal_analysis.py --causal_type='logit' --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=googlenet --uap_name=checkpoint_cifar10.pth.tar --model_name=googlenet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
python causal_analysis.py --causal_type='logit' --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --uap_name=checkpoint_cifar10.pth.tar --model_name=vgg19_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4

python train_general_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=43 --rec_type=first --epsilon=0.063 --num_iterations=4992 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=mse --confidence=0 --target_class=1 --ngpu=1 --workers=4

python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=alexnet --test_name=alexnet_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=googlenet --test_name=googlenet_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=vgg16 --test_name=vgg16_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=vgg19 --test_name=vgg19_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=resnet152 --test_name=resnet152_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
