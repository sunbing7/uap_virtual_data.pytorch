#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --is_nips=0 --pretrained_arch=alexnet --model_name=alexnet_cifar10.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=1000 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4 --batch_size=32
#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=googlenet --model_name=googlenet_cifar10.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=4992 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4 --batch_size=32
#python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=4992 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4 --batch_size=32


#python causal_analysis.py --causal_type='logit' --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=alexnet --uap_name=checkpoint_cifar10.pth.tar --model_name=alexnet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
#python causal_analysis.py --causal_type='logit' --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=googlenet --uap_name=checkpoint_cifar10.pth.tar --model_name=googlenet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
#python causal_analysis.py --causal_type='logit' --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --uap_name=checkpoint_cifar10.pth.tar --model_name=vgg19_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4

#python train_general_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=43 --rec_type=first --epsilon=0.063 --num_iterations=1562 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=mse --confidence=0 --target_class=1 --ngpu=1 --workers=4

#layer 43 uap act
#python causal_analysis.py --causal_type=uap_act --split_layer=43 --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=alexnet --uap_name=checkpoint_cifar10.pth.tar --model_name=alexnet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
#python causal_analysis.py --causal_type=uap_act --split_layer=43 --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=googlenet --uap_name=checkpoint_cifar10.pth.tar --model_name=googlenet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
#python causal_analysis.py --causal_type=uap_act --split_layer=43 --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --uap_name=checkpoint_cifar10.pth.tar --model_name=vgg19_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4

#python train_general_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=43 --split_num_n=4096 --do_val=10 --rec_type=first --epsilon=0.063 --num_iterations=1562 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=mse --confidence=0 --target_class=1 --ngpu=1 --workers=4


#layer 19 uap act
#python causal_analysis.py --causal_type=uap_act --split_layer=19 --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=alexnet --uap_name=checkpoint_cifar10.pth.tar --model_name=alexnet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
#python causal_analysis.py --causal_type=uap_act --split_layer=19 --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=googlenet --uap_name=checkpoint_cifar10.pth.tar --model_name=googlenet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
#python causal_analysis.py --causal_type=uap_act --split_layer=19 --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --uap_name=checkpoint_cifar10.pth.tar --model_name=vgg19_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4

#python train_general_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=19 --split_num_n=200704 --do_val=255 --rec_type=first --epsilon=0.063 --num_iterations=1562 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=mse --confidence=0 --target_class=1 --ngpu=1 --workers=4

#layer 10
#python causal_analysis.py --causal_type=uap_act --split_layer=10 --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --uap_name=checkpoint_cifar10.pth.tar --model_name=vgg19_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4

#python train_general_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=10 --split_num_n=401408 --do_val=255 --rec_type=first --epsilon=0.063 --num_iterations=1562 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=mse --confidence=0 --target_class=1 --ngpu=1 --workers=4

#layer 10
#python causal_analysis.py --causal_type=be_act --split_layer=10 --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --uap_name=checkpoint_cifar10.pth.tar --model_name=vgg19_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4

#python train_general_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=10 --split_num_n=401408 --do_val=255 --rec_type=first --epsilon=0.063 --num_iterations=1562 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=mse --confidence=0 --target_class=1 --ngpu=1 --workers=4


#test
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=alexnet --test_name=alexnet_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=googlenet --test_name=googlenet_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=vgg16 --test_name=vgg16_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=vgg19 --test_name=vgg19_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=cifar10 --test_arch=resnet152 --test_name=resnet152_cifar10.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4


#image net

#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=alexnet --model_name=alexnet_imagenet.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=1000 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --model_name=googlenet_imagenet.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=5000 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4
#python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --epsilon=0.063 --num_iterations=5000 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --targeted=True --target_class=1 --ngpu=1 --workers=4

#python causal_analysis.py --causal_type='logit' --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=alexnet --filter_arch=vgg19 --filter_dataset=imagenet --filter_name=vgg19_imagenet.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
#python causal_analysis.py --causal_type='logit' --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --filter_arch=vgg19 --filter_dataset=imagenet --filter_name=vgg19_imagenet.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4
#python causal_analysis.py --causal_type='logit' --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --filter_arch=vgg19 --filter_dataset=imagenet --filter_name=vgg19_imagenet.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4

#python train_general_uap.py --option=hidden --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=43 --rec_type=first --epsilon=0.063 --do_val=255 --num_iterations=5000 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=mse --confidence=0 --target_class=1 --ngpu=1 --workers=4
#python train_general_uap.py --option=class_embedding --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=43 --rec_type=first --epsilon=0.063 --num_iterations=5000 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=bounded_logit_fixed_ref --confidence=0 --target_class=1 --ngpu=1 --workers=4

#test
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=alexnet --test_name=alexnet_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=googlenet --test_name=googlenet_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=vgg16 --test_name=vgg16_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=vgg19 --test_name=vgg19_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=resnet152 --test_name=resnet152_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4

#python train_general_uap.py --option=hidden --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --split_layer=43 --rec_type=first --epsilon=0.063 --do_val=255 --num_iterations=5000 --result_subfolder=result --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --loss_function=mse --confidence=0 --target_class=1 --ngpu=1 --workers=4

#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=alexnet --test_name=alexnet_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=googlenet --test_name=googlenet_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=vgg16 --test_name=vgg16_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=vgg19 --test_name=vgg19_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet.pth --pretrained_seed=123 --uap_model=checkpoint.pth.tar --uap_name=uap_general.npy --test_dataset=imagenet --test_arch=resnet152 --test_name=resnet152_imagenet.pth --result_subfolder=result --target_class=1 --ngpu=1 --workers=4


#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=alexnet  --pretrained_seed=123 --uap_name=uap.npy --test_dataset=imagenet --test_arch=alexnet --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet  --pretrained_seed=123 --uap_name=uap.npy --test_dataset=imagenet --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg16 --pretrained_seed=123 --uap_name=uap.npy --test_dataset=imagenet --test_arch=vgg16 --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --uap_name=uap.npy --test_dataset=imagenet --test_arch=vgg19 --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet152  --pretrained_seed=123 --uap_name=uap.npy --test_dataset=imagenet --test_arch=resnet152 --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4


#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=alexnet  --pretrained_seed=123 --uap_name=perturbed_net.pth --test_dataset=imagenet --test_arch=alexnet --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet  --pretrained_seed=123 --uap_name=checkpoint.pth.tar --test_dataset=imagenet --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg16 --pretrained_seed=123 --uap_name=checkpoint.pth.tar --test_dataset=imagenet --test_arch=vgg16 --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --uap_name=checkpoint.pth.tar--test_dataset=imagenet --test_arch=vgg19 --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet152  --pretrained_seed=123 --uap_name=checkpoint.pth.tar --test_dataset=imagenet --test_arch=resnet152 --result_subfolder=result --targeted=True --target_class=1 --ngpu=1 --workers=4


#input attribution
#python analyze_input.py --option=analyze_inputs --causal_type=logit --targeted=True --dataset=cifar10 --arch=alexnet --model_name=alexnet_cifar10.pth --seed=123 --num_iterations=1 --result_subfolder=result --target_class=1 --batch_size=1 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_inputs --causal_type=logit --targeted=True --dataset=cifar10 --arch=alexnet --model_name=alexnet_cifar10.pth --seed=123 --num_iterations=8 --result_subfolder=result --target_class=1 --batch_size=8 --ngpu=1 --workers=4
#python analyze_input.py --option=calc_entropy --causal_type=logit --targeted=True --dataset=cifar10 --arch=alexnet --model_name=alexnet_cifar10.pth --seed=123 --num_iterations=1 --result_subfolder=result --target_class=1 --batch_size=1 --ngpu=1 --workers=4

#python analyze_input.py --option=analyze_layers --split_layer=6 --causal_type=logit --targeted=True --dataset=cifar10 --arch=alexnet --model_name=alexnet_cifar10.pth --seed=123 --num_iterations=8 --result_subfolder=result --target_class=1 --batch_size=8 --ngpu=1 --workers=4

#python analyze_input.py --option=analyze_layers --split_layer=43 --causal_type=logit --targeted=True --dataset=cifar10 --arch=vgg19 --model_name=vgg19_cifar10.pth --seed=123 --num_iterations=8 --result_subfolder=result --target_class=1 --batch_size=8 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_layers --split_layer=43 --causal_type=logit --targeted=True --dataset=cifar10 --arch=vgg19 --model_name=vgg19_cifar10.pth --seed=123 --num_iterations=1024 --result_subfolder=result --target_class=1 --batch_size=64 --ngpu=1 --workers=4
#python analyze_input.py --option=calc_entropy --causal_type=logit --targeted=True --dataset=cifar10 --arch=vgg19 --model_name=vgg19_cifar10.pth --seed=123 --num_iterations=1 --result_subfolder=result --target_class=1 --batch_size=1 --ngpu=1 --workers=4

#image net
python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_cifar10.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=214 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005

python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --uap_name=perturbed_net_214.pth --test_dataset=imagenet --test_arch=vgg19 --result_subfolder=result --targeted=True --target_class=214 --ngpu=1 --workers=4

python export_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --uap_model=perturbed_checkpoint_214.pth --pretrained_seed=123 --result_subfolder=result --uap_name=uap_214.npy --target_class=214 --ngpu=1 --workers=4



python analyze_input.py --option=calc_entropy --causal_type=act --analyze_clean=1 --num_iterations=50 --target_class=51 --split_layer=43
python analyze_input.py --option=calc_entropy --causal_type=act --idx=0 --target_class=51 --num_iterations=0 --split_layer=43

python analyze_input.py --option=test --dataset=imagenet --arch=vgg19 --seed=123 --num_iterations=1000 --result_subfolder=result --target_class=214 --batch_size=32 --ngpu=1 --workers=4

#test
python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --split_layer=28 --seed=123 --num_iterations=32 --result_subfolder=result --target_class=150 --batch_size=32 --ngpu=1 --workers=4
python analyze_input.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=150 --split_layer=19 --batch_size=32 --ngpu=1 --workers=4
python analyze_input.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --seed=123 --num_iterations=50 --result_subfolder=result --target_class=150 --split_layer=19 --batch_size=32 --ngpu=1 --workers=4
#python analyze_input.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --split_layer=19 --seed=123 --num_iterations=32 --result_subfolder=result --target_class=214 --batch_size=32 --ngpu=1 --workers=4
for IDX in {0..49} do
    python analyze_input.py --option=pcc --causal_type=act --idx=0 --target_class=150 --num_iterations=50 --split_layer=28
    python analyze_input.py --option=pcc --causal_type=act --idx=1 --target_class=150 --num_iterations=50 --split_layer=28
    python analyze_input.py --option=entropy --causal_type=act --target_class=150 --num_iterations=32 --split_layer=28
    python analyze_input.py --option=pcc --causal_type=act --target_class=150 --num_iterations=32 --split_layer=28

    python analyze_input.py --option=classify --causal_type=act --target_class=150 --num_iterations=32 --split_layer=28
done

python analyze_input.py --option=classify --causal_type=act --target_class=150 --num_iterations=32 --split_layer=28 --th=1

