################################################################################################################################################
#resnet50
python analyze_input.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=5 --target_class=547
python analyze_input.py --option=repair --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_ae_repaired_5.pth --learning_rate=0.0001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=547

python analyze_input.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=2 --target_class=547
python analyze_input.py --option=repair --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_ae_repaired_2.pth --learning_rate=0.0001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=547

python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet_ae_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=547 --ngpu=1 --workers=4


################################################################################################################################################
#googlenet
python analyze_input.py --option=repair_ae --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --learning_rate=0.001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=753
python analyze_input.py --option=repair --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet_ae_repaired_10.pth --learning_rate=0.0001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=753

python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --pretrained_seed=123 --test_dataset=imagenet --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=753 --ngpu=1 --workers=4
