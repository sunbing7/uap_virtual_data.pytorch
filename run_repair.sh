################################################################################################################################################
#resnet50
python analyze_input.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=5 --target_class=547
python analyze_input.py --option=repair --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_ae_repaired_5.pth --learning_rate=0.0001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=547

python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet_ae_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=547 --ngpu=1 --workers=4
################################################################################################################################################
#googlenet
python analyze_input.py --option=repair_ae --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --learning_rate=0.001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=753
python analyze_input.py --option=repair --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet_ae_repaired.pth --learning_rate=0.0001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=753

python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --pretrained_seed=123 --test_dataset=imagenet --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=753 --ngpu=1 --workers=4

################################################################################################################################################
#vgg19
python analyze_input.py --option=repair_ae --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --learning_rate=0.001 --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=15 --target_class=150
python analyze_input.py --option=repair --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet_ae_repaired_15.pth --learning_rate=0.0001 --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=800 --ngpu=1 --workers=4 --target_class=150

################################################################################################################################################
#mobilenet asl
python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --learning_rate=0.001 --split_layers 3 --seed=123 --num_iterations=15 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19
python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired_15_10.pth --learning_rate=0.001 --split_layers 3 --seed=123 --num_iterations=5 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19


#to test
python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --learning_rate=0.001 --split_layers 3 --seed=123 --num_iterations=20 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19
################################################################################################################################################
#shufflenetv2 caltech
python analyze_input.py --option=repair_ae --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --learning_rate=0.001 --split_layers 6 --seed=123 --num_iterations=2000 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=10 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=37
python analyze_input.py --option=repair --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech_ae_repaired_20_50.pth --learning_rate=0.0001 --split_layers 6 --seed=123 --num_iterations=500 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=10 --ngpu=1 --workers=4 --target_class=37

################################################################################################################################################
#resnet50 eurosat
python analyze_input.py --option=repair_ae --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --learning_rate=0.005 --split_layers 9 --seed=123 --num_iterations=200 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=34 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=3
python analyze_input.py --option=repair --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat_ae_repaired.pth --learning_rate=0.00002 --split_layers 9 --seed=123 --num_iterations=100 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=34 --ngpu=1 --workers=4 --target_class=3

################################################################################################################################################
#wideresnet cifar10
python analyze_input.py --option=repair_ae --dataset=cifar10 --arch=wideresnet --model_name=wideresnet_cifar10.pth --learning_rate=0.001 --split_layers 6 --seed=123 --num_iterations=50 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=64 --ngpu=1 --workers=4 --alpha=0.9 --ae_iter=5 --target_class=3
python analyze_input.py --option=repair --dataset=cifar10 --arch=wideresnet --model_name=wideresnet_cifar10_ae_repaired_50.pth --learning_rate=0.00001 --split_layers 6 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=64 --ngpu=1 --workers=4 --target_class=3



