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
python analyze_input.py --option=repair --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet_ae_repaired.pth --learning_rate=0.0001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=753

python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --pretrained_seed=123 --test_dataset=imagenet --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=753 --ngpu=1 --workers=4

################################################################################################################################################
#vgg19
python analyze_input.py --option=test --dataset=imagenet --arch=vgg19 --seed=123 --model_name=vgg19_imagenet.pth --num_iterations=2000 --result_subfolder=result --target_class=150 --batch_size=32 --ngpu=1 --workers=4
python analyze_input.py --option=repair_ae --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --learning_rate=0.001 --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=150
python analyze_input.py --option=test --dataset=imagenet --arch=vgg19 --seed=123 --model_name=vgg19_imagenet_ae_repaired.pth --num_iterations=2000 --result_subfolder=result --target_class=150 --batch_size=32 --ngpu=1 --workers=4
python analyze_input.py --option=repair --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet_ae_repaired_50_0_9.pth --learning_rate=0.000015 --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1600 --ngpu=1 --workers=4 --target_class=150
python analyze_input.py --option=repair_ae --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet_ae_repaired_50_0_9.pth --learning_rate=0.00002 --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1600 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=150
python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --test_name=vgg19_imagenet_ae_repaired.pth --pretrained_arch=vgg19 --pretrained_seed=123 --test_dataset=imagenet --test_arch=vgg19 --result_subfolder=result --targeted=True --target_class=214 --ngpu=1 --workers=4
python analyze_input.py --option=test --dataset=imagenet --arch=vgg19 --seed=123 --model_name=vgg19_imagenet_ae_repaired.pth --num_iterations=2000 --result_subfolder=result --target_class=150 --batch_size=32 --ngpu=1 --workers=4

python train_uap.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --model_name=vgg19_imagenet_ae_repaired.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=214 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005


################################################################################################################################################
#mobilenet asl
python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --learning_rate=0.001 --split_layers 3 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1020 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=19
python analyze_input.py --option=repair --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired_20_50.pth --learning_rate=0.0001 --split_layers 3 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1020 --ngpu=1 --workers=4 --target_class=19

python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired_20_50.pth --learning_rate=0.00002 --split_layers 3 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1020 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=19
python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired.pth --learning_rate=0.00002 --split_layers 3 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1020 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=19

python test_uap.py --targeted=True --dataset=asl --pretrained_dataset=asl --model_name=mobilenet_asl_ae_repaired_20_50.pth --test_arch=mobilenet --pretrained_seed=123 --test_dataset=caltech --test_arch=shufflenetv2 --result_subfolder=result --targeted=True --target_class=19 --ngpu=1 --workers=4


python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --learning_rate=0.001 --split_layers 3 --seed=123 --num_iterations=15 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19
python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired_15_10.pth --learning_rate=0.001 --split_layers 3 --seed=123 --num_iterations=5 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19

python analyze_input.py --option=repair --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired_20_50.pth --learning_rate=0.0001 --split_layers 3 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1020 --ngpu=1 --workers=4 --target_class=19

python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired_20_50.pth --learning_rate=0.00002 --split_layers 3 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1020 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=19
python analyze_input.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl_ae_repaired.pth --learning_rate=0.00002 --split_layers 3 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1020 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=19


################################################################################################################################################
#shufflenetv2 caltech
python analyze_input.py --option=repair_ae --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --learning_rate=0.001 --split_layers 6 --seed=123 --num_iterations=20 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=205 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=37
python analyze_input.py --option=repair --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech_ae_repaired_20_50.pth --learning_rate=0.0001 --split_layers 6 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=205 --ngpu=1 --workers=4 --target_class=37
python analyze_input.py --option=repair_ae --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech_ae_repaired_20_50.pth --learning_rate=0.00002 --split_layers 6 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=205 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=37
python analyze_input.py --option=repair_ae --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech_ae_repaired.pth --learning_rate=0.00002 --split_layers 6 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1600 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=37

python test_uap.py --targeted=True --dataset=caltech --pretrained_dataset=caltech --model_name=shufflenetv2_caltech_ae_repaired_20_50.pth --pretrained_arch=shufflenetv2 --pretrained_seed=123 --test_dataset=caltech --test_arch=shufflenetv2 --result_subfolder=result --targeted=True --target_class=37 --ngpu=1 --workers=4


python analyze_input.py --option=repair_ae --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --learning_rate=0.001 --split_layers 6 --seed=123 --num_iterations=500 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=10 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=37

python analyze_input.py --option=repair_ae --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech_ae_repaired_100_5.pth --learning_rate=0.001 --split_layers 6 --seed=123 --num_iterations=400 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=10 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=5 --target_class=37
python analyze_input.py --option=repair --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech_ae_repaired_20_50.pth --learning_rate=0.0001 --split_layers 6 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=205 --ngpu=1 --workers=4 --target_class=37


################################################################################################################################################
#resnet50 eurosat
python analyze_input.py --option=repair_ae --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --learning_rate=0.005 --split_layers 9 --seed=123 --num_iterations=10 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=300 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=3
python analyze_input.py --option=repair --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat_ae_repaired.pth --learning_rate=0.001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=300 --ngpu=1 --workers=4 --target_class=3
python analyze_input.py --option=repair_ae --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --learning_rate=0.00002 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=300 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=3
python analyze_input.py --option=repair_ae --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --learning_rate=0.00002 --split_layers 96 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=300 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=5 --target_class=3

python test_uap.py --targeted=True --dataset=eurosat --pretrained_dataset=eurosat --uap_name=uap.npy --model_name=resnet50_eurosat.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=eurosat --result_subfolder=result --targeted=True --target_class=3 --ngpu=1 --workers=4

