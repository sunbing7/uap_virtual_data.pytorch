#resnet50
#echo "#####################################resnet50################################"
#LEARNING_RATE=0.00001
#targeted
#echo "-------------------------------------targeted---------------------------------"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=$LEARNING_RATE --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=5 --target_class=547

#nontargeted
#echo "-------------------------------------nontargeted---------------------------------"
#LEARNING_RATE=0.000001
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=$LEARNING_RATE --split_layers 9 --seed=123 --num_iterations=1 --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=5 --target_class=547

#uap training
#echo "-------------------------------------known uap---------------------------------"
#echo "training UAP"
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_uap_multiple.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=resnet50 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#echo "repairing"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_uap --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --targets 547 --learning_rate=$LEARNING_RATE --split_layers 9 --seed=123 --num_iterations=1 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4 --alpha=0.5 --target_class=547
#python analyze_ae.py --option=repair_uap --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --targets 547 --learning_rate=0.00001 --split_layers 9 --seed=123 --num_iterations=1 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4 --alpha=0.5 --target_class=547
#python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --batch_size=32 --model_name=resnet50_imagenet_uap_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=547 --ngpu=1 --workers=4
################################################################################################################################################
#vgg19
#echo "#####################################vgg19################################"
#LEARNING_RATE=0.0000001
#targeted
#echo "-------------------------------------targeted---------------------------------"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --learning_rate=$LEARNING_RATE --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=15 --target_class=150
#nontargeted
#echo "-------------------------------------nontargeted---------------------------------"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --learning_rate=$LEARNING_RATE --split_layers 43 --seed=123 --num_iterations=1 --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=15 --target_class=150
#uap training
#echo "-------------------------------------known uap---------------------------------"
#echo "training UAP"
#for TARGET_CLASS in {150, 214,39,527,65,639,771,412}
#do
#    python train_uap_multiple.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap_multiple.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=150 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#echo "repairing"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_uap --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --targets 150 --learning_rate=$LEARNING_RATE --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=15 --target_class=150

################################################################################################################################################
#googlenet
#LEARNING_RATE=0.0001
#echo "#####################################googlenet################################"
#targeted
#echo "-------------------------------------targeted---------------------------------"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --learning_rate=$LEARNING_RATE --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=753
#nontargeted
#echo "-------------------------------------nontargeted---------------------------------"
#LEARNING_RATE=0.000001
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --learning_rate=$LEARNING_RATE --split_layers 17 --seed=123 --num_iterations=1 --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=753
#uap training
#echo "-------------------------------------known uap---------------------------------"
#echo "traicausal_typening UAP"
#train UAP first (10 for each class) 'uap_train_' + str(target_i) + '_' + str(idx) + '.npy'
#for TARGET_CLASS in {573,807,541,240,475,753,762,505}
#do
#    python train_uap_multiple.py --dataset=imagenet --pretrained_dataset=imagenet --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#echo "repairing"
#LEARNING_RATE=0.00001
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_uap --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --targets 753 --learning_rate=$LEARNING_RATE --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=753

################################################################################################################################################
#mobilenet asl

#echo "#####################################mobilenet asl################################"
#LEARNING_RATE=0.00001
#targeted only one target
#echo "-------------------------------------targeted---------------------------------"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --learning_rate=$LEARNING_RATE --split_layers 3 --seed=123 --num_iterations=20 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19
#nontargeted
#echo "-------------------------------------nontargeted---------------------------------"
#LEARNING_RATE=0.00001
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --learning_rate=$LEARNING_RATE --split_layers 3 --seed=123 --num_iterations=20 --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19
#python analyze_ae.py --option=repair_ae --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --learning_rate=0.00001 --split_layers 3 --seed=123 --num_iterations=20 --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19
#uap training
#echo "-------------------------------------known uap---------------------------------"
#echo "training UAP"
#for TARGET_CLASS in {19,17,8,21,2,9,23,6}
#do
#    python train_uap_multiple.py --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap_multiple.py --dataset=asl --pretrained_dataset=asl --pretrained_arch=mobilenet --model_name=mobilenet_asl.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=19 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#echo "repairing"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_uap --dataset=asl --arch=mobilenet --model_name=mobilenet_asl.pth --targets 19 --learning_rate=$LEARNING_RATE --split_layers 3 --seed=123 --num_iterations=20 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=101 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=19
################################################################################################################################################
#shufflenetv2 caltech
#echo "#####################################shufflenetv2 caltech################################"
#LEARNING_RATE=0.00001
#targeted
#echo "-------------------------------------targeted---------------------------------"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --learning_rate=$LEARNING_RATE --split_layers 6 --seed=123 --num_iterations=2000 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=10 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=10 --target_class=37
#nontargeted
#echo "-------------------------------------nontargeted---------------------------------"
#LEARNING_RATE=0.000001
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_ae --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --learning_rate=$LEARNING_RATE --split_layers 6 --seed=123 --num_iterations=2000 --result_subfolder=result --batch_size=32 --num_batches=10 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=10 --target_class=37
#uap training
#echo "-------------------------------------known uap---------------------------------"
#echo "training UAP"
#for TARGET_CLASS in {37,85,55,79,21,9,4,6}
#do
#    python train_uap_multiple.py --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#python train_uap_multiple.py --dataset=caltech --pretrained_dataset=caltech --pretrained_arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=37 --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#echo "repairing"
#echo $LEARNING_RATE
#python analyze_ae.py --option=repair_uap --dataset=caltech --arch=shufflenetv2 --model_name=shufflenetv2_caltech.pth --targets 37 --learning_rate=$LEARNING_RATE --split_layers 6 --seed=123 --num_iterations=2000 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=10 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=10 --target_class=37

################################################################################################################################################
#resnet50 eurosat
echo "#####################################resnet50 eurosat################################"
LEARNING_RATE=0.0001
#targeted
echo "-------------------------------------targeted---------------------------------"
echo $LEARNING_RATE
python analyze_ae.py --option=repair_ae --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --learning_rate=$LEARNING_RATE --split_layers 9 --seed=123 --num_iterations=200 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=34 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=10 --target_class=3
#python analyze_ae.py --option=repair_ae --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --learning_rate=0.00001 --split_layers 9 --seed=123 --num_iterations=200 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=34 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=3
#nontargeted
echo "-------------------------------------nontargeted---------------------------------"
LEARNING_RATE=0.00001
echo $LEARNING_RATE
python analyze_ae.py --option=repair_ae --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --learning_rate=$LEARNING_RATE --split_layers 9 --seed=123 --num_iterations=200 --result_subfolder=result --batch_size=32 --num_batches=34 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=10 --target_class=3
#python analyze_ae.py --option=repair_ae --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --learning_rate=0.00001 --split_layers 9 --seed=123 --num_iterations=200 --result_subfolder=result --batch_size=32 --num_batches=34 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=3
#uap training
#echo "-------------------------------------known uap---------------------------------"
#echo "training UAP"
#for TARGET_CLASS in {9,1,8,2,3,7,4,6}
#do
#    python train_uap_multiple.py --dataset=eurosat --pretrained_dataset=eurosat --pretrained_arch=resnet50 --model_name=resnet50_eurosat.pth --pretrained_seed=123 --epsilon=0.0392 --num_iterations=1000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=10 --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4 --batch_size=32 --learning_rate=0.005
#done
#echo "repairing"
#python analyze_ae.py --option=repair_uap --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --targets 3 --learning_rate=$LEARNING_RATE --split_layers 9 --seed=123 --num_iterations=200 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=34 --ngpu=1 --workers=4 --alpha=0.5 --ae_alpha=0.5 --ae_iter=10 --target_class=3
#python analyze_ae.py --option=repair_uap --dataset=eurosat --arch=resnet50 --model_name=resnet50_eurosat.pth --targets 3 --learning_rate=0.00001 --split_layers 9 --seed=123 --num_iterations=200 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=34 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=3
################################################################################################################################################
