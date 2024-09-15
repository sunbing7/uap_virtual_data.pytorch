################################################################################################################################################
#resnet50
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python train_ae.py --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.001 --seed=123 --targeted=True --result_subfolder=result --batch_size=1 --num_batches=110 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=$TARGET_CLASS
#done



for TARGET_CLASS in {755,743,804,700,922,174,547,369}
do
  echo "Analyzing target class:" $TARGET_CLASS
  #for LAYER in {9,7,4}
#  for LAYER in {9,8,7,6,5,4}
  LAYER=9
#  do
    echo $LAYER
    python analyze_ae.py --option=analyze_layers --analyze_clean=0 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_batches=100 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4
    python analyze_ae.py --option=analyze_clean --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --seed=123 --num_batches=100 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4
    python analyze_ae.py --option=analyze_layers --analyze_clean=1 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --seed=123 --num_batches=100 --result_subfolder=result --target_class=$TARGET_CLASS --split_layer=$LAYER --batch_size=32 --ngpu=1 --workers=4

    python analyze_ae.py --option=classify --causal_type=act --target_class=$TARGET_CLASS --num_batches=100 --split_layer=$LAYER --th=1

    #python analyze_input.py --option=analyze_layers --analyze_clean=2 --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=32 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=32 --ngpu=1 --workers=4

    rm attribution/ae*.npy
#  done
done


################################################################################################################################################
#adv training
--------------------------------------------------------------------------------------------------------------------------------------------
#resnet50
#adv training
#targeted
python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.0000001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=5 --target_class=547

#nontargeted
python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.0000001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=False --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=5 --target_class=547

#uap training
python analyze_ae.py --option=repair_uap --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layers 9 --seed=123 --num_iterations=1 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4 --alpha=0.5 --target_class=547

--------------------------------------------------------------------------------------------------------------------------------------------
#googlenet
python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet.pth --learning_rate=0.001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=10 --target_class=753
python analyze_ae.py --option=repair --dataset=imagenet --arch=googlenet --model_name=googlenet_imagenet_ae_repaired_10.pth --learning_rate=0.0001 --split_layers 17 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --target_class=753

python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=googlenet_imagenet_finetuned_repaired.pth --pretrained_seed=123 --test_dataset=imagenet --test_arch=googlenet --result_subfolder=result --targeted=True --target_class=753 --ngpu=1 --workers=4

--------------------------------------------------------------------------------------------------------------------------------------------
#vgg19
python analyze_ae.py --option=repair_ae --dataset=imagenet --arch=vgg19 --model_name=vgg19_imagenet.pth --learning_rate=0.0000001 --split_layers 43 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1500 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=150

--------------------------------------------------------------------------------------------------------------------------------------------
#asl

--------------------------------------------------------------------------------------------------------------------------------------------
#caltech

--------------------------------------------------------------------------------------------------------------------------------------------
#eurosat