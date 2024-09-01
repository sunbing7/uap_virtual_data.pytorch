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