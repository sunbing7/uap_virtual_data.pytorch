#resnet50
TARGET_CLASS=256

for EN_WEIGHT in {0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}
do
    LAYER=9
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=advanced --en_weight=$EN_WEIGHT --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
    python analyze_input.py --option=analyze_entropy --analyze_clean=0 --uap_name=advanced --en_weight=$EN_WEIGHT --causal_type=act --targeted=True --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet_finetuned_repaired.pth --split_layer=$LAYER --seed=123 --num_iterations=128 --result_subfolder=result --target_class=$TARGET_CLASS --batch_size=128 --ngpu=1 --workers=4
done




