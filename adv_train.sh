#modify analyze_input.py to call pgd_train instead of adv_train

#non-targeted adv training
#python analyze_input.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.0000001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1600 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=174

# targeted adv training
#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python analyze_input.py --option=repair_ae --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --learning_rate=0.000001 --split_layers 9 --seed=123 --num_iterations=1 --targeted=True --result_subfolder=result --batch_size=32 --num_batches=1600 --ngpu=1 --workers=4 --alpha=0.9 --ae_alpha=0.5 --ae_iter=50 --target_class=$TARGET_CLASS
#done

# known uap adv training
#python analyze_input.py --option=repair_uap --dataset=imagenet --arch=resnet50 --model_name=resnet50_imagenet.pth --split_layers 9 --seed=123 --num_iterations=1 --result_subfolder=result --batch_size=32 --ngpu=1 --workers=4 --alpha=0.5 --target_class=755

#test

    #python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_755_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=755 --ngpu=1 --workers=4
    #python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_743_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=743 --ngpu=1 --workers=4
    #python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_804_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=804 --ngpu=1 --workers=4
    #python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_700_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=700 --ngpu=1 --workers=4
    #python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_922_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=922 --ngpu=1 --workers=4
    #python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_174_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=174 --ngpu=1 --workers=4
    #python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_547_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=547 --ngpu=1 --workers=4
    #python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_369_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=369 --ngpu=1 --workers=4

for TARGET_CLASS in {755,743,804,700,922,174,547,369}
do
    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_pgd_untargeted_repaired.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
done

#for TARGET_CLASS in {755,743,804,700,922,174,547,369}
#do
#    python test_uap.py --targeted=True --dataset=imagenet --pretrained_dataset=imagenet --model_name=resnet50_imagenet_uap_repared.pth --test_arch=resnet50 --pretrained_seed=123 --test_dataset=imagenet --result_subfolder=result --targeted=True --target_class=$TARGET_CLASS --ngpu=1 --workers=4
#done