# Universal Adversarial Perturbation with virtual data
This is the repository accompanying our CVPR 2020 paper [Understanding Adversarial Examples from the Mutual Influence of Images and Perturbations](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Understanding_Adversarial_Examples_From_the_Mutual_Influence_of_Images_and_CVPR_2020_paper.pdf)

## Setup
You can install the requirements with `pip3 install requirements.txt`.

### Config
Copy the `sample_config.py` to `config.py` (`cp ./config/sample_config.py ./config/config.py`) and edit the paths accordingly.

### Datasets
The code supports training UAPs on ImageNet, MS COCO, PASCAL VOC and Places365

#### ImageNet
The [ImageNet](http://www.image-net.org/) dataset should be preprocessed, such that the validation images are located in labeled subfolders as for the training set. You can have a look at this [bash-script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) if you did not process your data already. Set the paths in your `config.py`.
```
IMAGENET_PATH = "/path/to/Data/ImageNet"
```

#### COCO
The [COCO](https://cocodataset.org/#home) 2017 images can be downloaded from here for [training](http://images.cocodataset.org/zips/train2017.zip) and [validation](http://images.cocodataset.org/zips/val2017.zip). After downloading and extracting the data update the paths in your `config.py`.
```
COCO_2017_TRAIN_IMGS = "/path/to/COCO/train2017/"			
COCO_2017_TRAIN_ANN = "/path/to/COCO/annotations/instances_train2017.json"
COCO_2017_VAL_IMGS = "/path/to/COCO/val2017/"
COCO_2017_VAL_ANN = "/path/to/instances_val2017.json"
```

#### PASCAL VOC
The training/validation data of the [PASCAL VOC2012 Challenge](http://host.robots.ox.ac.uk/pascal/VOC/) can be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar). After downloading and extracting the data update the paths in your `config.py`.
```
VOC_2012_ROOT = "/path/to/Data/VOCdevkit/"
```

#### Places 365
The [Places365](http://places2.csail.mit.edu/index.html) data can be downloaded from [here](http://places2.csail.mit.edu/download.html). After downloading and extracting the data update the paths in your `config.py`.
```
PLACES365_ROOT = "/home/user/Data/places365/"
```

## Run
Run `bash ./run.sh` to generate UAPs for different target models trained on ImageNet using virtual data Places365. The bash script should be easy to adapt to perform different experiments. The jupyter notebook `pcc_analysis.ipynb` is an example for the PCC-analysis discussed in the paper. 

## Citation
```
@inproceedings{zhang2020understanding,
  title={Understanding Adversarial Examples From the Mutual Influence of Images and Perturbations},
  author={Zhang, Chaoning and Benz, Philipp and Imtiaz, Tooba and Kweon, In So},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14521--14530},
  year={2020}
}
```


gen UAP from candidate models:
python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=alexnet --pretrained_seed=123 --epsilon=0.08 --num_iterations=5000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=0 --target_class=1 --ngpu=1 --workers=4 --model_name=alexnet_cifar10.pth
python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=googlenet --pretrained_seed=123 --epsilon=0.08 --num_iterations=5000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=0 --target_class=1 --ngpu=1 --workers=4 --model_name=googlenet_cifar10.pth
python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg16 --pretrained_seed=123 --epsilon=0.08 --num_iterations=5000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=0 --target_class=1 --ngpu=1 --workers=4 --model_name=vgg16_cifar10.pth
python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.08 --num_iterations=5000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=0 --target_class=1 --ngpu=1 --workers=4 --model_name=vgg19_cifar10.pth
python train_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=resnet152 --pretrained_seed=123 --epsilon=0.08 --num_iterations=10000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=0 --target_class=1 --ngpu=1 --workers=4 --model_name=resnet152_cifar10.pth


test generated UAPs:
python test_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=alexnet --pretrained_seed=123 --result_subfolder=result --target_class=1 --ngpu=1 --workers=4 --model_name=alexnet_cifar10.pth --uap_name=checkpoint_cifar10.pth.tar

causal analysis on filter model with candidate UAPs:
python causal_analysis.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=alexnet --uap_name=checkpoint_cifar10.pth.tar --model_name=alexnet_cifar10.pth --filter_arch=vgg19 --filter_dataset=cifar10 --filter_name=vgg19_cifar10.pth --pretrained_seed=123 --num_iterations=1562 --result_subfolder=result --target_class=1 --batch_size=32 --ngpu=1 --workers=4

generate general uap using filter model:
python train_general_uap.py --dataset=cifar10 --pretrained_dataset=cifar10 --pretrained_arch=vgg19 --pretrained_seed=123 --epsilon=0.08 --num_iterations=5000 --result_subfolder=result --loss_function=bounded_logit_fixed_ref --confidence=0 --target_class=1 --ngpu=1 --workers=4 --model_name=vgg19_cifar10.pth