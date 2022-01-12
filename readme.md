

## Make your dataset

In order to run the Sun-map correctly, you should prepare your dataset directory just like the tree below. *Note that the number of the volume and the label need to be exactly the same.*

```
└──dataroot *(your dataroot path)*
　　├── train
　　│　　├──volumes
　　│　　│　　└──volume-001.tiff
　　│　　├── labels
　　│　　│　　└──label-001.tiff
　　│　　├── nonlabel  *(if using semi-supervised model)*
　　│　　│　　└──label-100.tiff
　　│　　└──artifacts
　　│　　　　　└──volume-200.tiff
　　└── val
　　　　　├──volumes
　　　　　│　　└──volume-003.tiff
　　　　　├── labels
　　　　　│　　└──label-003.tiff
　　　　　└──artifacts
　　　　　　　　└──volume-400.tiff
```



## Train Sun-map

The baseline training command

```sh
python train.py --dataroot </path/to/your/train-set/> --n_samples 4 --gpu_ids <0, 1, 2,... gpu order> --lr 1e-4 --exp <experiment name> --checkpoint_name <.pth file name> --input_dim 128 --batch_size <batch_size> --net <unet | axialunet> --model segnet --pre_trained </path/to/your/pretrained/model.pth>
```

Semi-supervised training methods

```sh
python train.py --dataroot </path/to/your/train-set/> --n_samples 4 --gpu_ids <0, 1, 2,... gpu order> --lr 1e-4 --exp <experiment name> --checkpoint_name <.pth file name> --input_dim 128 --batch_size <batch_size> --net <unet | axialunet> --model semiseg --labeled_bs <generally half of the batch_size> --pre_trained </path/to/your/pretrained/model.pth>
```



 ## Evaluate your model

To evaluate cubes batch, the evaluating datasets directory must be like

```
└──dataroot *(your dataroot path)*
　　└── val
　　　　├──volumes
　　　　├── labels
　　　　└──artifacts
```

```sh
python eval.py --dataroot </path/to/your/val-set/> --gpu_ids <0, 1, 2,... gpu order> --net <axialunet | unet> --val_type cubes --pre_trained </path/to/your/pretrained/model.pth>
```

To evaluate a predicted volume and it's labeled GT

```sh
python eval.py --dataroot </path/to/your/predicted/volume.tiff> -- data_target </path/to/your/label.tiff> --gpu_ids <0, 1, 2,... gpu order> --net <axialunet | unet> --val_type volumes2 --pre_trained </path/to/your/pretrained/model.pth>
```

## Get the probability map with your trained-model

Write the defined parameters to *segpoints.ini* just like

```ini
[section-name]

dataroot=</path/to/whole-brain/image-sequences/>
start_point=0,0,0
end_point=1000,1000,1000
```

note that NO blank space between the coordinates, (x, y, z in order in ImageJ), just split by ','. then run

```sh
python eval.py --gpu_ids  <0, 1, 2,... gpu order> --batch_size <batch size> --process <processes number> --net <axialunet | unet> --val_type segment --section <section name> --exp <experiment name> --pre_trained </path/to/your/pretrained/model.pth>
```

