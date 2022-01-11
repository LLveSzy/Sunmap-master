

## How to make your dataset

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



## How to train Sun-map

The baseline training command

```sh
python train.py --dataroot </path/to/your/train-set/> --n_samples 4 --gpu_ids <0, 1, 2, 3,... gpu order> --lr 1e-4 --exp <experiment name> --checkpoint_name <.pth file name> --input_dim 128 --batch_size <batch_size> --net <unet | axialunet> --model segnet --pre_trained </path/to/your/pretrained/model.pth>
```

Semi-supervised training methods

```sh
python train.py --dataroot </path/to/your/train-set/> --n_samples 4 --gpu_ids <0, 1, 2, 3,... gpu order> --lr 1e-4 --exp <experiment name> --checkpoint_name <.pth file name> --input_dim 128 --batch_size <batch_size> --net <unet | axialunet> --model semiseg --labeled_bs <generally half of the batch_size> --pre_trained </path/to/your/pretrained/model.pth>
```



 ## How to evaluate your model

To evaluate cubes batch, the evaluating datasets directory must be like

```
└──dataroot *(your dataroot path)*
　　└── val
　　　　├──volumes
　　　　├── labels
　　　　└──artifacts
```

```sh
python eval.py --dataroot </path/to/your/val-set/> --gpu_ids <0, 1, 2, 3,... gpu order> --net <axialunet | unet> --model evalnet --val_type cubes --pre_trained </path/to/your/pretrained/model.pth>
```

To evaluate a predicted volume and it's labeled GT

```sh
python eval.py --dataroot </path/to/your/predicted/volume.tiff> -- data_target </path/to/your/label.tiff> --gpu_ids <0, 1, 2, 3,... gpu order> --net <axialunet | unet> --model evalnet --val_type volumes2 --pre_trained </path/to/your/pretrained/model.pth>
```

