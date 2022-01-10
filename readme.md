

 ## How to evaluate your model

To evaluate cubes batch, the evaluating datasets directory must be like:

└──dataroot

​        └── val

​        ├──volumes

​        ├── labels

​        └──artifacts

```shell
python eval.py --dataroot </path/to/your/val-set/> --gpu_ids <0, 1, 2, 3,... gpu order> --net <axialunet | unet> --model evalnet --val_type cubes --pre_trained </path/to/your/pretrained/model.pth>
```

To evaluate a predicted volume and it's labeled GT:

```shell
python eval.py --dataroot </path/to/your/predicted/volume.tiff> -- data_target </path/to/your/label.tiff> --gpu_ids <0, 1, 2, 3,... gpu order> --net <axialunet | unet> --model evalnet --val_type volumes2 --pre_trained </path/to/your/pretrained/model.pth>
```

