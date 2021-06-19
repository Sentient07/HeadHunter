# Head Detector

Code for the head detector (HeadHunter) proposed in our CVPR 2021 paper `Tracking Pedestrian Heads in Dense Crowd`. The `head_detection` module can be installed using `pip` in order to be able to plug-and-play with HeadHunter-T.

## Requirements

1. Nvidia Driver >= 418

2. Cuda 10.0 and compaitible CudNN

3. Python packages : To install the required python packages;
	`conda env create -f head_detection.yml`.

4. Use the anaconda environment `head_detection` by activating it, `source activate head_detection` or `conda activate head_detection`.

5. Alternatively pip can be used to install required packages using `pip install -r requirements.txt` or update your existing environment with the aforementioned yml file.


## Training

1. To train a model, define environment variable `NGPU`, config file and use the following command

``` $python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env train.py --cfg_file config/config_chuman.yaml --world_size $NGPU --num_workers 4 ```

2. Training is currently supported over (a) ScutHead dataset (b) CrowdHuman + ScutHead combined, (c) Our proposed CroHD dataset. This can be mentioned in the config file. 

3. To train the model, config files must be defined. More details about the config files are mentioned in the section below

## Evaluation and Testing

1. Unlike the training, testing and evaluation does not have a config file. Rather, all the parameters are set as argument variable while executing the code. Refer to the respective files, `evaluate.py` and `test.py`.
2. `evaluate.py` evaluates over the validation/test set using AP, MMR, F1, MODA and MODP metrics. 
3. `test.py` runs the detector over a "bunch of images" in the testing set for qualitative evaluation.

## Config file
A config file is necessary for all training. It's built to ease the number of arg variable passed during each execution. Each sub-sections are as elaborated below.

1. DATASET
    1. Set the `base_path` as the parent directory where the dataset is situated at.
    2. Train and Valid are `.txt` files that contains relative path to respective images from the `base_path` defined above and their corresponding Ground Truth in `(x_min, y_min, x_max, y_max)` format. Generation files for the three datasets can be seen inside `data` directory. For example, 
    ```
    /path/to/image.png
    x_min_1, y_min_1, x_max_1, y_max_1
    x_min_2, y_min_2, x_max_2, y_max_2
    x_min_3, y_min_3, x_max_3, y_max_3
    .
    .
    .
    ```
    3. `mean_std` are RGB means and stdev of the training dataset. If not provided, can be computed prior to the start of the training
2. TRAINING
    1. Provide `pretrained_model` and corresponding `start_epoch` for resuming.
    2. `milestones` are epoch at which the learning rates are set to `0.1 * lr`.
    3. `only_backbone` option loads just the Resnet backbone and not the head. Not applicable for mobilenet.

3. NETWORK
    1. The mentioned parameters are as described in experiment section of the paper.
    2. When using `median_anchors`, the anchors have to be defined in `anchors.py`.
    3. We experimented with mobilenet, resnet50 and resnet150 as alternative backbones. This experiment was not reported in the paper due to space constraints. We found the accuracy to significantly decrease with mobilenet but resnet50 and resnet150 yielded an almost same performance.
    4. We also briefly experimented with Deformable Convolutions but again didn't see noticable improvements in performance. The code we used are available in this repository.

### Note : 
This codebase borrows a noteable portion from pytorch-vision owing to the fact some of their modules cannot be "imported" as a package. 


## Citation :

```
@InProceedings{Sundararaman_2021_CVPR,
    author    = {Sundararaman, Ramana and De Almeida Braga, Cedric and Marchand, Eric and Pettre, Julien},
    title     = {Tracking Pedestrian Heads in Dense Crowd},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {3865-3875}
}
```


