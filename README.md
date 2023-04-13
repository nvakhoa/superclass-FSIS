## superclass-FSIS

This is the code for the paper "Few Shot Instance Segmentation with Class Hierarchy Mining".

This code is based on Detectron2 and parts of MTFA's source code.

We advise the users to create a new conda environment and install our source code in the same way as the detectron2 source code. See [INSTALL.md](INSTALL.md).

After setting up the dependencies, installation should simply be:

`pip install -e .` in this folder.

## Configurations

All our configs can be found in the `configs/ours` directory.

The first training stage is: `configs/ours/mask_rcnn_R_101_FPN_base_220k.yaml`

1shot,5shot and 10_shot SMS+LR configs for the all classes are named as such:

`configs/ours/fs/SMS_LR_{shot_number}shot.yaml`


## Models
Pre-trained weights are reported in Table 4 of the main paper [here](https://drive.google.com/drive/folders/1OZLqQ_bFefY-_6NmMxo8x9EVS3nynkX5?usp=sharing)

### Running the scripts

To run the training, the `tools/run_train.py` script is used. Run it with `-h` to get all available options

Alternatively, we provide the scripts in `bash` to easily produce the experiments.

### Seting up the data

We use the same `datasets` folder used in Detectron2 and MTFA. Download and unzip the cocosplit folder [here](https://drive.google.com/file/d/17-doo4n2pXneZwJFL9PkeZSuScEpbzTJ/view?usp=sharing).

Also, setup a `coco` directory in `datasets`, exactly the same way as MTFA. For this, just download COCO2014 train + val and place them in trainval, similarly download COCO2014 test.


### Generating the few-shots

See `prepare_coco_few_shot.py` for generating them manually, but the `cocosplit` folder provided above already includes the splits