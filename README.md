# 3D U-Net for Fetal Brain Segmentation

Code used for the following paper: [Development of Gestational Age–Based Fetal Brain and Intracranial Volume Reference Norms Using Deep Learning](http://www.ajnr.org/content/early/2022/12/21/ajnr.A7747). See [Citation](#citation).

Table of Contents
=================

   * [3D U-Net](#3d-u-net)
      * [Description](#description)
      * [Installation](#installation)
         * [Code](#code)
         * [Python Environment](#python-environment)
         * [Directory structure](#directory-structure)
      * [Usage](#usage)
         * [Tensorflow 2 version](#tensorflow-2-version)
         * [Configuration](#configuration)
         * [Split file](#split-file)
         * [Start training & testing](#start-training--testing)
         * [Outputs](#outputs)
      * [Others](#others)
         * [Tensorboard](#tensorboard)
            * [Launch](#launch)
            * [Read](#read)
            * [Setup](#setup)
      * [Contributing](#contributing)
      * [Citation](#citation)
## Description

This UNET has been developed to automatically segment fetal brain from 3D MRI images. This repository contains code to train a new model, continue the training, fine tune the training with frozen layers, and test a model with new images. It uses a cross entropy loss and computes a soft dice score during training, and regular dice score for the test part.

## Installation
You can either install the environment manually, or if you're interested in predictions and already have a model checkpoint use the provided `Singularity.def` file to build a container with all dependencies.

### Get code
Download this code directly, or use the following:
```sh
cd /directory/where/you/want/the/code
# with UCSF GitHub, you have to use SSH connection
git clone git@github.com:rauschecker-sugrue-labs/fetal-brain-segmentation.git
# to update later, simply use:
git pull origin
```
The code is accessible inside the `unet` directory, with a `START_HERE.py` file to help getting started. [See more here](#usage).

### Singularity container
Run the following command to build the container:
```sh
apptainer build unet.sif Singularity.def
# this one time command takes a couple of minutes to run and creates a container file
```
Then, to run predictions:
```sh
./predict.sh \
   path/to/data/inputs \
   path/to/data/outputs \
   path/to/model/checkpoint/dir
# you can add here the GPU_id to use as well as number of CPUs for preprocessing
```
### Python Environment
A valid `conda` install is needed (Anaconda, Miniconda).  
Install the environment from the `unet.yml` file.
```sh
# Standard install:
conda env create -f unet.yml
# Specify where to install:
conda env create --prefix /path/to/install -f unet.yml
```
### Directory structure
The code and configuration is set up to work best with a specific directory structure, described in [Configuration](#configuration).

## Usage
See [Singularity container section](#singularity-container) if only doing predictions.

### Tensorflow 2 version

Everything can be done from the `START_HERE.py` file. There are 2 actions to perform: load a `Config` object, and call `Config.start()`.

1. Load config: `c = Config(your_parameters)`. Specify the run parameters here, cf. [config](#configuration) section.
2. Launch training or testing: `c.start()`

### Configuration

Most configuration happens in the file `config.py`, with the object `Config(object)`. It contains train/test setup (paths, *etc.*), along with most model parameters (batch size, # epochs, learning rate, *etc.*).

Before running any experiment, a csv split file *must* exist in the main folder (`models/` or `predict/`).

The directories are organized in the following way:  
```
rootdir/ ## referred to as *root*
├── Data
│   ├── my_image_data
│   │   └── raw
│   └── my_image_data2
│        └── raw
├── models
│   ├── my_train_set
│   ├── my_train_set.csv
│   └── etc.
├── predict
│    └── etc.
└── fetal-brain-segmentation
```
A call to the configuration object could look like the following:
```py
# Training
c = Config(root='path/to/root',
            datasource='my_image_data',
            experiment='my_train_set',
            train=False,
            GPU='5,7')
c.print_recap()

# Testing
c = Config(root='path/to/root',
            datasource='my_image_data',
            experiment='my_test_set',
            train=False,
            model='my_train_set',
            which_model='latest',
            GPU='5,7')
c.print_recap()
```
Configuration call options:  

| Variable    | Type         | Default  | Description                                                                                                              | Usage                                                                                 | Required |
|-------------|--------------|----------|--------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|----------|
| root      | str          |          | Path to root directory (above Data/, models/)                                                                                    |                                                                                       | ✓      |
| datasource      | str          |          | Directory name for the source of data                                                                                    |                                                                                       | ✓      |
| experiment  | str          |          | Name of experiment (if train: model name, if predict: prediction name)                                                   |                                                                                       | ✓      |
| train       | bool         |          | True for train, False for predict                                                                                        |                                                                                       | ✓      |
| model       | str          | None     | If train = False, this is the model name used to predict the data                                                        |                                                                                       |          |
| move_images | bool         | True     | For predict only: move images into directory to allow processing (needs to be ran at least once per predict experiment). |                                                                                       |          |
| tmp_storage | str          | None     | Location of tfrecords                                                                                                    | *None=in main exp directory. 'TMP'=$TMPDIR.         |          |
| which_model | str          | 'latest' | Chooses which model to load for predict                                                                                  | 'latest' -> latest \| 'root' if checkpoint is at the root \| 'your_name' -> this name |          |
| read_only   | bool         | False    | If set to True, folders won't be created on init. To use only if the goal is to read data from a previous run            |                                                                                       |          |
| GPU         | int/list int | None     | Chooses which GPU(s) will be used for this session                                                                       | If None, all available GPUs will be used. GPU=0 \| GPU=[0,1] \| GPU='1' accepted      |          |

### Split file
The split file is a csv file that specifies filenames (e.g. 10101010_FLAIR.nii.gz) and whether that file represents a training, validation, or test case.  The split file must be saved in either the model/ or the predict/ directory. It must have the same name as the input of `experiment` variable in the `Config` call.  
It doesn't have headers, and consists of 3 columns: 
* first: image file name without extension
* second: disease class (not being used currently so could be NA)
* third: train, val, or test (for model training, validation during training, or testing)

### Start training & testing
Make your own copy of [START_HERE.py](START_HERE.py).
```py
## Load module
from utils.config import Config
```
#### Training
```py
## Config
# ... for training (OR)
c = Config(root='path/to/root',
            datasource='my_image_data',
            experiment='my_train_set',
            train=True,
            GPU='5,7')

# (OR) ... for continued training
c = Config(root='path/to/root',
            datasource='my_image_data',
            experiment='my_train_set',
            train=True,
            model='my_train_set',
            which_model='latest',
            GPU='5,7')
            
## Check that everything is setup correctly
c.print_recap(notes='notes_run')

## Start training
c.start(notes_run='notes_run')
```
#### Testing
If you are planning to simply apply a pre-trained model (such as the latest version of the FLAIR U-net) on a new dataset, you can do so as follows:
```py
## Config for testing
c = Config(root='path/to/root',
            datasource='my_image_data',
            experiment='my_test_set',
            train=False,
            model='my_train_set',
            which_model='latest',
            GPU='5,7')

## Check that everything is setup correctly
c.print_recap(notes='notes_run')

## Start testing
c.start(notes_run='notes_run')
```

### Outputs
**Training output:**
```
rootdir/models/model_name
├── config.txt
├── model
│   ├── date1
│   ├── date2
│   ├── checkpoint
│   ├── cp-030.ckpt.data-00000-of-00001
│   └── cp-030.ckpt.index
├── tfboard
│   ├── date1
│   ├── date2
│   └── date3
├── tf_records
└── validation_output
    ├── binarized_masks
    ├── Dice_score
    ├── predictions
    └── resampled_to_originalspacing
```
**Prediction outputs:**
```
rootdir/predict/Tr-model_name_Tt-pred_name/
├── binarized_masks
├── config.txt
├── Dice_score
├── predictions
├── preprocessed
├── raw
├── resampled_to_originalspacing
└── tf_records
```
**Tensorboard output:**  
See Tensorboard section.

## Others
### Tensorboard
#### Launch
```sh
source activate unet
tensorboard --logdir /path/to/model_dir/tfboard
```
It will open a port to be loaded in a browser:
```
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.4.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

If the data is on a server (e.g. SCS server), a few options:
* open a graphic interface to this server, launch terminal, run the above commands, and open a web browser at the indicated address,
* if using `vscode`, while being connected through `ssh` to the server, run the above commands, and `vscode` will automatically forward the open port to a local port on your computer,
* if using a local terminal, follow these steps, after having run the above steps *via* `ssh`:
```sh
# XX = port provided after running the $tensorboard command (usually 06, 07, etc.)
# server = server on which the model is training (e.g. callosum.radiology.ucsf.edu)
ssh -L 160XX:127.0.0.1:60XX server
# In your browser, go to:  http://127.0.0.1:160XX/
```

#### Read
* Scalars: graphs for loss, metrics, across batches, epochs, time.
* Graphs: model graph.
* Distribution & histograms: model weight evolution layer by layer across epochs. See more here: https://github.com/tensorflow/tensorboard/blob/master/docs/r1/histograms.md


#### Setup
In file `train_test.py`, function `train()`, there is a *Tensorboard Callback* `tb_callback`.


## Contributing
Edits and suggestions are welcome, using GitHub branch system to ask for merge.

## Citation
If you use our code, please cite [our paper](http://www.ajnr.org/content/early/2022/12/21/ajnr.A7747):  

C.B.N. Tran, P. Nedelec, D.A. Weiss, J.D. Rudie, L. Kini, L.P. Sugrue, O.A. Glenn, C.P. Hess, A.M. Rauschecker, [Development of Gestational Age–Based Fetal Brain and Intracranial Volume Reference Norms Using Deep Learning](http://www.ajnr.org/content/early/2022/12/21/ajnr.A7747). DOI:10.3174/ajnr.A7747.  

```
@article {Tran,
	author = {Tran, C.B.N. and Nedelec, P. and Weiss, D.A. and Rudie, J.D. and Kini, L. and Sugrue, L.P. and Glenn, O.A. and Hess, C.P. and Rauschecker, A.M.},
	title = {Development of Gestational Age{\textendash}Based Fetal Brain and Intracranial Volume Reference Norms Using Deep Learning},
	year = {2022},
	doi = {10.3174/ajnr.A7747},
	publisher = {American Journal of Neuroradiology},
	issn = {0195-6108},
	URL = {http://www.ajnr.org/content/early/2022/12/21/ajnr.A7747},
	eprint = {http://www.ajnr.org/content/early/2022/12/21/ajnr.A7747.full.pdf},
	journal = {American Journal of Neuroradiology}
}
```
