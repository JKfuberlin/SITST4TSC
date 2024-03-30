# Tree Species Classification with Convolutional Neural Networks using Satellite Image Time Series

This repository contains scripts and modules to train transformer and lst models, means to apply 
trained models for tree species classification in a somewhat portable state by using Poetry 
for dependency and environment management as well as a scalable, portable and maintainable 
Nextflow Workflow.

Not all of the following sections apply to all usecases.

## Installation

### Python Wheels

**Currently not available!**

### From Repository

#### Pytorch

Unfortunately, the dependency management poetry offers makes the installation of pytorch somewhat cumbersome. By default,
the CUDA 12.1 versions of Pytorch installed. Should you want to install other versions (i.e. CPU wheels or CUDA 11.8),
the following commands are necessary after installation:

```bash
poetry install

# for CUDA 11.8
poetry remove torch torchvision torchaudio
poetry add --source=pytorch_cu118 torch torchvision torchaudio

# for CPU wheels
poetry remove torch torchvision torchaudio
poetry add --source=pytorch_cpu torch torchvision torchaudio
```

To revert back to the CUDA 12.1 wheels, run:

```bash
poetry remove torch torchvision torchaudio
poetry add torch torchvision torchaudio
```

## Usage

### Standalone Scripts

Tree species can be predicted with the standalone `inference.py` script. Currently, inference is possible with **LSTM 
classifier only**. Please note, that a [FORCE](https://force-eo.readthedocs.io/en/latest/) datacube is expected as input.
If you installed `sits_classifier` by cloning this repository and running `poetry install`, you must work within the 
poetry shell which masks the python interpreter. All other installed system binaries are still available to you. 

> Note that other environment managers such as conda should probably be quit beforehand. Thus, running e.g. 
> `conda deactivate` is suggested. 

```bash
poetry shell

python apps/inference.py --help

# usage: inference.py [-h] -w WEIGHTS --input-tiles INPUT [--input-dir BASE] [--input-glob IGLOB] [--output-dir OUT]
#                     --date-cutoff DATE [--mask-dir MASKS] [--mask-glob MGLOB] [--row-size ROW-BLOCK]
#                     [--col-size COL-BLOCK] [--log] [--log-file LOG-FILE] [--cpus CPUS]
# 
# Run inference with already trained LSTM classifier on a remote-sensing time series represented as FORCE ARD
# datacube.
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   -w WEIGHTS, --weights WEIGHTS
#                         Path to pre-trained classifier to be loaded via `torch.load`. Can be either a relative or
#                         absolute file path.
#   --input-tiles INPUT   List of FORCE tiles which should be used for inference. Each line should contain one FORCE
#                         tile specifier (Xdddd_Ydddd).
#   --input-dir BASE      Path to FORCE datacube. By default, use the current PWD.
#   --input-glob IGLOB    Optional glob pattern to restricted files used from `input-dir`.
#   --output-dir OUT      Path to directory into which predictions should be saved. By default, use the current PWD.
#   --date-cutoff DATE    Cutoff date for time series which should be included in datacube for inference.
#   --mask-dir MASKS      Path to directory containing folders in FORCE tile structure storing binary masks with a
#                         value of 1 representing pixels to predict. Others can be nodata or 0. Masking is done on a
#                         row-by-row basis. I.e., the entire unmasked datacube is constructed from the files found in
#                         `input-dir`. Only when handing a row of pixels to the DL-model for inference are data
#                         removed. Thus, this does not reduce the memory footprint, but can speed up inference
#                         significantly under certain conditions.
#   --mask-glob MGLOB     Optional glob pattern to restricted file used from `mask-dir`.
#   --row-size ROW-BLOCK  Row-wise size to read in at once. If not specified, query dataset for block size and assume
#                         constant block sizes across all raster bands in case of multilayer files. Contrary to what
#                         GDAL allows, if the entire raster extent is not evenly divisible by the block size, an error
#                         will be raised and the process aborted. If only `row-size` is given, read the specified
#                         amount of rows and however many columns are given by the datasets block size. If both `row-
#                         size` and `col-size` are given, read tiles of specified size.
#   --col-size COL-BLOCK  Column-wise size to read in at once. If not specified, query dataset for block size and
#                         assume constant block sizes across all raster bands in case of multilayer files. Contrary to
#                         what GDAL allows, if the entire raster extent is not evenly divisible by the block size, an
#                         error will be raised and the process aborted. If only `col-size` is given, read the
#                         specified amount of columns and however many rows are given by the datasets block size. If
#                         both `col-size` and `row-size` are given, read tiles of specified size.
#   --log                 Emit logs?
#   --log-file LOG-FILE   If logging is enabled, write to this file. If omitted, logs are written to stdout.
#   --cpus CPUS           Number of CPUS for Inter-OP and Intra-OP parallelization of pytorch.
```
