# Solar Irradiance Forecasting


## **Overview**
> PyTorch repository for solar irradiance and cloud segmentation tasks with PyTorch Lightning, Hydra and Neptune included.


## Table of Contents
* [Requirements](#Requirements)
* [Data](#Data)
* [Structure](#Structure)
* [Usage](#Usage)

## Requirements

* Python *3.10.0*
* Python packages from the *[requirements.txt](./requirements.txt)* file

## Data

<div align="center">

|           **Task**          |                        **Dataset**                       |             **Samples**             | **Description** |
|:---------------------------:|:--------------------------------------------------------:|:-----------------------------------:|:---------------:|
| Solar Irradiance Regression |        [Folsom](https://zenodo.org/record/2826939)       |   3 years  (sampled every minute)   |                 |
| Solar Irradiance Regression |       [SIRTA](https://sirta.ipsl.fr/data-overview/)      | 8 years  (sampled every two minute) |                 |
| Solar Irradiance Regression | [Girasol](https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.zcrjdfn9m) | 244 individual days from 3 years period | |
|      Cloud Segmentation     | [SWINySEG](http://vintage.winklerbros.net/swinyseg.html) |                 6768                |                 |
|      Cloud Segmentation     |        [HYTA](https://github.com/Soumyabrata/HYTA)       |                  32                 |                 |

</div>

The data is stored in the `data` directory. The `data` directory is structured as shown below. Note that the `irradiance.csv` file is only present in the `Folsom` and `SIRTA` datasets and contains the irradiance values for each image. Whereas the `HYTA` and `SWINySEG` datasets contain the masks for each image in the `masks` directory. The `skip_images.txt` file contains the names of the images that should be skipped during training and evaluation due to the lack of ground-truth irradiance, mask or file corruption.

```console
├── data
│   └── <DATASET NAME>
│       ├── images
│       ├── masks
|       ├── irradiance.csv
|       └── skip_images.txt
```

## Project Structure

```console
├── config
├── data
├── outputs
├── src
|   └── solar_irradiance
│       ├── datamodules
│       │   └──  datasets
│       ├── losses
│       ├── metrics
│       ├── models
│       └── utils
└── tests
    └── unit
```

## Usage

* train

```shell
HYDRA_FULL_ERROR=1 python src/main.py --config-name regressor
```

* evaluate

```shell
HYDRA_FULL_ERROR=1 python src/main.py --config-name regressor test_only=true restore_from_ckpt=/home/path/to/checkpoint.ckpt
```

* export

```shell
HYDRA_FULL_ERROR=1 python src/main.py --config-name regressor test_only=true restore_from_ckpt=/home/path/to/checkpoint.ckpt export.export_to_onnx=true
```
