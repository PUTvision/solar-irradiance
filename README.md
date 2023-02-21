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

|           **Task**           |                        **Dataset**                       |             **Samples**             | **Description** |
|:----------------------------:|:--------------------------------------------------------:|:-----------------------------------:|:---------------:|
| Solar Irradiance  Regression |        [Folsom](https://zenodo.org/record/2826939)       |   3 years  (sampled every minute)   |                 |
| Solar Irradiance  Regression |       [SIRTA](https://sirta.ipsl.fr/data-overview/)      | 8 years  (sampled every two minute) |                 |
|      Cloud Segmentation      | [SWINySEG](http://vintage.winklerbros.net/swinyseg.html) |                 6768                |                 |
|      Cloud Segmentation      |        [HYTA](https://github.com/Soumyabrata/HYTA)       |                  32                 |                 |

</div>

The data is stored in the `data` directory. The `data` directory is structured as shown below. Note that the `irradiance.csv` file is only present in the `Folsom` and `SIRTA` datasets and contains the irradiance values for each image. Whereas the `HYTA` and `SWINySEG` datasets contain the masks for each image in the `masks` directory.

```console
├── data
│   └── <DATASET NAME>
│       ├── images
│       ├── masks
|       └── irradiance.csv
```

## Project Structure

```console
├── config
├── data
├── scripts
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
