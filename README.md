# Solar Irradiance Forecasting


## **Overview**
> PyTorch repository for solar irradiance forecasting task with DVC, PyTorch, Lightning, and Neptune included.


## Table of Contents
* [Requirements](#requirements)
* [Data](#data)
* [Project Structure](#project-structure)
* [Usage](#usage)

## Requirements

* Python *3.10.0*
* Python packages from the *[requirements.txt](./requirements.txt)* file

## Data

<div align="center">

|           **Task**          |                        **Dataset**                       |             **Samples**             | **Used** |
|:---------------------------:|:--------------------------------------------------------:|:-----------------------------------:|:--------:|
| Solar Irradiance Regression |        [Folsom](https://zenodo.org/record/2826939)       |   3 years  (sampled every minute)   |     *    |
| Solar Irradiance Regression |       [SIRTA](https://sirta.ipsl.fr/data-overview/)      | 8 years  (sampled every two minute) |          |
| Solar Irradiance Regression | [Girasol](https://datadryad.org/stash/dataset/doi%253A10.5061%252Fdryad.zcrjdfn9m) | 244 individual days from 3 years period | |
|      Cloud Segmentation     | [SWINySEG](http://vintage.winklerbros.net/swinyseg.html) |                 6768                |          |
|      Cloud Segmentation     |        [HYTA](https://github.com/Soumyabrata/HYTA)       |                  32                 |          |

</div>

The data is stored in the `data` directory. The `data` directory is structured as shown below. Note that the `irradiance.csv` file is only present in the `Folsom` and `SIRTA` datasets and contains the irradiance values for each image. Whereas the `HYTA` and `SWINySEG` datasets contain the masks for each image in the `masks` directory. The `skip_images.txt` file contains the names of the images that should be skipped during training and evaluation due to the lack of ground-truth irradiance, mask or file corruption.

```console
                             +-----------------+
                             | data/Folsom.dvc |*
                       ******+-----------------+ ******
                 ******               *                ******
          *******                     *                      ******
    ******                            *                            ******
****                         +-----------------+                         ****
*                            | clean_dataframe |                            *
*                            +-----------------+                            *
*                                     *                                     *
*                                     *                                     *
*                                     *                                     *
**                           +----------------+                            **
  ***                        | export_periods |                         ***
     ***                     +----------------+                      ***
        ***                ****                ****               ***
           ***          ***                        ***         ***
              **      **                              **     **
        +------------------+                  +---------------------+
        | train_forecaster |                  | export_eval_periods |
        +------------------+                  +---------------------+
```

## Project Structure

```console
├── config
├── data
├── outputs
├── solar_irradiance
|   ├── datamodules
│   │   └── datasets
│   ├── losses
│   ├── models
│   │   └── architectures
│   └── utils
└── tests
    └── unit
```

## Usage

* train

```shell
python scripts/train_forecaster.py --data-root data/Folsom --periods-path data/Prepared/periods.pickle
```
