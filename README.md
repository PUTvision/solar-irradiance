# Solar Irradiance Forecasting


## **Overview**
> Bag of tricks for ground-based solar irradiance forecasting using sky images.


## Table of Contents
* [Requirements](#requirements)
* [Methods](#methods)
* [Data](#data)
* [Usage](#usage)

## Requirements

* Python *3.12.6*
* Python packages defined in the *[pyproject.toml](./pyproject.toml)* file

## Methods

1. Sun mask
2. Irradiance channel
3. Optical flow
4. Cloud channel

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

The data is stored in the [data](./data) directory. Preprocess steps are tracked by DVC and described in [dvc.yaml](./dvc.yaml)

### Data Structure

- **Raw** - irradiance CSV and sky images extracted directly from files downloaded from [Folsom](https://zenodo.org/record/2826939) dataset
- **Prepared** - processed data (filtered, timestamp moved to local timezone, grouped in periods)
- **Eval** - small subset extracted for evaluation purposes

## Usage

* train

```shell
python -m scripts.train_forecaster --data-root data/Prepared
```
