# Bag of tricks for irradiance forecasting


## **Overview**
> Bag of tricks for ground-based solar irradiance forecasting using sky images.


## Table of Contents
* [Requirements](#requirements)
* [Sky image enhancement methods](#sky-image-enhancement-methods)
* [Dataset](#dataset)
* [Usage](#usage)

## Requirements

* Python *3.12.6*
* Python packages defined in the *[pyproject.toml](./pyproject.toml)* file

```bash
pip install -e .[dev]
```

## Sky image enhancement methods

1. Sun mask
2. Irradiance channel
3. Optical flow
4. Cloud channel

## Dataset

<div align="center">

|           **Task**          |                        **Dataset**                       |             **Samples**             | **Used** |
|:---------------------------:|:--------------------------------------------------------:|:-----------------------------------:|:--------:|
| Solar Irradiance Regression |        [Folsom](https://zenodo.org/record/2826939)       |   3 years  (sampled every minute)   |     *    |

</div>

The data is stored in the [data](./data) directory. Preprocess steps are tracked by DVC and described in [dvc.yaml](./dvc.yaml)

### Data Structure

```console
data/
├── raw
├── prepared
└── eval
```

- **raw** - raw irradiance CSV and sky images extracted directly from files downloaded from Folsom [Zenodo](https://zenodo.org/record/2826939) page
- **prepared** - processed data (filtered, timestamp moved to local timezone, corrupted images removed, and grouped into periods)
- **eval** - small subset extracted for evaluation purposes

## Usage

1. **Train**

```shell
python -m scripts.train_forecaster --data-root data/prepared
```

2. **Test**

Change `test_only: True` and provide path to checkpoint `restore_from_ckpt: path/to/checkpoint`. Then call train script.

3. **Evaluate**
