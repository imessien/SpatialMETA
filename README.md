# SpatialMETA

spatialMETA is a method for integrating spatial multi-omics data. SMOI aligns ST and SM to a unified resolution, integrates single or multiple sample data to identify cross-modal spatial patterns, and offers extensive visualization and analysis functions.

<img src="./docs/source/_static/imgs/spatialmeta_2025.png" />

## Documentation

[Documentation](https://spatialmeta.readthedocs.io/en/latest/)

## Installation
Recommended to use Python 3.9 environment.
### Installing via PyPI
```shell
pip3 install spatialmeta
```
### Installing from source
```shell
git clone git@github.com:WanluLiuLab/SpatialMETA.git
cd spatialmeta
pip3 install -r requirements.txt
python3 setup.py install
```

### Create a new environment

```shell
# This will create a new environment named spatialmeta
conda env create -f environment.yml
conda activate spatialmeta
```
## Change Log
- 0.0.2 (2025-03-06)
  - Update `spatialmeta.model.AlignmentModule` 
  - Update `spatialmeta.model.ConditionalVAESTSM`