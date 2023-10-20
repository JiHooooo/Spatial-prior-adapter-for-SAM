# Spatial prior Adapter

## Installation 
1. Create a virtual environment `conda create -n SPA python=3.10 -y` and activate it `conda activate SPA`
2. Install [Pytorch 2.0](https://pytorch.org/)
3. Install the necessary  python package through `pip install -r requirements.txt`
4. compile the CUDA operators for sparse cross attention(https://github.com/fundamentalvision/Deformable-DETR)

## Modify the training config file

see detail in `configs/poly_test_prompt.yaml`

## training code
python3 train_poly_engine.py --c path_to_config_file

## inference code
set the `path_to_config_file` and `path_to_pretrained_model` in `inference.py`

`python3 inference.py`


## Acknowledgements
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.
- We also thank Ma for sharing the baseline code [MedSAM](https://github.com/bowang-lab/MedSAM)

