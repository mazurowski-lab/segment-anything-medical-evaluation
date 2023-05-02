# Segment Anything Model for Medical Image Analysis: an Experimental Study

[![arXiv Paper](https://img.shields.io/badge/arXiv-2304.10517-orange.svg?style=flat)](https://arxiv.org/abs/2304.10517)

#### by [Maciej Mazurowski](https://sites.duke.edu/mazurowski/), Haoyu Dong, Hanxue Gu, Jichen Yang, [Nicholas Konz](https://nickk124.github.io/) and Yixing Zhang.

This is the official repository for our paper: [Segment Anything Model for Medical Image Analysis: an Experimental Study](https://arxiv.org/abs/2304.10517), where we evaluated Meta AI's Segment Anything Model (SAM) on many medical imaging datasets.

## Installation

The code requires installing SAM's repository [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything.git). The model and dependencies can be found at SAM's repository, or you can install them with

```
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

Optionally, we have included code to evaluate [Reviving Iterative Training with Mask Guidance for Interactive Segmentation (RITM)](https://arxiv.org/abs/2102.06583) on the datasets. All you need to do to use our code for this is to clone their repository locally:

```
git clone https://github.com/yzluka/ritm_interactive_segmentation
```

## Getting start
First, download SAM's model checkpoint 
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Run the code with:
```
python3 prompt_gen_and_exec_v1.py --num-prompt XXX --model sam/ritm 
```
where it will ask you to enter the dataset you wish to evaluate on.

Optionally, to run RITM, you need to download its weights via:
```
wget https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h32_itermask.pth
```

## Obtaining datasets from our paper

TODO

## Adding custom datasets
To evaluate your own dataset, you need to format the dataset as: 
```
  XXX:
     images:
        abc.png
        def.png
        ...
     masks:
        abc.png
        def.png
        ...
```
where images and masks should have the same name.

## Citation
If you find our work to be useful for your research, please cite our paper:
```
@inproceedings{Mazurowski2023SegmentAM,
  title={Segment Anything Model for Medical Image Analysis: an Experimental Study},
  author={Maciej A. Mazurowski and Haoyu Dong and Han Gu and Jichen Yang and N. Konz and Yixin Zhang},
  hournal={arXiv:2304.10517},
  year={2023}
}
```
