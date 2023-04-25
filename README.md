# Segment Anything Model for Medical Image Analysis: an Experimental Study

This is the official repository for paper: [Segment Anything Model for Medical Image Analysis: an Experimental Study](https://arxiv.org/abs/2304.10517).

## Installation

The code requires to install [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything.git). The model and depencency can be found at SAM's repository, or you can install with

```
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything; pip install -e .
```

Optinally, we have included the code to evaluate [Reviving Iterative Training with Mask GUidance for Interactive Segmentation (RITM)](https://arxiv.org/abs/2102.06583). It was imported directly in the script, so all you need to do is to clone the repository locally.

```
git clone https://github.com/yzluka/ritm_interactive_segmentation
```

## Getting start
First, download SAM's model checkpoint 
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

To run the code, 
```
python3 prompt_gen_and_exec_v1.py --num-prompt XXX --model sam/ritm 
```
And it will ask you to enter the dataset you want to evaluate on.

Optionally, to run RITM, you need to download its weights by
```
wget https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/coco_lvis_h32_itermask.pth
```

## Getting dataset from the paper

TODO

## Getting custom dataset
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

## Citation
If you find this work is useful for your research, please cite our paper:
```
@inproceedings{Mazurowski2023SegmentAM,
  title={Segment Anything Model for Medical Image Analysis: an Experimental Study},
  author={Maciej A. Mazurowski and Haoyu Dong and Han Gu and Jichen Yang and N. Konz and Yixin Zhang},
  hournal={arXiv:2304.10517},
  year={2023}
}
```
