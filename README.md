# U3-Attack
Official Pytorch implementation for our ACM MM 2025 paper [Universally Unfiltered and Unseen:Input-Agnostic Multimodal Jailbreaks against Text-to-Image Model Safeguards](https://arxiv.org/abs/2508.05658).

![Figure](https://github.com/yszbb/U3-Attack/blob/main/asserts/comparison.png)

## Requirements
- python 3.8
- Pytorch 1.13
- At least 1x12GB NVIDIA GPU
## Installation
```
git clone https://github.com/yszbb/AE-Patch
cd AE-Patch
pip install -r requirements.txt
```
## Preparation
### Dataset
The [SpikingJelly framework](https://github.com/fangwei123456/spikingjelly) automatically downloads the dataset to a designated folder. You can customize the download location by adjusting the root parameter. This parameter can be found on lines 85 and 87 of the [tutorial_every_class_untargeted.py](https://github.com/yszbb/AE-Patch/blob/main/tutorial_every_class_untargeted.py) file.
### Model
1. Download the .
2. Fine-tune .
### train and val
Once you have setup your path, you can run an experiment like so:
```
python patch_untargeted_everyone.py --arch "vggdvs" --datadvs "dvscifar" 
```

