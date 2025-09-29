# U3-Attack

Official PyTorch implementation of our **ACM MM 2025** paper:  
**[Universally Unfiltered and Unseen: Input-Agnostic Multimodal Jailbreaks against Text-to-Image Model Safeguards](https://arxiv.org/abs/2508.05658)**

---

![Figure](https://github.com/yszbb/U3-Attack/blob/main/asserts/comparison.png)

---

## 📌 Overview

**U3-Attack** introduces a universal, input-agnostic multimodal jailbreak method that bypasses safeguards in text-to-image generation models.  

---

## 🛠️ Requirements

- Python ≥ 3.8  
- PyTorch ≥ 2.3.1  
- At least **1× NVIDIA GPU with 24GB** memory

---

## 🚀 Installation

1. **Clone this repository and install dependencies:**

```bash
git clone https://github.com/yszbb/U3-Attack
cd U3-Attack
pip install -r requirements.txt
```

2. **Download the required models:**

```bash
cd checkpoints
git-lfs install
git clone https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting
git clone https://huggingface.co/openai/clip-vit-base-patch32

# Download ViT-L/14 weights (Baidu Netdisk)
# Link: https://pan.baidu.com/s/12R7RpUf2XsiNUgLDOW8LxA?pwd=ys66
```

3. **Download attack weights and datasets:**

```bash
cd universal-image-attack

# Download model weights
# Link: https://pan.baidu.com/s/1cjtFmK8hqLcXLCUAFrKyZA?pwd=ys66

# Download datasets:

# Period One:
# Due to sensitive content, this dataset is not publicly available.
# You may download it from the link below, but the access code must be requested via email:
# 📥 https://pan.baidu.com/s/1PXgs_5kMV8HXgxzaepHcbg
# Please contact 📧 yan61255873@163.com to request the extraction code.

# Period Two:
# Publicly available dataset:
# 📥 https://pan.baidu.com/s/1OpbRNweuG_ATDwvzjOzodQ?pwd=ys66
# Access code: ys66
```

4. **Load the diffusion model:**

```python
from diffusers import StableDiffusionInpaintPipeline

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "/checkpoints/stable-diffusion-inpainting"
).to(device)
```
---


## Usage

### Textual Modality
#### 1. Train Paraphrase Set

```bash
python train_universal_paraphrase.py
```
#### 2. Evaluate Paraphrase Set

```bash
python sd_test.py
```

### Image Modality
#### 1. Train Adversarial Patch

##### Period One:
```bash
python train_patch_period_one.py
```

##### Period Two:
```bash
python train_patch_period_two.py --iter 10 --epsl2 16.0 -s 3 -n 4
```

#### 2. Evaluate Patch Performance

#### For Inpainting Models:
```bash
python test_total_UMMA.py 
```

## 📚 Citation

If you find our work helpful, please cite the following paper:

```bibtex
@article{yan2025universally,
  title={Universally Unfiltered and Unseen: Input-Agnostic Multimodal Jailbreaks against Text-to-Image Model Safeguards},
  author={Yan, Song and Wei, Hui and Fei, Jinlong and Yang, Guoliang and Zhao, Zhengyu and Wang, Zheng},
  journal={arXiv preprint arXiv:2508.05658},
  year={2025}
}
```

---


## ✉️ Contact

For questions, please contact:  
📧 **yan61255873@163.com**
