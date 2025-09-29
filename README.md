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

# Download universal attack weights
# Link: https://pan.baidu.com/s/1cjtFmK8hqLcXLCUAFrKyZA?pwd=ys66

# Download datasets:

# Phase I:
# Due to sensitive content, this dataset is not publicly available.
# You may download it from the link below, but the access code must be requested via email:
# 📥 https://pan.baidu.com/s/1PXgs_5kMV8HXgxzaepHcbg
# Please contact 📧 yan61255873@163.com to request the extraction code.

# Phase II:
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

## 📊 Dataset

We provide two datasets for reproduction and evaluation:

- **Phase I:**  
  Due to sensitive content, this dataset is **not publicly available**.  
  You may download it from the link below, but the **access code must be requested via email**:  
  📥 [Download Link (Baidu Netdisk)](https://pan.baidu.com/s/1PXgs_5kMV8HXgxzaepHcbg)  
  Please contact us via email at 📧 **yan61255873@163.com** to request the extraction code.

- **Phase II:**  
  The Phase II dataset is publicly available:  
  📥 [Download (Baidu Netdisk)](https://pan.baidu.com/s/1OpbRNweuG_ATDwvzjOzodQ?pwd=ys66)  
  Access code: `ys66`
---


## Usage

### 1. Generate NSFW concept

First, generate the NSFW concept for your target concept:

```bash
python vec_gen.py --concept nudity --version 1-5-inpaint --dtype float16
```

### 2. Train Adversarial Generators

#### For Inpainting Models:
```bash
python opt_generator_inpaint.py \
    --ddim_steps 8 \
    --tar_steps 8 \
    --strength 1 \
    --vec_scale 2.5 \
    --concept nudity \
    --mask_dir img_clothes_masks \
    --version 1-5-inpaint \
    --dtype float16 \
    --epoch 100 \
    --lr 1e-5 \
    --eps 64/255 \
    --loss_type mse \
    --prefix ""
```

#### For P2P Models:
```bash
python opt_generator_p2p.py \
    --ddim_steps 8 \
    --tar_steps 8 \
    --strength 1 \
    --vec_scale 2.5 \
    --concept nudity \
    --version p2p \
    --dtype float16 \
    --epoch 100 \
    --lr 1e-5 \
    --eps 64/255 \
    --loss_type mse \
    --prefix ""
```

### 3. Evaluate Attack Performance

#### For Inpainting Models:
```bash
python eval_generator_inpaint.py \
    --ddim_steps 8 \
    --tar_steps 8 \
    --strength 1 \
    --vec_scale 2.5 \
    --concept nudity \
    --mask_dir img_clothes_masks \
    --version 1-5-inpaint \
    --dtype float16 \
    --lr 1e-5 \
    --eps 64/255 \
    --loss_type mse \
    --prefix "eval_gen_time" \
    --ckpt your_checkpoint
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

## 📜 License

This project is released under the MIT License.  
Please see the [LICENSE](./LICENSE) file for more details.

---

## ✉️ Contact

For questions, please contact:  
📧 **yan61255873@163.com**
