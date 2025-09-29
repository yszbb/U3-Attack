# U3-Attack

Official PyTorch implementation of our **ACM MM 2025** paper:  
**[Universally Unfiltered and Unseen: Input-Agnostic Multimodal Jailbreaks against Text-to-Image Model Safeguards](https://arxiv.org/abs/2508.05658)**

---

![Figure](https://github.com/yszbb/U3-Attack/blob/main/asserts/comparison.png)

---

## 📌 Overview

**U3-Attack** presents a universal, input-agnostic multimodal jailbreak technique designed to bypass content moderation in text-to-image generation models.  
It achieves high attack success rates without relying on prompt-specific information, making it highly transferable and effective.

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
# 📥 https://pan.baidu.com/s/12R7RpUf2XsiNUgLDOW8LxA?pwd=ys66
```

3. **Download model weights and datasets:**

```bash
cd universal-image-attack

# Download model weights:
# 📥 https://pan.baidu.com/s/1cjtFmK8hqLcXLCUAFrKyZA?pwd=ys66

# Download datasets:

# Period one datasets:
# Due to sensitive content, this dataset is not publicly available.
# 📥 https://pan.baidu.com/s/1PXgs_5kMV8HXgxzaepHcbg
# Please contact 📧 yan61255873@163.com to request the access code.

# Period two datasets:
# Publicly available:
# 📥 https://pan.baidu.com/s/1OpbRNweuG_ATDwvzjOzodQ?pwd=ys66
# Access code: ys66
```

4. **Load the diffusion model:**

```python
from diffusers import StableDiffusionInpaintPipeline
import torch

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "../checkpoints/stable-diffusion-inpainting",
    revision="fp16",
    torch_dtype=torch.float16,
    safety_checker=None  # Disable built-in safety checker
).to(device)
```

---

## 📁 Image Modality Directory Structure

The `universal-image-attack` folder contains the following structure:

```
universal-image-attack/
├── __pycache__/
├── data/
├── data_patch/
├── heuristic_candidate_set/
├── log/
├── results/
├── results_period_one/
├── weights/
├── config.py
├── dataloader.py
├── recover_img.py
├── save_white_box.py
├── test_total_UMMA.py
├── train_patch_period_one.py
├── train_patch_period_two.py
└── utils.py
```

---

## 🧪 Usage

### 📝 Textual Modality

**1. Train the Paraphrase Set:**

```bash
python train_universal_paraphrase.py
```

**2. Evaluate the Paraphrase Set:**

```bash
python sd_test.py
```

---

### 🖼️ Image Modality

**1. Train the Adversarial Patch**

- **Period One:**

```bash
python train_patch_period_one.py
```

- **Period Two:**

```bash
python train_patch_period_two.py --iter 10 --epsl2 16.0 -s 3 -n 4
```

**2. Evaluate Patch Performance (Inpainting Models):**

```bash
python test_total_UMMA.py
```

---

## 📚 Citation

If you find our work useful, please consider citing:

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

If you have any questions, feel free to reach out:  
📧 **yan61255873@163.com**
