import os

import torchvision
from diffusers import StableDiffusionInpaintPipeline
from torch import nn
import numpy as np
import config
from PIL import Image
from utils import prepare_mask_and_masked_image, select_location, mask_generation
import torch
from typing import Union, List
import random
import pickle
import torchvision.transforms as T
from utils import patch_initialization, mask_generation, image_grid, select_location, set_seed, recover_image
import matplotlib.pyplot as plt
import PIL
import open_clip

safety_checker = torch.load("weights/safety_checker.pt")
concept_embeds = torch.load("weights/concept_embeds.pt")
concept_embeds_weights = torch.load("weights/concept_embeds_weights.pt")  # ! 概念编码阈值 threshold
device = torch.device("cuda")
totensor = T.ToTensor()
topil = T.ToPILImage()

re = torchvision.transforms.Resize([224, 224])
set_seed(config.seed)

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "../checkpoints/stable-diffusion-inpainting",  # put your model path here
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe_inpaint = pipe_inpaint.to("cuda")

text_encoder = pipe_inpaint.text_encoder
tokenizer = pipe_inpaint.tokenizer
text_encoder = text_encoder.to("cuda")

cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)


def norm(images):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    normlize = torchvision.transforms.Normalize(clip_mean, clip_std)
    imgs = normlize(images)
    return imgs


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(input_path, pipe_inpaint, detector, universal, multimodal):
    # pipeline patch
    # path_patch = "results_pipeline/0_patch_pipe.pkl"
    # sample patch
    path_patch = "results/4_patch_period2.pkl"
    # init patch ##################
    # path_patch = "results_period_one/best_patch.pkl"
    patch_type = 'rectangle'
    image_size = (3, 512, 512)
    with open(path_patch, "rb") as file:
        list1 = pickle.load(file)
        patch = (list1[0]["best_patch"] + 1.0) * 0.5
        # patch = list1[0]["best_patch"]
    # patch = patch_initialization()
    labels = config.labels_test
    if not multimodal:
        prompt_list = config.prompt_list
        print(prompt_list)

    for ind in range(61):
        # #   #############################
        # path_show = "show/{}/".format(ind)
        # if not os.path.exists(path_show):
        #     os.makedirs(path_show)

        ind += 2

        label_ind = labels[ind]
        prompt = prompt_list[label_ind]
        print(f"attack image id: {ind}, prompt: {prompt}")

        mask_name = input_path + str(ind) + "_mask.pt"
        mask_name_png_pro = mask_name[:-3] + 'processed_mask.png'
        image_name = input_path + str(ind) + ".png"

        init_image = Image.open(image_name).convert('RGB').resize((512, 512))
        # # ################################
        # init_image.save(path_show+"init.png")
        mask_image = Image.open(mask_name_png_pro).convert('RGB').resize((512, 512))
        # ################################
        mask_image.save("image_mask/mask_{}.png".format(ind))

        # ################################
        # picture_adv.save("adv_pictures/adv_{}.png".format(ind))
        guidance_scale = 7.5
        num_inference_steps = 100

        content = pipe_inpaint(prompt=[prompt] * 1,
                               image=init_image,
                               mask_image=mask_image,
                               eta=1,
                               num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale,
                               )
        images = content.images
        ###############################
        images[0].save("image_mask/{}.png".format(ind))
        grid = image_grid(images, 1, 1, 512)
        # grid.save("image_mask/{}.png".format(ind))
        plt.imshow(grid)
        plt.show()


path = "./data_patch/test/"
main(path, pipe_inpaint, detector="q16", universal=True, multimodal=False)
