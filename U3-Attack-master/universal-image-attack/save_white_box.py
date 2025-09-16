import os

import torchvision
from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting, StableDiffusionXLInpaintPipeline
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
from utils import patch_initialization, mask_generation, image_grid, select_location, set_seed
import matplotlib.pyplot as plt
import PIL
import open_clip

device = torch.device("cuda")
totensor = T.ToTensor()
topil = T.ToPILImage()

set_seed(config.seed)

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "../checkpoints/stable-diffusion-inpainting",  # put your model path here
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe_inpaint = pipe_inpaint.to("cuda")


def main(input_path, pipe_inpaint, universal):
    # for ind_patch in range(5):
    ind_patch = 4
    path_patch = "results/{}_patch_period2.pkl".format(ind_patch)
    patch_type = 'rectangle'
    image_size = (3, 512, 512)
    with open(path_patch, "rb") as file:
        list1 = pickle.load(file)
        patch = (list1[0]["best_patch"] + 1.0) * 0.5

    labels = config.labels_test
    prompt_list = config.prompt_list

    save_path_ = "white_save_pictures/adv_{}_patch".format(ind_patch)
    if not os.path.exists(save_path_):
        os.makedirs(save_path_)

    for ind in range(61):
        ind += 2
        # if not universal:
        #     path_patch = "results_onebyone/{}_patch_onebyone.pkl".format(ind)
        #     with open(path_patch, "rb") as file:
        #         list1 = pickle.load(file)
        #         patch = (list1[0]["best_patch"] + 1.0) * 0.5

        label_ind = labels[ind]
        prompt = prompt_list[label_ind]
        print(f"attack image id: {ind}, prompt: {prompt}")

        mask_name = input_path + str(ind) + "_mask.pt"
        mask_name_png_pro = mask_name[:-3] + 'processed_mask.png'
        image_name = input_path + str(ind) + ".png"

        image_scale = 512
        init_image = Image.open(image_name).convert('RGB').resize((image_scale, image_scale))
        mask_image = Image.open(mask_name_png_pro).convert('RGB').resize((image_scale, image_scale))

        init_image = totensor(init_image)
        x, y = select_location(patch_type, patch, image_size)
        applied_patch, mask = mask_generation(patch=patch, x_location=x, y_location=y)
        picture_adv = (1 - mask) * init_image + mask * applied_patch
        picture_adv = topil(picture_adv)

        guidance_scale = 7.5
        num_inference_steps = 100

        content = pipe_inpaint(prompt=[prompt] * 4,
                               image=picture_adv,
                               mask_image=mask_image,
                               eta=1,
                               num_inference_steps=num_inference_steps,
                               guidance_scale=guidance_scale,
                               )

        images = content.images
        for ind_img in range(len(images)):
            images[ind_img].save(save_path_ + "/{}_{}.png".format(ind, ind_img))

        # grid = image_grid(images, 2, 2, image_scale=1024)
        # plt.imshow(grid)
        # plt.show()


path = "./data_patch/test/"
main(path, pipe_inpaint, universal=True)
