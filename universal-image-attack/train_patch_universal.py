from colorama import init, Fore, Back, Style

init(autoreset=True)

import os
from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
import torchvision.transforms.transforms
import torchvision
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as T
from typing import Union, List, Optional, Callable
import torch.nn as nn
import argparse
import cv2
from rich import print
from utils import preprocess, prepare_mask_and_masked_image, recover_image, prepare_image, get_logger, select_location, \
    mask_generation, patch_initialization
import config
import pickle
from dataloader import PatchDataset
from torch.utils.data import DataLoader

to_pil = T.ToPILImage()
totensor = T.ToTensor()
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "../checkpoints/stable-diffusion-inpainting",  # put your model path here
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe_inpaint = pipe_inpaint.to("cuda")
# please download the "safety_checker.pt" from the hugging_face page listed in README.md
safety_checker = torch.load("weights/safety_checker.pt")
concept_embeds = torch.load("weights/concept_embeds.pt")
concept_embeds_weights = torch.load("weights/concept_embeds_weights.pt")  # ! threshold

feature_extractor = torch.load("weights/feature_extractor.pt")
special_care_embeds = torch.load("weights/special_care_embeds.pt")
special_care_embeds_weights = torch.load("weights/special_care_weights.pt")  # ! threshold
processor = torch.load("weights/open_clip_preprocess.pt")

labels = config.labels_train
device = torch.device("cuda")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denoise_and_dilate(mask, kernel_size=5):
    # Create a square kernel with the specified size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Remove small regions using opening
    denoised_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Dilate the mask using the kernel
    dilated_mask = cv2.dilate(denoised_mask, kernel, iterations=1)

    return dilated_mask


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


def decode_latents(self, latents):
    latents = 1 / self.vae.config.scaling_factor * latents
    image = self.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images_source to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images_source
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def safety_checker_loss(clip_input, images, args=None):
    pooled_output = safety_checker.vision_model(clip_input)[1]  # pooled_output
    image_embeds = safety_checker.visual_projection(pooled_output)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    cos_dist = cosine_distance(image_embeds, concept_embeds)  # 1 * 17

    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}
        # increase this value to create a stronger nfsw filter
        # at the cost of increasing the possibility of filtering begign images_source
        adjustment = args.adjustment
        attack_loss = 0
        loss_total = 0
        for concept_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concept_idx]
            concept_threshold = concept_embeds_weights[concept_idx].item()  # 0.18
            result_img["concept_scores"][concept_idx] = torch.round(concept_cos - concept_threshold + adjustment,
                                                                    decimals=3)
            loss_total = loss_total + concept_cos
            if result_img["concept_scores"][concept_idx] > 0:
                attack_loss = attack_loss + concept_cos
        if attack_loss:
            # print(Back.YELLOW+"Detected NSFW objects, return attack_loss")
            return attack_loss
        else:
            print(Back.GREEN + "Loss total")
            return loss_total


def attack_forward(
        self,
        prompt: Union[str, List[str]],
        masked_image: Union[torch.FloatTensor, Image.Image],
        mask: Union[torch.FloatTensor, Image.Image],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
):
    # guidance_scale=7.5 num_inference_steps=8 eta=1
    # 图像的生成过程
    text_inputs = self.tokenizer(
        prompt,
        padding="max_length",
        max_length=self.tokenizer.model_max_length,
        return_tensors="pt",
    )
    # 1 * 77
    text_input_ids = text_inputs.input_ids
    # 1 * 77 * 768
    text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

    uncond_tokens = [""]
    max_length = text_input_ids.shape[-1]
    uncond_input = self.tokenizer(
        uncond_tokens,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    # 1 * 77 * 768
    uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

    seq_len = uncond_embeddings.shape[1]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])  # 2 * 77 * 768

    text_embeddings = text_embeddings.detach()

    num_channels_latents = self.vae.config.latent_channels  # 4

    latents_shape = (1, num_channels_latents, height // 8, width // 8)  # 1 * 4 * 64 * 64
    latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)

    mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))  # 1 * 1 * 64 * 64
    mask = torch.cat([mask] * 2)  # mask.shape = [2, 1, 64, 64]

    masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()  # 1 4 64 64

    masked_image_latents = 0.18215 * masked_image_latents
    masked_image_latents = torch.cat([masked_image_latents] * 2)  # 2 4 64 64
    # self.scheduler.init_noise_sigma : 1.0
    latents = latents * self.scheduler.init_noise_sigma
    a = latents
    self.scheduler.set_timesteps(num_inference_steps)
    timesteps_tensor = self.scheduler.timesteps.to(self.device)

    for i, t in enumerate(timesteps_tensor):
        latent_model_input = torch.cat([latents] * 2)  # shape [2, 4, 64, 64]
        latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)  # shape [2, 9, 64, 64]
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

    latents = 1 / 0.18215 * latents
    b = latents
    image = self.vae.decode(latents).sample
    return image


def compute_grad(cur_mask, cur_masked_image, patch, prompt, pipe_inpaint, x, y, **kwargs):
    # guidance_scale=7.5 num_inference_steps=8 eta=1
    applied_patch, mask_patch = mask_generation(patch=patch.clone(), x_location=x, y_location=y)
    applied_patch = applied_patch.to(device)
    mask_patch = mask_patch.to(device)

    perturbated_image = (torch.mul(mask_patch.type(torch.float16), applied_patch.type(torch.float16)) +
                         torch.mul((1 - mask_patch.type(torch.float16)), cur_masked_image))

    image_gen = attack_forward(pipe_inpaint, mask=cur_mask,
                               masked_image=perturbated_image,
                               prompt=prompt,
                               **kwargs)

    torch.set_grad_enabled(True)
    patch_new = patch.clone().half().cuda()
    patch_new.requires_grad_()

    applied_patch_op, mask_patch_op = mask_generation(patch=patch_new, x_location=x, y_location=y)
    applied_patch_op = applied_patch_op.to(device)
    loss_total = 0.

    for ind in range(image_gen.shape[0]):
        var_patch = torch.mul(mask_patch.type(torch.float16), image_gen[ind]) - applied_patch.type(torch.float16)

        applied_patch_op_new = applied_patch_op + var_patch

        image_nat = torch.mul(mask_patch.type(torch.float16), applied_patch_op_new.type(torch.float16)) + torch.mul(
            (1 - mask_patch.type(torch.float16)), image_gen[ind])

        image_nat = image_nat.unsqueeze(0)
        image_nat_invnorm = image_nat / 2 + 0.5  # ! [0,1]
        # save_batch_img(image_nat_invnorm, "save_path")  # +++++++++++++++++++++++++++++
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        re = torchvision.transforms.Resize([224, 224])
        normlize = torchvision.transforms.Normalize(clip_mean, clip_std)
        image_nat_re = re(image_nat_invnorm)
        image_nat_normlize = normlize(image_nat_re)
        # safety_checker_input = feature_extractor(numpy_to_pil(image_nat_num), return_tensors="pt").to("cuda")
        # #! safety_checker_input is a dict with keys: 'pixel_values' tensor shaped as [1, 3, 224, 224]

        loss = safety_checker_loss(images=image_nat_normlize, clip_input=image_nat_normlize.to(torch.float16),
                                   args=args)
        loss_total += loss

    grad = torch.autograd.grad(loss_total, [patch_new])[0]

    return grad, (loss_total.item()) / image_gen.shape[0], image_nat.data.cpu()


def compute_grad_pipeline(cur_mask, cur_masked_image, patch, prompt, pipe_inpaint, x, y, **kwargs):
    # guidance_scale=7.5 num_inference_steps=8 eta=1
    torch.set_grad_enabled(True)
    cur_mask = cur_mask.clone()
    cur_masked_image = cur_masked_image.clone()
    patch = patch.clone().half().cuda()

    cur_mask.requires_grad = False
    cur_masked_image.requires_grad = False
    patch.requires_grad_()

    applied_patch, mask_patch = mask_generation(patch=patch, x_location=x, y_location=y)
    applied_patch = applied_patch.to(device)
    mask_patch = mask_patch.to(device)

    perturbated_image = torch.mul(mask_patch.type(torch.float16), applied_patch.type(torch.float16)) + torch.mul(
        (1 - mask_patch.type(torch.float16)), cur_masked_image)

    image_nat = attack_forward(pipe_inpaint, mask=cur_mask,
                               masked_image=perturbated_image,
                               prompt=prompt,
                               **kwargs)

    image_nat_invnorm = image_nat / 2 + 0.5  # ! [0,1]
    # save_batch_img(image_nat_invnorm, "save_path")  # +++++++++++++++++++++++++++++
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    re = torchvision.transforms.Resize([224, 224])
    normlize = torchvision.transforms.Normalize(clip_mean, clip_std)
    image_nat_re = re(image_nat_invnorm)
    image_nat_normlize = normlize(image_nat_re)
    # safety_checker_input = feature_extractor(numpy_to_pil(image_nat_num), return_tensors="pt").to("cuda")
    # #! safety_checker_input is a dict with keys: 'pixel_values' tensor shaped as [1, 3, 224, 224]

    loss = safety_checker_loss(images=image_nat_normlize, clip_input=image_nat_normlize.to(torch.float16), args=args)
    grad = torch.autograd.grad(loss, [patch])[0]

    return grad, loss.item(), image_nat.data.cpu()


def super_l2(cur_mask, cur_masked_image, patch, prompt, step_size, iters, eps, clamp_min, clamp_max, pipe_inpaint, x, y,
             grad_reps=5, pipeline=False, **kwargs):
    loss_best = 1000.0
    patch_best = None
    patch_adv = patch.clone().cuda()
    #  计算梯度更新patch
    iterator = tqdm(range(iters))
    for i in iterator:
        all_grads = []
        losses = []
        for i in range(grad_reps):
            if not pipeline:
                c_grad, loss, last_image = compute_grad(cur_mask, cur_masked_image, patch_adv, prompt, pipe_inpaint, x,
                                                        y, **kwargs)
            else:
                c_grad, loss, last_image = compute_grad_pipeline(cur_mask, cur_masked_image, patch_adv, prompt,
                                                                 pipe_inpaint, x, y, **kwargs)
            all_grads.append(c_grad)
            losses.append(loss)

        grad = torch.stack(all_grads).mean(0)
        # #####################计算损失###########################
        iterator.set_description_str(f'AVG Loss: {np.mean(losses):.3f}')

        actual_step_size = step_size
        try:
            patch_adv = patch_adv - grad.detach().sign() * actual_step_size
        except:
            print("no grad")
            patch_adv = patch_adv

        # patch_adv = torch.minimum(torch.maximum(patch_adv, patch_adv - eps), patch_adv + eps)
        patch_adv.data = torch.clamp(patch_adv, min=clamp_min, max=clamp_max)
        # --------------------------------------- #
        if loss_best > np.mean(losses):
            loss_best = np.mean(losses)
            patch_best = patch_adv
    torch.cuda.empty_cache()

    return patch_best, last_image


def main(args):
    patch_type = 'rectangle'
    image_size = (3, 512, 512)
    SEED = config.seed
    set_seed(SEED)
    prompt_list = config.prompt_list
    # load models
    # saved_path = args.save_path + str(args.random_seed) + "_n_step" + str(
    #     args.num_inference_steps) + "_eps_" + str(args.epsl2) + "/"
    # pathlib.Path(saved_path).mkdir(parents=True, exist_ok=True)
    # 加载patch的初始值
    if args.pipe:
        patch = patch_initialization()
    else:
        path_patch = "results_period_one/best_patch.pkl"
        with open(path_patch, "rb") as file:
            list1 = pickle.load(file)
            patch = list1[0]["best_patch"]

    patch = patch * 2.0 - 1.0
    patch = patch.to(device)

    # 加载日志文件
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.pipe:
        log_dir = os.path.join(log_dir, 'patch_train_period_pipe.log')
    else:
        log_dir = os.path.join(log_dir, 'patch_train_period2.log')

    logger = get_logger(log_dir)
    logger.info('start training!')
    if args.pipe:
        patch_url = "./results_pipeline"
    else:
        patch_url = "./results"

    if not os.path.exists(patch_url):
        os.makedirs(patch_url)

    train_set = PatchDataset(images_dir=args.input_path)
    len_data = train_set.__len__()
    train_loader = DataLoader(train_set, batch_size=config.batch_size, drop_last=True)
    if args.pipe:
        best_patch_url = patch_url + "/" + "best_patch_pipe.pkl"
    else:
        best_patch_url = patch_url + "/" + "best_patch_period2.pkl"

    best_attack_rate = 0
    best_patch = None
    for epoch in range(config.epoch):
        acc = 0
        acc_1 = 0
        for ind, (img_path, label) in enumerate(train_loader):
            print("label is ::::::::", label[0])
            label_ind = labels[label[0]]
            prompt = prompt_list[label_ind]
            # logger.info(f"attack image id: {ind}, prompt: {prompt}")
            mask_name_png_pro = args.input_path + "/" + "{}_maskprocessed_mask.png".format(label[0])

            init_image = Image.open(img_path[0]).convert('RGB').resize((512, 512))
            mask_image = Image.open(mask_name_png_pro).convert('RGB').resize((512, 512))

            guidance_scale = 7.5
            num_inference_steps = args.num_inference_steps
            num_inference_steps_inference = 100
            cur_mask, cur_masked_image = prepare_mask_and_masked_image(init_image, mask_image)
            cur_mask = cur_mask.half().cuda()
            cur_masked_image = cur_masked_image.half().cuda()
            # patch = patch.half().cuda()

            x_location, y_location = select_location(patch_type, patch, image_size)
            # 更新前的测试，查看结果##########
            applied_patch, mask = mask_generation(patch_type, (patch + 1.0) * 0.5, image_size, x_location,
                                                  y_location)
            applied_patch = applied_patch.to(device)
            mask = mask.to(device)

            adv_image = (torch.mul(mask.type(torch.float32), applied_patch.type(torch.float32)) +
                         torch.mul((1 - mask.type(torch.float32)), totensor(init_image).to(device)))
            adv_image = to_pil(adv_image)
            content = pipe_inpaint(prompt=[prompt] * 4,
                                   image=adv_image,
                                   mask_image=mask_image,
                                   eta=1,
                                   num_inference_steps=num_inference_steps_inference,
                                   guidance_scale=guidance_scale,
                                   )
            # import ipdb; ipdb.set_trace()
            flag = content.nsfw_content_detected

            if sum(flag) > 1:
                if args.inference:
                    # attack to bypass safety_checker
                    if args.l2:
                        print("gose to l2")
                        patch_new, last_image = super_l2(cur_mask, cur_masked_image,
                                                         patch=patch,
                                                         prompt=prompt,
                                                         eps=args.epsl2,
                                                         step_size=config.learning_rate,
                                                         iters=config.max_iter,
                                                         clamp_min=-1,
                                                         clamp_max=1,
                                                         eta=1,
                                                         pipe_inpaint=pipe_inpaint,
                                                         x=x_location,
                                                         y=y_location,
                                                         num_inference_steps=num_inference_steps,
                                                         guidance_scale=guidance_scale,
                                                         grad_reps=config.grad_reps,
                                                         pipeline=args.pipe
                                                         )

                    patch = patch_new

                    applied_patch, mask = mask_generation(patch_type, (patch + 1.0) * 0.5, image_size, x_location,
                                                          y_location)
                    applied_patch = applied_patch.to(device)
                    mask = mask.to(device)

                    adv_image = (torch.mul(mask.type(torch.float32), applied_patch.type(torch.float32)) +
                                 torch.mul((1 - mask.type(torch.float32)), totensor(init_image).to(device)))
                    adv_image = to_pil(adv_image)

                # 开始测试

                content = pipe_inpaint(prompt=[prompt] * 4,
                                       image=adv_image,
                                       mask_image=mask_image,
                                       eta=1,
                                       num_inference_steps=num_inference_steps_inference,
                                       guidance_scale=guidance_scale,
                                       )
                flag = content.nsfw_content_detected
                # torch.save(content, saved_path + str(ind) + "_adv_gen_4.pt")

                print("The flag is::::::", flag)
            else:
                print("The flag is::::::", flag)

            acc_1 += sum(flag)
            if False in flag:
                acc += 1
        if 100 * acc_1 / (len_data * 4) > best_attack_rate:
            best_attack_rate = 100 * acc_1 / (len_data * 4)
            best_patch = patch

        if args.pipe:
            epoch_patch_url = patch_url + "/" + "%d_patch_pipe.pkl" % epoch
        else:
            epoch_patch_url = patch_url + "/" + "%d_patch_period2.pkl" % epoch

        with open(epoch_patch_url, "wb") as file:
            list_best = [{
                "best_patch": patch
            }]
            pickle.dump(list_best, file, True)
        logger.info("attack successful rate: {}%".format(100 * acc / len_data))
        logger.info("attack successful top1 rate: {}%".format(100 * (1 - acc_1 / (len_data * 4))))

    with open(best_patch_url, "wb") as file:
        list_best = [{
            "best_patch": best_patch
        }]
        pickle.dump(list_best, file, True)
    logger.info("The best patch is found  with success rate {}%".format(best_attack_rate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="args for SD attack")
    # parser.add_argument("--iter", type=int, default=10)
    # parser.add_argument("--accumulate", type=int, default=5)
    parser.add_argument("--epsl2", type=float, default=16.0)
    parser.add_argument("--epslinf", type=float, default=16 / 255)
    parser.add_argument("--adjustment", type=float, default=0.07)
    parser.add_argument("--save_path", type=str, default="./save_path/")
    # you can download the (image, mask) pairs from our hugging face page.
    parser.add_argument("--input_path", type=str, default="./data_patch/train")
    parser.add_argument('-i', "--inference", action="store_false")
    parser.add_argument("--l2", action="store_true", default=True)
    parser.add_argument('-s', '--random_seed', type=int, default=3)
    parser.add_argument('-n', "--num_inference_steps", type=int, default=4)  # Troubleshooting Out-of-Memory Issues
    parser.add_argument('--pipe', "-pipeline", type=bool, default=False)

    args = parser.parse_args()
    print(args)
    main(args)
    # python image_editing_attack.py --iter 20 --epsl2 16.0 -s 3 -n 8
