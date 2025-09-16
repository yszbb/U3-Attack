import torch
from PIL import Image
import numpy as np
import random
import config
import logging
import os
import torchvision.transforms as T

totensor = T.ToTensor()
topil = T.ToPILImage()


def image_grid(imgs, rows, cols, image_scale):
    assert len(imgs) == rows * cols

    # w, h = imgs[0].size
    w, h = (image_scale, image_scale)
    grid = Image.new('RGB', size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def patch_initialization(patch_type='rectangle', image_size=(3, 512, 512), noise_percentage=0.06):
    if patch_type == 'rectangle':
        patch_length = int((noise_percentage * image_size[1] * image_size[2]) ** 0.5)
        patch = torch.rand(image_size[0], patch_length, patch_length)
        # patch = (patch * 255).type(torch.uint8)
        return patch


def mask_generation(mask_type='rectangle', patch=None, image_size=(3, 512, 512), x_location=0, y_location=0):
    applied_patch = torch.zeros(image_size)

    mask = torch.zeros(image_size)
    mask_value = torch.ones(patch.shape)

    if mask_type == 'rectangle':
        applied_patch[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = patch

        mask[:, x_location:x_location + patch.shape[1], y_location:y_location + patch.shape[2]] = mask_value

    return applied_patch, mask


def select_location(mask_type='rectangle', patch=None, image_size=(3, 512, 512), location_global=0, target=0):
    if mask_type == 'rectangle':
        if location_global != 0:
            # patch location  随机生成一个位置
            x_location = np.random.randint(low=0, high=image_size[2] - patch.shape[2])
            y_location = np.random.randint(low=0, high=image_size[3] - patch.shape[3])
        else:
            x, y = config.receptive_field[target]
            x_location = np.random.randint(low=x[0], high=x[1])
            y_location = np.random.randint(low=y[0], high=y[1])
    return x_location, y_location


def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)


def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    # 将像素值处理到[-1, 1]之间
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image


def prepare_image(image):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    return image[0]


def save_batch_img(batch_img, output_dir, prefix=None):
    """
    img: tensor, [b, c, h, w] , [0, 1]
    """
    for i in range(batch_img.shape[0]):
        img = batch_img[i]
        img = img.permute(1, 2, 0)  # shape HWC
        img = img.detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)

        # # 保存图像
        os.makedirs(output_dir, exist_ok=True)
        if prefix is not None:
            img.save('{}/{}_{}.jpg'.format(output_dir, prefix, i))
        else:
            img.save('{}/{}.jpg'.format(output_dir, i))
