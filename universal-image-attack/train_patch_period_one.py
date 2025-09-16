import os
import pickle
import torchvision
from colorama import init, Fore, Back, Style
import torch
from torch import nn
import pandas as pd
from sklearn import model_selection
from collections import Counter
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import config
from PIL import Image
from utils import patch_initialization, mask_generation, image_grid, select_location, set_seed, get_logger
import torchvision.transforms as T

safety_checker = torch.load("weights/safety_checker.pt")
concept_embeds = torch.load("weights/concept_embeds.pt")
concept_embeds_weights = torch.load("weights/concept_embeds_weights.pt")  # ! 概念编码阈值 threshold
device = torch.device("cuda")
re = torchvision.transforms.Resize([224, 224])

totensor = T.ToTensor()
topil = T.ToPILImage()

path_patch = "./data/attack.jpg"
patch = Image.open(path_patch).convert('RGB').resize((125, 125))
source_patch = totensor(patch)


def compare_loss(patch_1, patch_2):
    return (((patch_1 - patch_2) ** 2 + 1e-9) ** 0.5).mean()


def tv_loss(img):
    h, w = img.shape[-2], img.shape[-1]
    img_a = img[..., : h - 1, : w - 1]
    img_b = img[..., 1:, : w - 1]
    img_c = img[..., : h - 1, 1:]
    tv = ((img_a - img_b) ** 2 + (img_a - img_c) ** 2 + 1e-9) ** 0.5
    return tv.mean()


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


def norm(images):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    normlize = torchvision.transforms.Normalize(clip_mean, clip_std)
    imgs = normlize(images)
    return imgs


encode_labels = {"normal": 0, "sexual": 1, "violent": 2, "disturbing": 3, "hateful": 4, "political": 5}
unsafe_contents = list(encode_labels.keys())[1:]


# 划分数据集  训练集和测试集
class BinaryAnnotatedDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir, split="train", head=None, train_test_split=0.4):

        labels_df = pd.read_excel(labels_dir)
        images, labels = [], []
        for i in labels_df.index:
            image_index = labels_df.loc[i, "image_index"]
            images.append(f"{images_dir}{image_index}")
            label = labels_df.loc[i, "final_label"]

            raw_labels = []
            for annotator in ["rater_0", "rater_1", "rater_2"]:
                _label = labels_df.loc[i, annotator]
                _label = [int(l) for l in str(_label).split(",")]
                raw_labels.extend(_label)
            label_collection = Counter(raw_labels).most_common()
            label_collection_dict = {}
            for l, n in label_collection:
                label_collection_dict[l] = n
            if head:
                target_label = encode_labels[head]
                try:
                    if int(label_collection_dict[target_label]) >= 2:
                        label = 1
                except:
                    label = 0

            labels.append(label)

        images_train, images_test, labels_train, labels_test = model_selection.train_test_split(images, labels,
                                                                                                test_size=train_test_split,
                                                                                                shuffle=True,
                                                                                                random_state=1)
        if split == "train":
            self.images = images_train
            self.labels = labels_train
        elif split == "test":
            self.images = images_test
            self.labels = labels_test

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)

    def weights(self):
        count = Counter(self.labels)
        class_count = np.array([count[0], count[1]])
        weight = 1.0 / class_count
        weights = np.array([weight[0] if la == 0 else weight[1] for la in self.labels])
        return weights


def safety_checker_classifier(clip_input, checker):
    pooled_output = checker.vision_model(clip_input)[1]  # pooled_output
    image_embeds = checker.visual_projection(pooled_output)

    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    cos_dist = cosine_distance(image_embeds, concept_embeds)  # 1 * 17
    result = []
    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}

        adjustment = 0.07

        for concept_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concept_idx]
            concept_threshold = concept_embeds_weights[concept_idx].item()
            result_img["concept_scores"][concept_idx] = torch.round(concept_cos - concept_threshold + adjustment,
                                                                    decimals=3)
            if result_img["concept_scores"][concept_idx] > 0:
                result_img["bad_concepts"].append(concept_idx)

        result.append(result_img)

    has_nsfw_concepts = [len(res["bad_concepts"]) > 0 for res in result]

    return has_nsfw_concepts


# 损失函数计算
class Safety_Checker(torch.nn.Module):
    def __init__(self):
        super(Safety_Checker, self).__init__()
        self.safety_checker = safety_checker

    def forward(self, clip_input, adjustment=0.07):
        pooled_output = self.safety_checker.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.safety_checker.visual_projection(pooled_output)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        cos_dist = cosine_distance(image_embeds, concept_embeds)  # 1 * 17

        batch_size = image_embeds.shape[0]
        for i in range(batch_size):
            result_img = {"special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []}
            # increase this value to create a stronger nfsw filter
            # at the cost of increasing the possibility of filtering begign images_source
            adjustment = adjustment
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
                return loss_total


def compute_grad(patch, images, sefety_checker, patch_type, image_size, x_location, y_location):
    torch.set_grad_enabled(True)
    patch = patch.clone()
    images = images.clone()
    patch.requires_grad_()
    images.requires_grad = False
    # ############### EOT ##############
    rand_noise = (torch.rand((patch.shape[0], patch.shape[1], patch.shape[2])) * 2 - 1) / 10
    patch_noise = patch + rand_noise

    applied_patch, mask = mask_generation(patch_type, patch_noise, image_size, x_location, y_location)
    applied_patch = applied_patch.to(device)
    mask = mask.to(device)
    perturbated_image = torch.mul(mask.type(torch.float32), applied_patch.type(torch.float32)) + torch.mul(
        (1 - mask.type(torch.float32)), images)

    perturbated_image = re(perturbated_image)
    image_nat_normlize = norm(perturbated_image)
    # 攻击损失
    loss_1 = sefety_checker(image_nat_normlize.type(torch.float16))
    # 增广损失
    loss_2 = tv_loss(patch)

    # loss_3 = compare_loss(patch, source_patch)
    # lambda_1 = 0.05
    # lambda_2 = 0.8
    # loss_total = (1 - lambda_1 - lambda_2) * loss_1 + lambda_1 * loss_2 + lambda_2 * loss_3 * 30

    lambda_1 = 0.03
    loss_total = lambda_1 * loss_1 + (1 - lambda_1) * loss_2
    grad = torch.autograd.grad(loss_total, [patch])[0]
    return grad, loss_total.item()


def train(opt, record=True):
    EPOCH = config.epoch
    MAX_ITER = config.max_iter
    grad_reps = config.grad_reps
    LR = config.learning_rate
    BATCH_SIZE = config.batch_size
    output_dir = opt.output_dir
    patch_type = 'rectangle'
    image_size = (3, 512, 512)
    noise_percentage = config.noise_percentage
    SEED = config.seed
    set_seed(SEED)

    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, 'patch_train.log')
    logger = get_logger(log_dir)
    logger.info('start testing!')

    patch_url = "./results"
    if not os.path.exists(patch_url):
        os.makedirs(patch_url)

    best_patch_url = patch_url + "/" + "best_patch.pkl"

    # 构建训练集 测试集
    trainset = BinaryAnnotatedDataset(images_dir=opt.images_dir, labels_dir=opt.labels_dir, split="train",
                                      head=unsafe_contents[0])

    # sampler = WeightedRandomSampler(trainset.weights(), num_samples=trainset.weights().shape[0], replacement=True)

    testset = BinaryAnnotatedDataset(images_dir=opt.images_dir, labels_dir=opt.labels_dir, split="test",
                                     head=unsafe_contents[0])

    print("训练集长度:::::::::", trainset.__len__())  # 951
    print("测试集长度:::::::::", testset.__len__())  # 634

    # train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=True, sampler=sampler)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, drop_last=False)

    # patch 初始化
    patch = patch_initialization(patch_type, image_size, noise_percentage)
    # 损失函数计算
    safe_check = Safety_Checker()
    best_attack_rate = 0
    best_patch = None
    for epoch in range(EPOCH):
        train_total, train_actual_total, train_success = 0, 0, 0
        test_total, test_actual_total, test_success = 0, 0, 0
        for idx, (imgs, labels) in enumerate(train_loader):
            labels = labels.to(device)
            labels = labels.type(torch.float32)
            train_total += labels.shape[0]
            images = [np.array(Image.open(img_path).convert('RGB').resize((512, 512))) for img_path in imgs]
            images = np.array(images).transpose(0, 3, 1, 2)
            images = torch.from_numpy(images).to(device)
            images = images.to(dtype=torch.float32) / 255.0
            assert images.shape[0] == 1, 'Only one picture should be loaded each time.'
            # 进行检查
            images_re = re(images)
            images_re = norm(images_re)
            has_nsfw_contents = safety_checker_classifier(images_re.type(torch.float16), safe_check.safety_checker)
            if any(has_nsfw_contents):
                train_actual_total += 1
                # ##############    添加噪音
                rand_noise = (torch.rand((patch.shape[0], patch.shape[1], patch.shape[2])) * 2 - 1) / 10
                patch_noise = patch + rand_noise
                x_location, y_location = select_location(patch_type, patch_noise, image_size)
                # 事先测试一下patch是否有用 ++++
                applied_patch, mask = mask_generation(patch_type, patch_noise, image_size, x_location, y_location)
                applied_patch = applied_patch.to(device)
                mask = mask.to(device)
                perturbated_image = (torch.mul(mask.type(torch.float32), applied_patch.type(torch.float32)) +
                                     torch.mul((1 - mask.type(torch.float32)), images))

                perturbated_image = re(perturbated_image)
                perturbated_image = norm(perturbated_image)

                has_nsfw_contents = safety_checker_classifier(perturbated_image.type(torch.float16),
                                                              safe_check.safety_checker)
                if any(has_nsfw_contents):
                    for idx in range(MAX_ITER):
                        all_grads = []
                        losses = []
                        # 多次求解梯度
                        for i in range(grad_reps):
                            grad, loss = compute_grad(patch, images, safe_check, patch_type, image_size, x_location,
                                                      y_location)
                            all_grads.append(grad)
                            losses.append(loss)

                        grad = torch.stack(all_grads).mean(0)
                        print(f'AVG Loss: {np.mean(losses):.3f}')
                        # actual_step_size = step_size - (step_size - step_size / 100) / iters * i
                        actual_step_size = LR

                        try:
                            patch = patch - grad.detach().sign() * actual_step_size
                        except:
                            print("no grad")
                            patch = patch

                        # 第一种求解方法
                        patch = torch.clamp(patch, min=0., max=1.)

                    # 测试是否更新成功
                    rand_noise = (torch.rand((patch.shape[0], patch.shape[1], patch.shape[2])) * 2 - 1) / 10
                    patch_noise = patch + rand_noise
                    applied_patch, mask = mask_generation(patch_type, patch_noise, image_size, x_location, y_location)
                    applied_patch = applied_patch.to(device)
                    mask = mask.to(device)
                    perturbated_image = (torch.mul(mask.type(torch.float32), applied_patch.type(torch.float32)) +
                                         torch.mul((1 - mask.type(torch.float32)), images))

                    perturbated_image = re(perturbated_image)
                    perturbated_image = norm(perturbated_image)
                    has_nsfw_contents = safety_checker_classifier(perturbated_image.type(torch.float16),
                                                                  safe_check.safety_checker)
                    if not (any(has_nsfw_contents)):
                        train_success += 1
                else:
                    train_success += 1
                print(f"累计攻击{train_success}个")

        for idx, (imgs, labels) in enumerate(test_loader):
            test_total += labels.shape[0]
            images = [np.array(Image.open(img_path).convert('RGB').resize((512, 512))) for img_path in imgs]
            images = np.array(images).transpose(0, 3, 1, 2)
            images = torch.from_numpy(images).to(device)  # [b_s, 3, 224, 224]
            images = images.to(dtype=torch.float32) / 255.0
            assert images.shape[0] == 1, 'Only one picture should be loaded each time.'
            images_re = re(images)
            images_re = norm(images_re)

            has_nsfw_contents = safety_checker_classifier(images_re.type(torch.float16), safe_check.safety_checker)
            if any(has_nsfw_contents):
                test_actual_total += 1
                rand_noise = (torch.rand((patch.shape[0], patch.shape[1], patch.shape[2])) * 2 - 1) / 10
                patch_noise = patch + rand_noise

                x_location, y_location = select_location(patch_type, patch_noise, image_size)
                # 事先测试一下patch是否有用 ++++
                applied_patch, mask = mask_generation(patch_type, patch_noise, image_size, x_location, y_location)
                applied_patch = applied_patch.to(device)
                mask = mask.to(device)
                perturbated_image = (torch.mul(mask.type(torch.float32), applied_patch.type(torch.float32)) +
                                     torch.mul((1 - mask.type(torch.float32)), images))

                perturbated_image = re(perturbated_image)
                perturbated_image = norm(perturbated_image)
                has_nsfw_contents = safety_checker_classifier(perturbated_image.type(torch.float16),
                                                              safe_check.safety_checker)
                if not (any(has_nsfw_contents)):
                    test_success += 1

        ASR = 100 * test_success / test_actual_total
        logger.info(
            "Epoch:{} Patch attack success rate on testset: {:.3f}%".format(epoch, ASR))

        if ASR > best_attack_rate:
            best_attack_rate = ASR
            best_patch = patch

        epoch_patch_url = patch_url + "/" + "%d_patch.pkl" % epoch
        with open(epoch_patch_url, "wb") as file:
            list_best = [{
                "best_patch": best_patch
            }]
            pickle.dump(list_best, file, True)

    with open(best_patch_url, "wb") as file:
        list_best = [{
            "best_patch": best_patch
        }]
        pickle.dump(list_best, file, True)
    logger.info("The best patch is found  with success rate {}% on testset".format(best_attack_rate))


# +++++++++++++++++++++++++++ train ++++++++++++++++++++++++++++++++++++
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--images_dir",
        type=str,
        nargs="?",
        default="./data/images_filter/",
        help="adv_1_patch folder"
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        nargs="?",
        default="./data/labels_filter.xlsx",
        help="the directory saved prompts"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./results_period_one/"
    )

    opt = parser.parse_args()

    train(opt, record=True)
