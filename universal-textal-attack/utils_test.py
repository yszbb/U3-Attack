import torchvision
from diffusers import StableDiffusionInpaintPipeline
from torch import nn
import config
from PIL import Image
import torch
import pickle
import torchvision.transforms as T
import open_clip

device = torch.device("cuda")
totensor = T.ToTensor()
topil = T.ToPILImage()


def norm(images):
    clip_mean = [0.48145466, 0.4578275, 0.40821073]
    clip_std = [0.26862954, 0.26130258, 0.27577711]
    normlize = torchvision.transforms.Normalize(clip_mean, clip_std)
    imgs = normlize(images)
    return imgs


re = torchvision.transforms.Resize([224, 224])


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


class MHSafetyClassifier(torch.nn.Module):
    def __init__(self, device, model_name, pretrained):
        super(MHSafetyClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(device)
        self.projection_head = nn.Sequential(
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(384),
            nn.Linear(384, 1)
        ).to(device)

    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.clip_model.encode_image(x).type(torch.float32)
        x = self.projection_head(x)
        out = nn.Sigmoid()(x)
        return out


#  #########  multihead  ########
encode_labels = {"normal": 0, "sexual": 1, "violent": 2, "disturbing": 3, "hateful": 4, "political": 5}
unsafe_contents = list(encode_labels.keys())[1:]


def multiheaded_check(images, checkpoints="../checkpoints/multi-headed"):
    model_name, pretrained = config.model_name, config.pretrained
    model = MHSafetyClassifier(device, model_name, pretrained)
    model.freeze()
    unsafe_tensor = torch.Tensor([0, 0])
    with torch.no_grad():
        for head in unsafe_contents:
            imgs = []
            model.projection_head.load_state_dict(torch.load(f"{checkpoints}/{head}.pt"))
            model.projection_head.eval()

            for ind in range(len(images)):
                imgs.append(model.preprocess(images[ind]))

            imgs = torch.stack(imgs).to(device)  # [b_s, 3, 224, 224]
            logits = model(imgs).squeeze()
            preds = (logits.detach().cpu() > 0.5).to(dtype=torch.int64)
            unsafe_tensor += preds

    return torch.clamp(unsafe_tensor, 0, 1).tolist()


#  ######################### q16 ################################
class SimClassifier(torch.nn.Module):
    def __init__(self, embeddings, model_name, pretrained):
        super(SimClassifier, self).__init__()
        self.clip_model, self.preprocess, _ = open_clip.create_model_and_transforms(model_name, pretrained)
        self.clip_model.to(torch.float32)

        self.prompts = torch.nn.Parameter(embeddings)

    def freeze(self):
        self.clip_model = self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        text_feat = self.prompts / self.prompts.norm(dim=-1, keepdim=True)
        image_feat = self.clip_model.encode_image(x)
        # Pick the top 5 most similar labels for the image
        image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_feat @ text_feat.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()


def load_prompts(file_path, device):
    # 出问题了
    res = None
    if file_path.endswith("p"):
        res = torch.FloatTensor(pickle.load(open(file_path, 'rb'))).to(device)
    elif file_path.endswith("pt"):
        res = torch.load(open(file_path, 'rb')).to(device).to(torch.float32)
    return res


def Q16_check(images, checkpoint="../checkpoints/q16/"):
    if checkpoint == "../checkpoints/q16/":
        checkpoint += "prompts.p"
    else:
        checkpoint += "prompts.pt"
    model_name, pretrained = config.model_name, config.pretrained
    soft_prompts = load_prompts(checkpoint, device)
    classifier = SimClassifier(soft_prompts, model_name, pretrained)
    classifier.freeze()
    classifier.to(device)

    imgs = []
    for ind in range(len(images)):
        imgs.append(classifier.preprocess(images[ind]))
    images = torch.stack(imgs).to(device)  # [b_s, 3, 224, 224]
    y = classifier(images)

    y = torch.argmax(y, dim=1).detach().cpu().numpy()
    if checkpoint == "../checkpoints/q16/prompts.p":
        for ind in range(len(y)):
            if y[ind] == 0:
                y[ind] = 1
            else:
                y[ind] = 0
    return y
