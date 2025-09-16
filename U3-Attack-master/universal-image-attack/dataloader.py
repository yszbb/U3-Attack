import torch
from collections import Counter
import pandas as pd
from sklearn import model_selection
import config
import os


class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, split="train", train_test_split=0.001):
        files = os.listdir(images_dir)
        num_pngs = int(len(files) / 2)
        images, labels = [], []
        for i in range(num_pngs):
            images.append(f"{images_dir}/{i}.png")
            label = i
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
