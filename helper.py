import os
from collections import defaultdict
from glob import glob

import cv2
import helper
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from scipy.signal import convolve2d
from sklearn.metrics import accuracy_score, f1_score
from torchvision.datasets import ImageFolder
from tqdm import tqdm


GAUSSIAN_3X3_WEIGHT = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
GAUSSIAN_3X3_WEIGHT = np.divide(GAUSSIAN_3X3_WEIGHT, 16)


def load_image(directory, size=(224, 224)):
    image = cv2.imread(directory)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if not size is None:
        image = cv2.resize(image, size)
    return np.uint16(image)


def save_image(image, save_path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(save_path, image):
        raise ValueError(f"Unable to save to {save_path}!")


def clip(image):
    return np.uint16(np.clip(image, 0, 255))


def convolve(image, weight):
    args = dict(mode="same", boundary="symm")
    size = np.shape(image)
    if len(size) < 3:
        image = convolve2d(image, weight, **args)
        return clip(image)
    for i in range(size[-1]):
        image[..., i] = convolve2d(image[..., i], weight, **args)
    return clip(image)


def gaussian_blur(image, num_convolve):
    image_copy = np.copy(image)

    for _ in range(num_convolve):
        image_copy = convolve(image_copy, GAUSSIAN_3X3_WEIGHT)

    return image_copy


def gaussian_pixel_noise(image, std):
    size = np.shape(image)
    noise = np.random.normal(scale=std, size=size)
    return clip(image + noise)


def scale_contrast(image, scale):
    return clip(image * scale)


def change_brightness(image, value):
    return clip(image + value)


def occlusion(image, edge_length):
    image_copy = np.copy(image)
    if edge_length > 0:
        size = np.shape(image)

        h_start = np.random.randint(size[0] - edge_length)
        h_end = h_start + edge_length

        w_start = np.random.randint(size[1] - edge_length)
        w_end = w_start + edge_length

        mask = np.zeros([edge_length] * 2).astype(np.int16)
        if len(size) > 2:
            mask = np.expand_dims(mask, -1)

        image_copy[h_start:h_end, w_start:w_end] = mask

    return clip(image_copy)


def salt_and_pepper(image, rate, salt_ratio=0.5):
    size = np.shape(image)
    mask = np.random.random(size)
    pepper = mask < rate
    salt = mask < rate * salt_ratio

    image_copy = np.copy(image)

    image_copy[pepper] = 0
    image_copy[salt] = 255

    return clip(image_copy)


def f1_macro(outputs, targets):
    outputs = outputs.argmax(-1).cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, outputs, average="macro")


def accuracy(outputs, targets):
    outputs = outputs.argmax(-1).cpu().numpy()
    targets = targets.cpu().numpy()
    return accuracy_score(targets, outputs)


# Dataloader hyperparameter
def construct_dataset(batch_size):
    transforms = [
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transforms_train = T.Compose([T.RandAugment(), *transforms])
    transforms_eval = T.Compose(transforms)

    dataset = {
        "train": ImageFolder("dataset/train", transform=transforms_train),
        "valid": ImageFolder("dataset/valid", transform=transforms_eval),
        "test": ImageFolder("dataset/test", transform=transforms_eval),
    }

    dataloader = {
        "train": torch.utils.data.DataLoader(
            dataset["train"], batch_size, shuffle=True, pin_memory=True
        ),
        "valid": torch.utils.data.DataLoader(
            dataset["valid"], batch_size, pin_memory=True
        ),
        "test": torch.utils.data.DataLoader(
            dataset["test"], batch_size, pin_memory=True
        ),
    }

    return dataset, dataloader


def pretrained(backbone):
    backbone = timm.create_model(backbone, pretrained=True)
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


def generate_classifier(backbone, num_classes, p_dropout=0):
    dropout = torch.nn.Identity()
    if p_dropout > 0:
        dropout = torch.nn.Dropout(p_dropout)
    if isinstance(backbone, timm.models.resnet.ResNet):
        in_features = backbone.fc.in_features
        backbone.fc = torch.nn.Sequential(
            dropout, torch.nn.Linear(in_features, num_classes)
        )
    if isinstance(backbone, timm.models.ConvNeXt):
        in_features = backbone.head.fc.in_features
        backbone.head.drop = dropout
        backbone.head.fc = torch.nn.Linear(in_features, num_classes)
    if isinstance(backbone, timm.models.maxxvit.MaxxVit):
        in_features = backbone.head.fc.in_features
        backbone.head.fc = torch.nn.Sequential(
            dropout, torch.nn.Linear(in_features, num_classes)
        )
    if isinstance(backbone, timm.models.mlp_mixer.MlpMixer):
        in_features = backbone.head.in_features
        backbone.head = torch.nn.Sequential(
            dropout, torch.nn.Linear(in_features, num_classes)
        )
    if isinstance(backbone, timm.models.densenet.DenseNet):
        in_features = backbone.classifier.in_features
        backbone.classifier = torch.nn.Sequential(
            dropout, torch.nn.Linear(in_features, num_classes)
        )
    return backbone


def train_one_epoch(
    dataloader,
    model,
    criterion,
    optimizer,
    scheduler=None,
    scaler=None,
    ema=None,
    subset="train",
):
    record = defaultdict(float)
    record["Subset"] = subset.title()
    with torch.autocast("cuda"):
        model.train()
        for inputs, targets in dataloader["train"]:
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()

            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if not scheduler is None:
                scheduler.step()

            if not ema is None:
                ema.update(model)

            record["Loss"] += loss.item()
            record["Accuracy"] += accuracy(outputs, targets)
            record["F1-Macro"] += f1_macro(outputs, targets)

        record["Loss"] /= len(dataloader["train"])
        record["Accuracy"] /= len(dataloader["train"])
        record["F1-Macro"] /= len(dataloader["train"])

    return dict(record)


def evaluate(dataloader, model, criterion, subset):
    record = defaultdict(float)
    record["Subset"] = subset.title()
    with torch.autocast("cuda"), torch.inference_mode():
        model.eval()
        for inputs, targets in dataloader[subset]:
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            record["Loss"] += loss.item()
            record["Accuracy"] += accuracy(outputs, targets)
            record["F1-Macro"] += f1_macro(outputs, targets)

        record["Loss"] /= len(dataloader[subset])
        record["Accuracy"] /= len(dataloader[subset])
        record["F1-Macro"] /= len(dataloader[subset])

    return dict(record)


def train(backbone):
    torch.cuda.empty_cache()
    torch.manual_seed(2022)
    epochs = 200
    dataset, dataloader = construct_dataset(512)
    num_classes = len(dataset["train"].classes)
    model = generate_classifier(pretrained(backbone), num_classes).cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.001 * dataloader["train"].batch_size / 256,
        momentum=0.9,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 50 * len(dataloader["train"]), 1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    history = []
    with tqdm(total=epochs) as pbar:
        for _ in range(pbar.total):
            history.append(
                train_one_epoch(
                    dataloader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    scaler,
                )
            )

            history.append(evaluate(dataloader, model, criterion, "valid"))

            pbar.set_postfix(history[-1])
            pbar.update()
    torch.save(model.state_dict(), f"weights/{backbone}.pt")
    del model
    return pd.DataFrame(history)
