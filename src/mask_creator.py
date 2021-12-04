import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def group_to_mask_image(group):
    imgs = []
    for idx, line in group.iterrows():
        img = np.zeros(group["width"].max() * group["height"].max(), dtype="uint8")
        encoded_mask_list = [int(val) for val in line["annotation"].split()]
        for i in range(int(len(encoded_mask_list) / 2)):
            img[encoded_mask_list[i * 2] - 1: encoded_mask_list[i * 2] + encoded_mask_list[i * 2 + 1] - 1] = 255
        imgs.append(img.reshape(group["height"].max(), group["width"].max()))
    mask = np.max(imgs, axis=0)
    return mask


def main():
    train_df = pd.read_csv("../sartorius-cell-instance-segmentation/train.csv")
    groups = train_df.groupby("id")
    for idx, group in tqdm(groups):
        mask = group_to_mask_image(group)
        with open(f"../masks/mask_{idx}.pkl", "wb") as f:
            pickle.dump(mask, f)
