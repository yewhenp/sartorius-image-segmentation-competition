import pickle
import json
import os
import numpy as np

from tqdm.auto import tqdm
from argparse import ArgumentParser
from typing import Dict

# START PROJECT IMPORTS
from utilities import load_train_labels
from ..constants import ConfigKeys as ck, HEIGHT, WIDTH
# END PROJECT_IMPORTS


def group_to_mask_image(group):
    imgs = []
    for idx, line in group.iterrows():
        img = np.zeros(WIDTH * HEIGHT, dtype="uint8")
        encoded_mask_list = [int(val) for val in line["annotation"].split()]
        for i in range(int(len(encoded_mask_list) / 2)):
            img[encoded_mask_list[i * 2] - 1: encoded_mask_list[i * 2] + encoded_mask_list[i * 2 + 1] - 1] = 255
        imgs.append(img.reshape(HEIGHT, WIDTH))
    mask = np.max(imgs, axis=0)
    return mask


def main(cnf: Dict):
    groups = load_train_labels(cnf[ck.TRAIN_CSV_PATH])

    # make sure masks directory exists
    masks_dir = cnf[ck.MASK_DIR_PATH]
    if not os.path.exists(masks_dir):
        os.mkdir(masks_dir)

    for idx, group in tqdm(groups):
        mask = group_to_mask_image(group)
        with open(f"masks/mask_{idx}.pkl", "wb") as f:
            pickle.dump(mask, f)


if __name__ == '__main__':
    # parse config
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default="config/unet_config.json")
    cnf_filename = parser.parse_args().config
    cnf = json.load(open(cnf_filename, 'r'))

    main(cnf)
