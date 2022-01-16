import cv2
# from cv2 import cv2
import numpy as np
import pandas as pd

# START PROJECT IMPORTS
# from ..constants import HEIGHT, WIDTH
WIDTH = 704
HEIGHT = 520
# END PROJECT_IMPORTS


def split_mask(probability, threshold=0.5, min_size=20):
    probability = np.squeeze(probability)
    if probability.max() > 1:
        num_component = probability.max()
        component = probability.astype(np.int)
    else:
        probability = probability.astype("float32")
        mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
        num_component, component = cv2.connectedComponents(mask.astype(np.uint8))

    for c in range(1, int(num_component)):
        p = (component == c)
        if p.sum() < min_size:
            component[p] = 0
    return component


def submask2rle(mask):
    mask = np.array(mask)
    pixels = mask.flatten()
    pad = np.array([0])
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def mask2rle(id, mask):
    # mask = cv2.resize(mask.astype("float32"), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype("int")
    mask = split_mask(mask)
    submasks = []
    for c in range(1, int(mask.max())):
        p = (mask == c)
        if p.sum() > 20:
            a_prediction = np.zeros(mask.shape, np.int)
            a_prediction[p] = 1
            submasks.append(a_prediction)
    ids = []
    pred = []
    for submask in submasks:
        ids.append(id)
        pred.append(submask2rle(submask))
    return pd.DataFrame(list(zip(ids, pred)), columns=['id', 'predicted'])
