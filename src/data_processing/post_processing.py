import cv2
import numpy as np
import pandas as pd

from ..constants import HEIGHT, WIDTH


def split_mask(probability, threshold=0.5, min_size=300):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = []
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            a_prediction = np.zeros((520, 704), np.float32)
            a_prediction[p] = 1
            predictions.append(a_prediction)
    return predictions


def submask2rle(mask):
    mask = np.array(mask)
    pixels = mask.flatten()
    pad = np.array([0])
    pixels = np.concatenate([pad, pixels, pad])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def mask2rle(id, mask):
    main_mask = cv2.resize(mask.astype("uint8"), (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
    submasks = split_mask(main_mask)
    ids = []
    pred = []
    for submask in submasks:
        ids.append(id)
        pred.append(submask2rle(submask))
    return pd.DataFrame(list(zip(ids, pred)), columns=['id', 'predicted'])
