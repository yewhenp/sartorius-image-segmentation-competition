import json
import os
import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from argparse import ArgumentParser, Namespace
import multiprocessing as mp

from src.constants import ConfigKeys as ck, WIDTH, REDUCED_HEIGHT, HEIGHT
from src.data_processing.data_generator import DataLoader
from src.metrics import competition_metric
from src.models import get_model
from src.visualisation import display
from src.train import train_model
from src.predict import predict_submission
from src.models.unet_watershed import calc_watershed_transform
import numpy as np
from tqdm import trange


def main(args: Namespace):
    np.random.seed(2021)
    tf.random.set_seed(2021)
    with open(args.config) as file:
        cnf = json.load(file)

    if cnf[ck.PREDICT_SUBMISSION]:
        predict_submission(cnf)
        return

    print("Preparing data generators...")
    dl = DataLoader(cnf, cnf[ck.GENERATOR_TYPE])
    dl.load_data()
    data_generator_train, data_generator_validate = dl.split_data()

    if not cnf[ck.USE_PRETRAINED]:
        model = train_model(cnf, (data_generator_train, data_generator_validate))
    else:
        print("Loading weights...")
        model = get_model(cnf, input_shape=(REDUCED_HEIGHT, WIDTH, 3))
        weights_path = os.path.join(cnf[ck.WEIGHTS_DIR], cnf[ck.EXPERIMENT_NAME], "val_best.h5")
        model.load_weights(weights_path)

    if cnf[ck.DISPLAY]:
        for i in range(10):
            y_hat = model.predict(data_generator_validate[i][0])
            rez = y_hat[0]
            example = data_generator_validate[i][1][0]
            if example.shape[-1] > 1:
                example = calc_watershed_transform(example)
            display([data_generator_validate[i][0][0], example, rez])

    if cnf[ck.CALC_METRICS]:
        ys = []
        yhats = []
        for i in trange(len(data_generator_validate)):
            y_hat = model.predict(data_generator_validate[i][0])
            for img_ind in range(y_hat.shape[0]):
                ys.append(data_generator_validate[i][1][img_ind][:, :, 0])
                yhats.append(y_hat[img_ind])

        print(f"Mean competition metric: {competition_metric(ys, yhats)}")


"""
if PREDICT_SUBMISSION - predicts summissions, creates submission.csv and exits
if USE_PRETRAINED - loads model from weights, else trains it
if DISPLAY - vizualizes prediction on random validation image
"""
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default="config/unet_config.json")
    args = parser.parse_args()
    main(args)
