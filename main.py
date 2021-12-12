import json
import os
import wandb
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from argparse import ArgumentParser, Namespace

from src.constants import ConfigKeys as ck, WIDTH, REDUCED_HEIGHT
from src.data_processing.data_generator import DataLoader, SubmissionDataLoader
from src.models import get_model
from src.loss import get_loss
from src.metrics import get_metrics
from src.visualisation import display
from src.data_processing.post_processing import mask2rle
from src.train import WandbCustomCallback, train_model


def main(args: Namespace):
    with open(args.config) as file:
        cnf = json.load(file)

    if not cnf[ck.USE_PRETRAINED] or cnf[ck.DISPLAY]:
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
        y_hat = model.predict(data_generator_validate[0][0])
        display([data_generator_validate[0][0][0], data_generator_validate[0][1][0], y_hat[0]])

    if cnf[ck.PREDICT_SUBMISSION]:
        print("Submision photos predicting...")
        submission_dl = SubmissionDataLoader(cnf)
        data_generator_submission = submission_dl.load_data()
        submission = pd.DataFrame([], columns=['id', 'predicted'])
        for i in range(len(data_generator_submission)):
            y_hat = model.predict(data_generator_submission[i][0])
            image_id = submission_dl.df.iloc[i]['id']
            mask_df = mask2rle(image_id, y_hat[0])
            submission = pd.concat([submission, mask_df])
            display([data_generator_submission[i][0][0], data_generator_submission[i][1][0], y_hat[0]])

        submission.to_csv("submission.csv", index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default="config/unet_config.json")
    args = parser.parse_args()
    main(args)
