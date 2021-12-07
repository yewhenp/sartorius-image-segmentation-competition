import json
import wandb
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from argparse import ArgumentParser, Namespace

from src.constants import ConfigKeys as ck, WIDTH, REDUCED_HEIGHT
from src.data_generator import DataLoader, SubmissionDataLoader
from src.models import get_model
from src.loss import comb_loss, tversky_loss
from src.visualisation import display
from src.post_processing import mask2rle
from src.wandb_communications import WandbCustomCallback


def main(args: Namespace):
    with open(args.config) as file:
        cnf = json.load(file)

    if not cnf[ck.USE_PRETRAINED] or cnf[ck.DISPLAY]:
        print("Preparing data generators...")
        dl = DataLoader(cnf, cnf[ck.GENERATOR_TYPE])
        dl.load_data()
        data_generator_train, data_generator_validate = dl.split_data()

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    model = get_model(name=cnf[ck.MODEL_NAME], input_shape=(REDUCED_HEIGHT, WIDTH, 3))
    model.compile(optimizer="adam",
                  loss=tversky_loss(0.5),
                  metrics=[tf.keras.metrics.MeanIoU(2)])
    print("Compiling model...")

    if not cnf[ck.USE_PRETRAINED]:
        wandb.init(project="sartorius", config=cnf)
        wdb = WandbCustomCallback(cnf, cnf[ck.WEIGHTS_DIR], cnf[ck.SAVE_WEIGHTS_EACH])
        # Build model

        print("Training model")
        model.fit(x=data_generator_train,
                  epochs=cnf[ck.EPOCHS],
                  validation_data=data_generator_validate,
                  shuffle=cnf[ck.SHUFFLE],
                  callbacks=[wdb])

    else:
        print("Loading weights...")
        model.load_weights(cnf[ck.WEIGHTS_DIR] + "/val_best.h5")

    if cnf[ck.DISPLAY]:
        y_hat = model.predict(data_generator_validate[0][0])
        display([data_generator_validate[0][0][0], data_generator_validate[0][1][0], y_hat[0]])

    if cnf[ck.PREDICT_SUBMISSION]:
        print("Train photos predicting...")
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
