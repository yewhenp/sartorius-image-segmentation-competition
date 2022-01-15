import json
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
from argparse import ArgumentParser, Namespace

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
        i = 0
        y_hat = model.predict(data_generator_validate[i][0])
        rez = y_hat[0]
        example = data_generator_validate[i][1][0]
        if example.shape[-1] > 1:
            example = calc_watershed_transform(example)
        display([data_generator_validate[i][0][0], example, rez])
    
    if cnf[ck.CALC_METRICS]:
        mean_comp_metric = 0
        for i in trange(len(data_generator_validate)):
            # i = 0
            y = data_generator_validate[i][1][0][:, :, 0]
            y_hat = model.predict(data_generator_validate[i][0])
            print("\n predicted")
            y_hat = np.squeeze(y_hat)
            met = competition_metric(y.astype(float), y_hat.astype(float))
            print(met)
            mean_comp_metric += met
        print(f"Mean competition metric: {mean_comp_metric / len(data_generator_validate)}")
        # for i, metric_name in enumerate(cnf[ck.METRICS]):
        #     print(f"{metric_name}: {metrics[i](y, y_hat)}")


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
