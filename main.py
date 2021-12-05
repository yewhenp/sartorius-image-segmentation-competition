import json
import wandb
from tensorflow import keras
from argparse import ArgumentParser, Namespace

from src.constants import ConfigKeys as ck, WIDTH, REDUCED_HEIGHT
from src.data_generator import DataLoader
from src.models import get_model
from src.wandb_communications import WandbCustomCallback


def main(args: Namespace):
    with open(args.config) as file:
        cnf = json.load(file)
    wandb.init(project="sartorius", config=cnf)
    wdb = WandbCustomCallback(cnf, cnf[ck.WEIGHTS_DIR], cnf[ck.SAVE_WEIGHTS_EACH])

    print("Preparing data generators...")
    dl = DataLoader(cnf, cnf[ck.GENERATOR_TYPE])
    dl.load_data()
    data_generator_train, data_generator_validate = dl.split_data()

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    print("Compiling model...")
    model = get_model(name=cnf[ck.MODEL_NAME], input_shape=(REDUCED_HEIGHT, WIDTH, 3))
    model.compile(optimizer="adam",
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.MeanIoU(num_classes=2)])

    print("Training model")
    model.fit(x=data_generator_train,
              epochs=cnf[ck.EPOCHS],
              validation_data=data_generator_validate,
              shuffle=cnf[ck.SHUFFLE],
              callbacks=[wdb])


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=False, default="config/unet_config.json")
    args = parser.parse_args()
    main(args)
