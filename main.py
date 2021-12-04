import json

import wandb
from tensorflow import keras

from src.config_keys import ConfigKeys
from src.data_generator import DataLoader
from src.unet import get_model as get_unet_model
from src.wandb_communications import WandbCustomCallback


def main():
    with open("config/unet_config.json") as file:
        cnf = json.load(file)
    wandb.init(project="sartorius", entity="yevpan", config=cnf)
    wdb = WandbCustomCallback(cnf, cnf[ConfigKeys.WEIGHTS_DIR], cnf[ConfigKeys.SAVE_WEIGHTS_EACH])

    dl = DataLoader(cnf)
    dl.load_data()
    data_generator_train, data_generator_validate = dl.split_data()

    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()

    # Build model
    model = get_unet_model((512, 704, 3))
    model.compile(optimizer="adam",
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.MeanIoU(num_classes=2)])

    model.fit(x=data_generator_train,
              epochs=cnf[ConfigKeys.EPOCHS],
              validation_data=data_generator_validate,
              shuffle=cnf[ConfigKeys.SHUFFLE],
              callbacks=[wdb])


if __name__ == '__main__':
    main()
