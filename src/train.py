import wandb
import os
import numpy as np

from typing import Dict
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model

from .data_processing.data_generator import DataLoader
from .constants import ConfigKeys as ck, REDUCED_HEIGHT, WIDTH
from .models import get_model
from .loss import get_loss
from .metrics import get_metrics


class WandbCustomCallback(Callback):
    def __init__(self, config: Dict, all_weights_dir: str, save_each: int):
        super().__init__()
        self.config = config
        self.weights_dir = os.path.join(all_weights_dir, config[ck.EXPERIMENT_NAME])
        self.save_each = save_each
        self.best_train_loss = np.inf
        self.best_val_loss = np.inf

        if not os.path.exists(all_weights_dir):
            os.mkdir(all_weights_dir)
        if not os.path.exists(self.weights_dir):
            os.mkdir(self.weights_dir)

    def on_epoch_end(self, epoch, logs=None) -> Model:
        if logs is not None:
            wandb.log({metric: logs[metric] for metric in logs})
        if epoch % self.save_each == 0:
            self.model.save_weights(f"./{self.weights_dir}/w{epoch}.h5")
        if logs["loss"] < self.best_train_loss:
            self.best_train_loss = logs["loss"]
            self.model.save_weights(f"./{self.weights_dir}/train_best.h5")
        if logs["val_loss"] < self.best_val_loss:
            self.best_val_loss = logs["val_loss"]
            self.model.save_weights(f"./{self.weights_dir}/val_best.h5")


def train_model(cnf: Dict, data_generators=None):
    if data_generators is None:
        print("Preparing data generators...")
        dl = DataLoader(cnf, cnf[ck.GENERATOR_TYPE])
        dl.load_data()
        data_generator_train, data_generator_validate = dl.split_data()
    else:
        data_generator_train, data_generator_validate = data_generators

    print("Compiling model...")
    # Free up RAM in case the model definition cells were run multiple times
    keras.backend.clear_session()
    model = get_model(cnf, input_shape=(REDUCED_HEIGHT, WIDTH, 3))
    loss_function = get_loss(cnf)
    metrics = get_metrics(cnf)
    model.compile(optimizer=cnf[ck.OPTIMIZER],
                  loss=loss_function,
                  metrics=[metrics])

    wandb.init(project="sartorius", entity="happy_geese", config=cnf)
    wdb = WandbCustomCallback(cnf, cnf[ck.WEIGHTS_DIR], cnf[ck.SAVE_WEIGHTS_EACH])
    # Build model

    print("Training model")
    model.fit(x=data_generator_train,
              epochs=cnf[ck.EPOCHS],
              validation_data=data_generator_validate,
              shuffle=cnf[ck.SHUFFLE],
              callbacks=[wdb])

    return model
