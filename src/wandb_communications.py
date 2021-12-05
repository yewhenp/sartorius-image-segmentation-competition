import os
from typing import Dict

import numpy as np
import wandb
from tensorflow.keras.callbacks import Callback
from .constants import ConfigKeys as ck


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

    def on_epoch_end(self, epoch, logs=None):
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
