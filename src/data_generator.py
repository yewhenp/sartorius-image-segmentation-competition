import pickle
from typing import Union, Dict

import cv2
import numpy as np
import pandas as pd
from tensorflow import keras
# from tensorflow.keras import layers
# for some reasaon there is no Rescaling in above in my tensorflow version # TODO: fix tensorflow
from tensorflow.python.keras import layers
from .utilities import load_train_labels
from .constants import WIDTH, HEIGHT, REDUCED_HEIGHT, ConfigKeys as ck


class NeuronDataGenerator(keras.utils.Sequence):
    """
    Generates data for Keras
    Sequence based data generator.
    """

    def __init__(self, cnf: Dict, grouped_labels: pd.core.groupby.GroupBy):
        """
        Initialization
        :param cnf: config
        :param grouped_labels: grouped dataframe by id
        """
        self.cnf = cnf
        self.df = grouped_labels
        self.photo_keys = list(grouped_labels.groups.keys())
        self.batch_size = cnf[ck.BATCH_SIZE]
        self.shuffle = cnf[ck.SHUFFLE]
        self.image_dim = (REDUCED_HEIGHT, WIDTH)     # TODO: try not reshape
        self.channels = 3
        self.indexes = []
        self.rescaling = layers.Rescaling(1. / 255) # TODO: -mean / std
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the batch
        :return: x and y
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        photo_ids = [self.photo_keys[ind] for ind in indexes]
        x = self.get_x(photo_ids)
        y = self.get_y(photo_ids)
        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def get_x(self, photo_ids):
        """
        Generates data containing batch_size images
        :param photo_ids: list of ids each representing one image
        :return: batch of images
        """
        x = np.empty((self.batch_size, *self.image_dim, self.channels), dtype="float")
        for i, photo_id in enumerate(photo_ids):
            image = cv2.imread(f"{self.cnf[ck.IMAGES_DIR_PATH]}/{photo_id}.png")
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image.shape[1], REDUCED_HEIGHT))
            image = self.rescaling(image)
            assert 0 <= image.numpy().min() < image.numpy().max() <= 1
            x[i, ] = image
        return x

    def get_y(self, photo_ids):
        """
        Generates data containing batch_size masks
        :param photo_ids: list of ids each representing one image
        :return: batch of masks
        """
        y = np.empty((self.batch_size, *self.image_dim), dtype="int")
        for i, photo_id in enumerate(photo_ids):
            with open(f"{self.cnf[ck.MASK_DIR_PATH]}/mask_{photo_id}.pkl", 'rb') as f:
                image = pickle.load(f)
                image = cv2.resize(image, (image.shape[1], REDUCED_HEIGHT))
                image = np.round(image / 255)
                assert image.min() == 0 and image.max() == 1
                y[i] = image
        return y


class DataLoader:
    """
    Loads data
    """

    def __init__(self, cnf: Dict):
        """
        Initialisation
        :param cnf: config
        """
        self.cnf = cnf
        self.train_df = None

    def load_data(self):
        """
        Loads data from file
        :return: None
        """
        # self.train_df = pd.read_csv(self.cnf[ConfigKeys.TRAIN_CSV_PATH])
        self.train_df = load_train_labels(self.cnf[ck.TRAIN_CSV_PATH])

    def split_data(self):
        """
        Splits data to train and test and creates data generators
        :return: Union[NeuronDataGenerator, NeuronDataGenerator]
        """

        # TODO drop unused
        df_groups = [self.train_df.get_group(x) for x in self.train_df.groups]
        split_index = int(len(df_groups) * self.cnf[ck.TRAIN_RATIO])

        # TODO: shuffle
        df_train = pd.concat(df_groups[:split_index]).groupby("id")
        df_validate = pd.concat(df_groups[split_index:]).groupby("id")

        data_generator_train = NeuronDataGenerator(self.cnf, df_train)
        data_generator_validate = NeuronDataGenerator(self.cnf, df_validate)
        return data_generator_train, data_generator_validate
