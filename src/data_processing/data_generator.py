import pickle
import cv2
import numpy as np
import pandas as pd
import random

from tensorflow import keras
from typing import Dict

# START PROJECT IMPORTS
from .utilities import load_train_labels
from ..constants import WIDTH, HEIGHT, REDUCED_HEIGHT, ConfigKeys as ck
from ..models.mrcnn.utils import Dataset as MrcnnDataSet
# END PROJECT_IMPORTS


class DynamicDataGenerator(keras.utils.Sequence):
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
        # self.rescaling = layers.Rescaling(1. / 255) # TODO: -mean / std
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
            image = image.astype(float) / 255
            # image = self.rescaling(image)
            # assert 0 <= image.numpy().min() < image.numpy().max() <= 1
            assert 0 <= image.min() < image.max() <= 1
            x[i, ] = image
        return x

    def get_y(self, photo_ids):
        """
        Generates data containing batch_size masks
        :param photo_ids: list of ids each representing one image
        :return: batch of masks
        """
        y = np.empty((self.batch_size, *self.image_dim, 1), dtype="int")
        flag = True
        for i, photo_id in enumerate(photo_ids):
            with open(f"{self.cnf[ck.MASK_DIR_PATH]}/mask_{photo_id}.pkl", 'rb') as f:
                image = pickle.load(f)
                if len(image.shape) == 2:
                    image = cv2.resize(image, (image.shape[1], REDUCED_HEIGHT))
                    image = np.round(image / 255)
                    image = np.reshape(image, (*image.shape, 1))
                else:
                    new_images = []
                    for idx in range(image.shape[0]):
                        mask = cv2.resize(image[idx], (WIDTH, REDUCED_HEIGHT)) / 255
                        new_images.append(mask)
                    image = np.asarray(new_images)
                    image = np.transpose(image, (1, 2, 0))
                    if flag:
                        flag = False
                        y = np.empty((self.batch_size, *image.shape), dtype="int")
                assert image.min() == 0 and image.max() == 1
                y[i] = image
        return y


class StaticDataGenerator(keras.utils.Sequence):
    def __init__(self, cnf: Dict, grouped_labels: pd.core.groupby.GroupBy, train_mode=True):
        """
        Initialization
        :param cnf: config
        :param grouped_labels: grouped dataframe by id
        """
        self.cnf = cnf
        self.batch_size = cnf[ck.BATCH_SIZE]
        self.shuffle = cnf[ck.SHUFFLE]
        self.train_mode = train_mode

        self.image_dim = (REDUCED_HEIGHT, WIDTH)     # TODO: try not reshape
        self.new_dim = None
        self.channels = 3
        self.indexes = []

        self.n_images = len(grouped_labels)
        photo_ids = list(grouped_labels.groups.keys())
        # rescaling = layers.Rescaling(1. / 255) # TODO: -mean / std

        self.images = np.ndarray((self.n_images, *self.image_dim, self.channels))
        self.masks = np.ndarray((self.n_images, *self.image_dim, 1))

        # self.image_info = []

        flag = True
        for i, photo_id in enumerate(photo_ids):
            # self.image_info.append({
            #     'id': i,
            #     'source': None,
            #     'path': f"{self.cnf[ck.IMAGES_DIR_PATH]}/{photo_id}.png"
            # })
            # if i == 20:
            #     self.n_images = 20
            #     break
            # load image
            image = cv2.imread(f"{self.cnf[ck.IMAGES_DIR_PATH]}/{photo_id}.png")
            # image = rescaling(cv2.resize(image, (image.shape[1], REDUCED_HEIGHT)))       # normalize
            image = cv2.resize(image, (image.shape[1], REDUCED_HEIGHT)).astype(float) / 255
            # assert 0 <= image.numpy().min() < image.numpy().max() <= 1
            assert 0 <= image.min() < image.max() <= 1
            self.images[i] = image

            if self.train_mode:
                # load mask
                with open(f"{self.cnf[ck.MASK_DIR_PATH]}/mask_{photo_id}.pkl", 'rb') as f:
                    mask = pickle.load(f)
                    if len(mask.shape) == 2:
                        mask = cv2.resize(mask, (mask.shape[1], REDUCED_HEIGHT))
                        mask = np.round(mask / 255)
                        mask = np.reshape(mask, (*mask.shape, 1))
                    else:
                        new_images = []
                        for idx in range(mask.shape[0]):
                            mask_curr = cv2.resize(mask[idx], (WIDTH, REDUCED_HEIGHT)) / 255
                            new_images.append(mask_curr)
                        mask = np.asarray(new_images)
                        mask = np.transpose(mask, (1, 2, 0))
                        if flag:
                            flag = False
                            self.masks = np.empty((self.n_images, *mask.shape))
                            self.new_dim = mask.shape
                    assert mask.min() == 0 and mask.max() == 1
                    self.masks[i] = mask
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(self.n_images / self.batch_size))

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index: index of the batch
        :return: x and y
        """
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        x = np.empty((self.batch_size, *self.image_dim, self.channels), dtype="float")
        if self.new_dim is None:
            y = np.empty((self.batch_size, *self.image_dim), dtype="int")
        else:
            y = np.empty((self.batch_size, *self.new_dim), dtype="int")
        for i, idx in enumerate(indexes):
            x[i] = self.images[idx]
            if self.train_mode:
                y[i] = self.masks[idx]

        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(self.n_images)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    # def load_image(self, image_id):
    #     return self.images[image_id]

    # def load_mask(self, image_id):
    #     return self.masks[image_id]
    
    # def image_reference(self, image_id):
    #     return ""

    # @property
    # def image_ids(self):
    #     return self.indexes


class DataLoader:
    """
    Loads data
    """

    def __init__(self, cnf: Dict, type_: str= "dynamic"):
        """
        Initialisation
        :param cnf: config
        """
        self.cnf = cnf
        self.train_df = None
        self.generator = DynamicDataGenerator if type_ == "dynamic" else StaticDataGenerator

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

        df_groups = [self.train_df.get_group(x) for x in self.train_df.groups]
        split_index = int(len(df_groups) * self.cnf[ck.TRAIN_RATIO])

        random.shuffle(df_groups)
        df_train = pd.concat(df_groups[:split_index]).groupby("id")
        df_validate = pd.concat(df_groups[split_index:]).groupby("id")

        print("Loading training data")
        data_generator_train = self.generator(self.cnf, df_train)
        print("Loading validation data")
        data_generator_validate = self.generator(self.cnf, df_validate)
        return data_generator_train, data_generator_validate


class SubmissionDataLoader:
    """
    Loads submission data
    """

    def __init__(self, cnf: Dict):
        """
        Initialisation
        :param cnf: config
        """
        self.cnf = cnf
        self.df = None
        self.submission_df = None
        self.generator = StaticDataGenerator

    def load_data(self):
        """
        Loads data from file
        :return: NeuronDataGenerator
        """
        self.df = pd.read_csv(self.cnf[ck.SUBMISSION_CSV_PATH])
        self.submission_df = self.df.groupby("id")
        return self.generator({
            ck.BATCH_SIZE: 1,
            ck.SHUFFLE: False,
            ck.IMAGES_DIR_PATH: self.cnf[ck.SUBMISSION_DIR_PATH]
        }, self.submission_df, train_mode=False)


class MyMrcnnDataSet(MrcnnDataSet):
    def load_data(self, cnf: Dict, grouped_labels: pd.core.groupby.GroupBy, train_mode=True):
        """
        Initialization
        :param cnf: config
        :param grouped_labels: grouped dataframe by id
        """
        self.add_class("sartorius", 1, "neuron")


        self.cnf = cnf
        self.train_mode = train_mode

        self.image_dim = (REDUCED_HEIGHT, WIDTH)     # TODO: try not reshape
        self.new_dim = None
        self.channels = 3

        self.n_images = len(grouped_labels)
        # self.n_images = 20
        photo_ids = list(grouped_labels.groups.keys())
        self.images = np.ndarray((self.n_images, *self.image_dim, self.channels))
        self.masks = np.ndarray((self.n_images, *self.image_dim, 1))

        # self.image_info = []

        flag = True
        for i, photo_id in enumerate(photo_ids):
            # if i == 20:
                # break
            # load image
            image = cv2.imread(f"{self.cnf[ck.IMAGES_DIR_PATH]}/{photo_id}.png")
            # image = rescaling(cv2.resize(image, (image.shape[1], REDUCED_HEIGHT)))       # normalize
            image = cv2.resize(image, (image.shape[1], REDUCED_HEIGHT))#.astype(float) / 255
            # assert 0 <= image.numpy().min() < image.numpy().max() <= 1
            # assert 0 <= image.min() < image.max() <= 1
            self.images[i] = image

            self.add_image("sartorius", image_id=i, path=None, width=WIDTH, height=HEIGHT)

            if self.train_mode:
                # load mask
                with open(f"{self.cnf[ck.MASK_DIR_PATH]}/mask_{photo_id}.pkl", 'rb') as f:
                    mask = pickle.load(f)
                    if len(mask.shape) == 2:
                        mask = cv2.resize(mask, (mask.shape[1], REDUCED_HEIGHT))
                        mask = np.round(mask / 255)
                        mask = np.reshape(mask, (*mask.shape, 1))
                    else:
                        new_images = []
                        for idx in range(mask.shape[0]):
                            mask_curr = cv2.resize(mask[idx], (WIDTH, REDUCED_HEIGHT)) / 255
                            new_images.append(mask_curr)
                        mask = np.asarray(new_images)
                        mask = np.transpose(mask, (1, 2, 0))
                        if flag:
                            flag = False
                            self.masks = np.empty((self.n_images, *mask.shape))
                            self.new_dim = mask.shape
                    assert mask.min() == 0 and mask.max() == 1
                    self.masks[i] = mask

    def load_image(self, image_id):
        # return np.array([self.images[image_id]])
        # print("<<<<<<<<<<<<<<<<<")
        # print( np.expand_dims(self.images[image_id], -1).shape)
        # return np.expand_dims(self.images[image_id], -1)
        return self.images[image_id]

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        # mask = 
        # print(">>>>>>>>>>>>>>")
        # print(np.expand_dims(self.masks[image_id].astype(bool), -1).shape)
        return self.masks[image_id].astype(bool), np.array([1,], dtype=np.int32)

def mrcnn_get_train_valid_datasets(cnf: Dict):
    train_df = load_train_labels(cnf[ck.TRAIN_CSV_PATH])
    df_groups = [train_df.get_group(x) for x in train_df.groups]
    split_index = int(len(df_groups) * cnf[ck.TRAIN_RATIO])

    random.shuffle(df_groups)
    df_train = pd.concat(df_groups[:split_index]).groupby("id")
    df_validate = pd.concat(df_groups[split_index:]).groupby("id")

    print("Loading training data")
    ds_train = MyMrcnnDataSet()
    ds_train.load_data(cnf, grouped_labels=df_train)
    ds_train.prepare()
    print("Loading validation data")
    ds_val = MyMrcnnDataSet()
    ds_val.load_data(cnf, grouped_labels=df_validate)
    ds_val.prepare()
    return ds_train, ds_val