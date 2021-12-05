import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
matplotlib.use('GTK4Agg')


def display(display_list):
    fig = plt.figure()
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        ax = fig.add_subplot(1, 3, i+1)
        img = display_list[i]
        if len(img.shape) == 2:
            img = cv2.merge([img, img, img])
        img = tf.keras.utils.array_to_img(img)
        plt.imshow(np.array(img))
        ax.set_title(title[i])
    plt.show()
