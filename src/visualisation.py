import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        img = display_list[i]
        if len(img.shape) == 2:
            img = cv2.merge([img, img, img])
        img = tf.keras.utils.array_to_img(img)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
