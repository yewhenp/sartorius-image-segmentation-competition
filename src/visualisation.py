import matplotlib.pyplot as plt
import numpy as np
import cv2


def display(display_list):
    fig = plt.figure()
    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        ax = fig.add_subplot(1, 3, i+1)
        img = display_list[i]
        if len(img.shape) == 2:
            img = cv2.merge([img, img, img])
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        plt.imshow(np.squeeze(np.array(img)))
        ax.set_title(title[i])
    plt.show()
