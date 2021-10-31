
import matplotlib.pyplot as plt
import numpy as np


def image_plot(images, captions=None, cmap=None ):
    f, axes = plt.subplots(1, len(images), sharey=True)
    f.set_figwidth(15)
    for ax,image in zip(axes, images):
        ax.imshow(image, cmap)


def image_show(image):
    plt.imshow(image)


def image_normalize(image):
    norm1_image = image / 255
    norm2_image = image - np.min(image) / np.max(image) - np.min(image)
    norm3_image = image - np.percentile(image, 5) / np.percentile(image, 95) - np.percentile(image, 5)
    image_plot([image, norm1_image, norm2_image, norm3_image], cmap='gray')


def image_leftRightFlip(image):
    image_flipr = np.fliplr(image)
    image_plot([image, image_flipr])


def image_upDownFlip(image):
    image_flipud = np.flipud(image)
    image_plot([image, image_flipud])