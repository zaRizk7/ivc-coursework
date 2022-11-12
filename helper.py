import cv2
import numpy as np
from scipy.signal import convolve2d

GAUSSIAN_3X3_WEIGHT = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
GAUSSIAN_3X3_WEIGHT = np.divide(GAUSSIAN_3X3_WEIGHT, 16)


def load_image(directory, size=None):
    image = cv2.imread(directory)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if not size is None:
        image = cv2.resize(image, size)
    return np.uint16(image)


def clip(image):
    return np.uint16(np.clip(image, 0, 255))


def convolve(image, weight):
    args = dict(mode="same", boundary="symm")
    size = np.shape(image)
    if len(size) < 3:
        image = convolve2d(image, weight, **args)
        return clip(image)
    for i in range(size[-1]):
        image[..., i] = convolve2d(image[..., i], weight, **args)
    return clip(image)


def gaussian_blur(image, num_convolve):
    image_copy = np.copy(image)

    for _ in range(num_convolve):
        image_copy = convolve(image_copy, GAUSSIAN_3X3_WEIGHT)

    return image_copy


def gaussian_pixel_noise(image, std):
    size = np.shape(image)
    noise = np.random.normal(scale=std, size=size)
    return clip(image + noise)


def scale_contrast(image, scale):
    return clip(image * scale)


def change_brightness(image, value):
    return clip(image + value)


def occlusion(image, edge_length):
    size = np.shape(image)

    h_start = np.random.randint(image.shape[0] - edge_length)
    h_end = h_start + edge_length

    w_start = np.random.randint(image.shape[1] - edge_length)
    w_end = w_start + edge_length

    mask = np.zeros([edge_length] * 2).astype(np.int16)
    if len(size) > 2:
        mask = np.expand_dims(mask, -1)

    image_copy = np.copy(image)
    image_copy[h_start:h_end, w_start:w_end] = mask

    return clip(image_copy)


def salt_and_pepper(image, rate):
    size = np.shape(image)
    mask1 = np.random.random(size) < rate
    mask2 = np.random.random(size) < 0.5

    image_copy = np.copy(image)

    image_copy[np.bitwise_and(mask1, mask2)] = 0
    image_copy[np.bitwise_and(mask1, ~mask2)] = 255

    return clip(image_copy)
