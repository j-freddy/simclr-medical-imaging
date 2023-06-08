from PIL import Image, ImageEnhance, ImageOps
import numpy as np

from utils import convert_to_rgb, convert_to_ycbcr


class RandomAdjustSharpness:
    def __init__(self, factor_low, factor_high):
        """
        Adjust image sharpness with factor randomly chosen between factor_low
        and factor_high. A factor of 0 gives blurred image and a factor of 1
        gives the original image.

        Args:
            factor_low (float): The lower bound of the sharpness factor.
            factor_high (float): The upper bound of the sharpness factor.
        """
        self.factor_low = factor_low
        self.factor_high = factor_high

    def __call__(self, img):
        factor = np.random.uniform(self.factor_low, self.factor_high)
        return ImageEnhance.Sharpness(img).enhance(factor)


class RandomEqualize:
    def __init__(self, p=0.5):
        """
        Equalise the image histogram. Can be applied to colour images by
        converting to YCbCr format first (which separates raw intensity from
        other channels), applying equalisation, then converting back to RGB.

        Args:
            p (float, optional): The probability of applying equalization.
                Defaults to 0.5.
        """
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = convert_to_ycbcr(img)
            y, cb, cr = img.split()
            y_eq = ImageOps.equalize(y)
            img = Image.merge("YCbCr", (y_eq, cb, cr))
            img = convert_to_rgb(img)
            return img
        return img
