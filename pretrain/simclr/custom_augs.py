from PIL import Image, ImageEnhance, ImageOps
import numpy as np

from utils import convert_to_rgb, convert_to_ycbcr


class RandomAdjustSharpness:
    def __init__(self, factor_low, factor_high):
        self.factor_low = factor_low
        self.factor_high = factor_high

    def __call__(self, img):
        factor = np.random.uniform(self.factor_low, self.factor_high)
        return ImageEnhance.Sharpness(img).enhance(factor)


class RandomEqualize:
    def __init__(self, p=0.5):
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
