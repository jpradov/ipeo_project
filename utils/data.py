"""This module contains all function related to data loading and processing. """

import numpy as np
from skimage import exposure


def normalized_image(image):
    """
    Rescale image to values between 0 to 255 (capping outlier values)

    Parameters
    ==================
    image: Numpy array
        Image numpy array with shape (height, width, num_bands)

    percentiles: tuple
        Tuple of min and max percentiles to cap outlier values

    Returns
    ==================
    output: Numpy array
        Normalized image numpy array

    """
    output = np.zeros_like(image)
    for k in range(image.shape[2]):  # for each band
        p_min, p_max = np.min(image[:, :, k]), np.max(image[:, :, k])
        output[:, :, k] = exposure.rescale_intensity(
            image[:, :, k], in_range=(p_min, p_max), out_range=(0, 255)
        )
    return output.astype(np.uint8)
