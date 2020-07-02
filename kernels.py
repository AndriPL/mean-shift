import math

import numpy as np


# def gaussian_kernel(distance, bandwidth): # TODO vectorize
#     return (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(
#         -0.5 * ((distance / bandwidth)) ** 2
#     )

def flat_kernel(distances, bandwidth):
    weights = np.zeros_like(distances)
    weights[distances <= bandwidth] = 1
    return weights