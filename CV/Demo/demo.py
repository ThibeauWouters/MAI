
"""
KUL Computer Vision demo
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2

def numpy_example():
    zeros = np.zeros((100, 100))
    hundreds = np.full_like(zeros, 100)
    two_hundreds = np.full_like(zeros, 200)

    concatenated = np.concatenate([zeros, hundreds, two_hundreds], axis=1)
    noisy = concatenated + 50 * (np.random.rand(100, 300) - 0.5)

    plt.figure()
    # plt.imshow(concatenated, cmap="gray")
    plt.imshow(noisy, cmap="gray")
    plt.show()

def cv_example():
    image = cv2.imread("nighthawks.jpg")
    print(image[:, :, 0])
    #print(image[:, :, 1])
    #print(image[:, :, 2])
    plt.figure()
    plt.imshow(image)
    plt.show()

    # Smoothen it with a Gaussian kernel
    blur = cv2.GaussianBlur(image, (5, 5    ), 0)
    plt.figure()
    plt.imshow(blur)
    plt.show()

cv_example()
