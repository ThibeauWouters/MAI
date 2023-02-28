
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


def bilateral_filter_example():

    # Load original image
    image = cv2.imread("apple.jpeg")
    # Edit it as you desire
    d = 10
    sigma_color = 50
    sigma_space = 50
    image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    # Show the output
    cv2.imshow("Image", image)
    cv2.waitKey(0)

def nothing(x):
    pass

def sliders_HSV(filename):
    # Load in image
    image = cv2.imread(filename)

    # Create a window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # create trackbars for color change
    cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
    cv2.createTrackbar('SMin','image',0,255,nothing)
    cv2.createTrackbar('VMin','image',0,255,nothing)
    cv2.createTrackbar('HMax','image',0,179,nothing)
    cv2.createTrackbar('SMax','image',0,255,nothing)
    cv2.createTrackbar('VMax','image',0,255,nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('HMax', 'image', 179)
    cv2.setTrackbarPos('SMax', 'image', 255)
    cv2.setTrackbarPos('VMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    output = image
    wait_time = 33

    while(1):

        # get current positions of all trackbars
        hMin = cv2.getTrackbarPos('HMin','image')
        sMin = cv2.getTrackbarPos('SMin','image')
        vMin = cv2.getTrackbarPos('VMin','image')

        hMax = cv2.getTrackbarPos('HMax','image')
        sMax = cv2.getTrackbarPos('SMax','image')
        vMax = cv2.getTrackbarPos('VMax','image')

        # Set minimum and max HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        output = cv2.bitwise_and(image,image, mask= mask)

        # Print if there is a change in HSV value
        if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
            print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax), end='\r')
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display output image
        im = cv2.resize(output, (540, 270))  # Resize image
        cv2.imshow('image', im)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def grabbing():
    # Load original image
    image = cv2.imread("apple.jpeg")
    # Show the output
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([155, 25, 0])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def sliders_RGB(filename):
    # Load in image
    image = cv2.imread(filename)

    # Create a window
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    # create trackbars for color change
    cv2.createTrackbar('BMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('GMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('RMin', 'image', 0, 255, nothing)
    cv2.createTrackbar('BMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('GMax', 'image', 0, 255, nothing)
    cv2.createTrackbar('RMax', 'image', 0, 255, nothing)

    # Set default value for MAX HSV trackbars.
    cv2.setTrackbarPos('BMax', 'image', 255)
    cv2.setTrackbarPos('GMax', 'image', 255)
    cv2.setTrackbarPos('RMax', 'image', 255)

    # Initialize to check if HSV min/max value changes
    BMin = GMin = RMin = BMax = GMax = RMax = 0
    pBMin = pGMin = pRMin = pBMax = pGMax = pRMax = 0

    output = image
    wait_time = 33

    while(1):

        # get current positions of all trackbars
        BMin = cv2.getTrackbarPos('BMin', 'image')
        GMin = cv2.getTrackbarPos('GMin', 'image')
        RMin = cv2.getTrackbarPos('RMin', 'image')

        BMax = cv2.getTrackbarPos('BMax', 'image')
        GMax = cv2.getTrackbarPos('GMax', 'image')
        RMax = cv2.getTrackbarPos('RMax', 'image')

        # Set minimum and max HSV values to display
        lower = np.array([BMin, GMin, RMin])
        upper = np.array([BMax, GMax, RMax])

        # Create HSV Image and threshold into a range.
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image, lower, upper)
        output = cv2.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if( (pBMin != BMin) | (pGMin != GMin) | (pRMin != RMin) | (pBMax != BMax) | (pGMax != GMax) | (pRMax != RMax) ):
            print("(BMin = %d , GMin = %d, RMin = %d), (BMax = %d , GMax = %d, RMax = %d)" % (BMin, GMin, RMin, BMax, GMax, RMax), end='\r')
            pBMin = BMin
            pGMin = GMin
            pRMin = RMin
            pBMax = BMax
            pGMax = GMax
            pRMax = RMax

        # Display output image
        # output = cv2.resize(output, (540, 270))  # Resize image
        cv2.imshow('image', output)

        # Wait longer to prevent freeze for videos.
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def grabbing():
    # Load original image
    image = cv2.imread("apple.jpeg")
    # Show the output
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower = np.array([155, 25, 0])
    upper = np.array([179, 255, 255])
    image = cv2.inRange(image, lower, upper)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


# grabbing()
sliders_HSV('onion2.jpg')
# sliders_RGB('frame.jpg')
