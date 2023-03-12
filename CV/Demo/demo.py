
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
        output = cv2.bitwise_and(image, image, mask=mask)

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
    # Create a window for showing later on
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # Load original image
    frame = cv2.imread("onion2.jpg")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Blur
    # hsv = cv2.GaussianBlur(hsv, (3, 3), 0)
    # Specify threshold ranges
    lower = np.array([0, 150, 15])
    upper = np.array([179, 250, 250])
    mask = cv2.inRange(hsv, lower, upper)
    # # Bitwise-AND mask and original image
    white = 255 * np.ones_like(hsv)
    frame = cv2.bitwise_and(white, white, mask=mask)
    # Show
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    # Now improve with morphological operations
    kernel = np.ones((7, 7), np.uint8)
    # frame = cv2.erode(frame, kernel, iterations=1)
    # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
    frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
    # kernel = np.ones((3, 3), np.uint8)
    # frame = cv2.dilate(frame, kernel, iterations=1)
    # Show
    cv2.imshow('Second', frame)
    cv2.waitKey(0)


def sobel():
    # Create a window for showing later on
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # Load original image
    frame = cv2.imread("onion.jpg")
    # Blur for better results
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    # We are going to apply the Sobel operator on each channel
    channels = cv2.split(frame)
    # Sobel parameters
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grads = []
    for channel in channels:
        grad_x = cv2.Sobel(channel, ddepth, 1, 0, ksize=5, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # Append the final result together again
        grads.append(grad_x)
    # Merge back into one picture
    frame = cv2.merge(grads)
    # Convert the scale again
    frame = cv2.convertScaleAbs(frame)
    # Show
    cv2.imshow('Frame', frame)
    cv2.waitKey()


def hough_circles():
    # Create a window for showing later on
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # Load original image
    frame = cv2.imread("oranges.jpg")
    # Blur for better results
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    # Convert image to grayscale for Hough circles:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, gray.shape[0] / 16, param1=100, param2=50, minRadius=120, maxRadius=500)
    # Draw detected circles on the original frame
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw inner circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

    # Convert the scale again
    cv2.imshow('Frame', frame)
    cv2.waitKey()


def template_matching():
    # Create a window for showing later on
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # Load original image
    frame = cv2.imread("frame.png")
    template = cv2.imread("template_apple.png")

    # Blur a bit
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    template = cv2.GaussianBlur(template, (3, 3), 0)
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(np.shape(gray))
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = gray_template.shape[::-1]
    # All possible methods:
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    meth = methods[0]
    method = eval(meth)
    # Apply template Matching
    res = cv2.matchTemplate(gray, gray_template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(gray, top_left, bottom_right, 255, 2)

    # What are the dimensions
    print(np.shape(gray))

    cv2.imshow('Frame', gray)
    cv2.waitKey()


def template_matching_likelihood():
    # Create a window for showing later on
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # Load original image
    frame = cv2.imread("frame.png")
    height, width, channels = frame.shape
    template = cv2.imread("template_apple_4.png")

    # Blur a bit
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    template = cv2.GaussianBlur(template, (3, 3), 0)
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = gray_template.shape[::-1]
    # All possible methods:
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
               'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    # meth = methods[0]
    meth = 'cv2.TM_CCOEFF_NORMED'
    method = eval(meth)
    # Apply template Matching
    result = cv2.matchTemplate(gray, gray_template, method)
    cv2.imshow('Frame', result)
    cv2.waitKey()
    # Normalize the values to grayscale
    result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
    # Convert to grayscale
    result = (255*result).astype(np.uint8)
    # Resize the result variable to the same dimensions as the original frames
    result = cv2.resize(result, (width, height))
    # Merge into 3 color channels
    result = cv2.merge((result, result, result))
    print(result)
    cv2.imshow('Frame', result)
    cv2.waitKey()


def features_demo():
    # Create a window for showing later on
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
    # Load original image
    frame = cv2.imread("other_1.png")
    # height, width, channels = frame.shape
    template = cv2.imread("template_1.png")

    # Convert both to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray_template, 25, 0.01, 10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(gray_template, (x, y), 3, 255, -1)

    cv2.imshow("Frame", gray_template)
    cv2.waitKey(0)


features_demo()
# template_matching()
# template_matching_likelihood()
# hough_circles()
# sobel()
# sliders_HSV('dice_frame.png')
# sliders_RGB('frame.jpg')


