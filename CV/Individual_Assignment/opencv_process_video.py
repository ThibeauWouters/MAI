"""
Code for the first 40 seconds of the video submitted for individual assignment 1 of Computer Vision course.
By Thibeau Wouters
"""
import argparse
import cv2
import os
import sys
import numpy as np

# Specify hyperparameters that are helpful when working on the videos
SHOW_FRAMES = False  # show the frames when running this script
SHOW_PROGRESS = True  # show timestamp when processing video
MIN_TIME = 0  # start time of video to be processed
MAX_TIME = 40000  # max time that we are going to process
# Make sure max time is at most 40000 for this first part of the video
MAX_TIME = min(40000, MAX_TIME)


# helper function to get the time in seconds
def get_time(cap):
    """
    Returns the time of the current frame in milliseconds.
    :param cap: VideoCapture
    :return: Time of frame in milliseconds.
    """
    return int(cap.get(cv2.CAP_PROP_POS_MSEC))


# helper function to specify what to do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Show which part of the video we are going to work on
    print(f"Processing frames between {MIN_TIME/1000} seconds and {MAX_TIME/1000} seconds.")
    font = cv2.FONT_HERSHEY_PLAIN

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Initialize the subtitles
            subtitle = ""
            explanation_subtitle = ""

            # Skip frames that occur before the min time
            if between(cap, 0, MIN_TIME):
                continue

            "0 -- 4 seconds: color to grayscale"
            if between(cap, 0, 4000):
                subtitle = "Switch between greyscale and color"
            if between(cap, 0, 1000) or between(cap, 2000, 3000):
                # Convert to greyscale, but make sure we still have three channels for VideoWriter
                frame[:, :, 1] = frame[:, :, 0]
                frame[:, :, 2] = frame[:, :, 0]

            "4 -- 8 seconds: Gaussian blurring"
            if between(cap, 4000, 5000):
                kernel = (11, 11)
                sigma = 0
                frame = cv2.GaussianBlur(frame, kernel, sigma)

                subtitle = "Gaussian filtering"
                explanation_subtitle = f"Kernel {kernel}, sigma: {sigma}"

            if between(cap, 5000, 6000):
                kernel = (31, 31)
                sigma = 0
                frame = cv2.GaussianBlur(frame, kernel, sigma)

                subtitle = "Gaussian filtering"
                explanation_subtitle = f"Kernel {kernel}, sigma: {sigma}:  bigger kernel, smoother result"

            if between(cap, 6000, 7000):
                kernel = (31, 31)
                sigma = 20
                frame = cv2.GaussianBlur(frame, kernel, sigma)

                subtitle = "Gaussian filtering"
                explanation_subtitle = f"Kernel {kernel}, sigma: {sigma}: bigger sigma, smoother result"

            if between(cap, 7000, 8000):
                kernel = (51, 51)
                sigma = 50
                frame = cv2.GaussianBlur(frame, kernel, sigma)

                subtitle = "Gaussian filtering"
                explanation_subtitle = f"Kernel {kernel}, sigma: {sigma}: bigger kernel + bigger sigma, smoother result"

            "8 -- 12 seconds: bilateral filter"
            if between(cap, 8000, 10000):
                # The pixel neighbourhood size will be computed automatically from sigma space by putting -1
                d = -1
                sigma_color = 10
                sigma_space = 10
                frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

                subtitle = "Bilateral filter: preserves edges better, due to 2 weightings rather than 1"
                explanation_subtitle = f"sigma color {sigma_color}, sigma space: {sigma_space}"

            if between(cap, 10000, 12000):
                d = -1
                sigma_color = 20
                sigma_space = 20
                frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

                subtitle = "Bilateral filter: preserves edges better, due to 2 weightings rather than 1"
                explanation_subtitle = f"sigma color {sigma_color}, sigma space: {sigma_space}: smoother result"

            "12 -- 16 seconds: simple grabbing by thresholding"
            if between(cap, 12000, 16000):
                # Convert to HSV space
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Blur
                hsv = cv2.GaussianBlur(hsv, (13, 13), 0)
                # Specify threshold ranges
                lower = np.array([0, 100, 55])
                upper = np.array([179, 150, 150])
                mask = cv2.inRange(hsv, lower, upper)
                # Bitwise-AND mask on the original image
                white = 255*np.ones_like(hsv)
                frame = cv2.bitwise_and(white, white, mask=mask)
                subtitle = "Grabbing by threshold in HSV space"
                explanation_subtitle = "(hard to distinguish between similarly coloured objects)"

            "16 -- 20 seconds: improved grabbing"
            if between(cap, 16000, 20000):
                # Convert to HSV space
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Blur
                hsv = cv2.GaussianBlur(hsv, (13, 13), 0)
                # Specify threshold ranges
                lower = np.array([0, 100, 55])
                upper = np.array([179, 150, 150])
                mask = cv2.inRange(hsv, lower, upper)
                # Show new improvements in blue
                white = 255 * np.ones_like(hsv)
                blue = white
                blue[:, :, 1] = 0
                blue[:, :, 2] = 0
                # Bitwise-AND mask and original image
                frame = cv2.bitwise_and(white, white, mask=mask)
                # Now improve with morphological operations: use opening
                kernel_size = (15, 15)
                kernel = np.ones(kernel_size, np.uint8)
                nb_iterations = 2
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=nb_iterations)

                subtitle = "Less false positives with morphological operation"
                explanation_subtitle = f"(opening with kernel: {kernel_size}, iterations: {nb_iterations})"

            "20 -- 25 seconds: Sobel filter"
            if between(cap, 20000, 25000):
                # Blur
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                ddepth = cv2.CV_16S
                # We vary the Sobel parameters during these 5 seconds:
                x_val = 1
                y_val = 0
                ksize = 3
                scale = 1
                delta = 0

                if between(cap, 20000, 21000):
                    # Sobel parameters
                    x_val = 1
                    y_val = 0

                    subtitle = "Sobel filter: horizontal (gradients in blue)"
                    explanation_subtitle = f"(kernel size: {ksize})"

                elif between(cap, 21000, 22000):
                    # Sobel parameters
                    x_val = 1
                    y_val = 0
                    ksize = 5
                    scale = 0.15

                    subtitle = "Sobel filter: horizontal (gradients in blue)"
                    explanation_subtitle = f"(kernel size: {ksize})"

                elif between(cap, 22000, 23000):
                    # Sobel parameters
                    x_val = 0
                    y_val = 1
                    ksize = 3

                    subtitle = "Sobel filter: vertical (gradients in blue)"
                    explanation_subtitle = f"(kernel size: {ksize})"

                elif between(cap, 23000, 24000):
                    # Sobel parameters
                    x_val = 0
                    y_val = 1
                    ksize = 5
                    scale = 0.15

                    subtitle = "Sobel filter: vertical (gradients in blue)"
                    explanation_subtitle = f"(kernel size: {ksize})"

                elif between(cap, 24000, 25000):
                    # Sobel parameters
                    x_val = 1
                    y_val = 1
                    ksize = 3
                    scale = 3

                    subtitle = "Sobel filter: diagonal (gradients in blue)"
                    explanation_subtitle = f"(kernel size: {ksize})"

                # Convert color to gray
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Detect gradients on the grayscale image, use params defined above
                grad = cv2.Sobel(gray, ddepth, x_val, y_val, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
                # Convert the scale
                grad = cv2.convertScaleAbs(grad)
                # Get the mask of ROI, where we have the Sobel gradients, and the inverse mask
                ret, mask = cv2.threshold(grad, 50, 255, cv2.THRESH_BINARY)
                inv_mask = cv2.bitwise_not(mask)
                # Get the ROI of frame, but black out Sobel
                frame = cv2.bitwise_and(frame, frame, mask=inv_mask)
                # Get the Sobel gradients in blue:
                blue = np.zeros_like(frame)
                blue[:, :, 0] = mask
                # Add the Sobel derivatives on top of the original image
                frame = cv2.add(frame, blue)

            "25 -- 35s: Hough transform to detect circles"
            if between(cap, 25000, 35000):
                # Blur for better results
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                # Convert image to grayscale for Hough circles method:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Default params of Hough transform:
                minDist = gray.shape[0] / 16
                param1 = 150
                param2 = 50
                minR = 120
                maxR = 500

                # Parameters for each period:
                if between(cap, 25000, 27000):
                    param2 = 35

                    subtitle  = f"Hough transform: param1: {param1}, param2: {param2}, minRadius: {minR}, param2: {maxR}"
                    explanation_subtitle = "Circles easily accepted"

                elif between(cap, 27000, 29000):

                    subtitle  = f"Hough transform: param1: {param1}, param2: {param2}, minRadius: {minR}, param2: {maxR}"
                    explanation_subtitle = "Circles less easily accepted"

                elif between(cap, 29000, 31000):
                    minDist = gray.shape[0] / 4
                    param1 = 100
                    param2 = 50

                    subtitle  = f"Hough transform: minDist: {minDist}, param1: {param1}, param2: {param2}, minR: {minR}, maxR: {maxR}"
                    explanation_subtitle = "Different centers are farther away"

                elif between(cap, 31000, 33000):
                    minR = 25
                    maxR = 75
                    minDist = gray.shape[0] / 12
                    param2 = 45

                    subtitle  = f"Hough transform: minDist: {minDist}, param1: {param1}, param2: {param2}, minR: {minR}, maxR: {maxR}"
                    explanation_subtitle = "Smaller circles detected"

                elif between(cap, 33000, 35000):
                    minR = 150
                    maxR = 200
                    minDist = gray.shape[0] / 2
                    param1 = 250
                    param2 = 20

                    subtitle  = f"Hough transform: minDist: {minDist}, param1: {param1}, param2: {param2}, minR: {minR}, maxR: {maxR}"
                    explanation_subtitle = "Ideal parameters to detect only the apple"

                # Detect circles with Hough transform with above specified parameters
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=param1, param2=param2,
                                           minRadius=minR, maxRadius=maxR)
                # Draw detected circles on the original frame
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        # Draw outer circle
                        cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        # Draw inner circle
                        cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)

            "35 - 40s: template matching"
            if between(cap, 35000, 40000):
                subtitle = "Template matching"
                # Read in a template
                template = cv2.imread("template_apple.png")

                # Blur a bit
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                template = cv2.GaussianBlur(template, (3, 3), 0)
                # Convert the images to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                # Get the dimensions (width and height) of the template
                w, h = gray_template.shape[::-1]
                # Choose a particular method and evaluate it
                meth = 'cv2.TM_CCOEFF_NORMED'
                method = eval(meth)
                # Apply template Matching
                result = cv2.matchTemplate(gray, gray_template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                if between(cap, 35000, 37000):
                    # Draw flashy rectangle for first two seconds
                    top_left = max_loc
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 10)
                    explanation_subtitle = "Max/min gives most probable location"

                if between(cap, 37000, 40000):
                    # Show likelihood
                    explanation_subtitle = "Showing likelihood"
                    # First, normalize to float values between 0 and 255
                    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
                    # Convert to integers
                    result = result.astype(np.uint8)
                    # Resize the result variable to the same dimensions as the original frames
                    result = cv2.resize(result, (frame_width, frame_height))
                    # Merge into 3 color channels into frame to be written
                    frame = cv2.merge((result, result, result))

            # Write subtitle, with a timestamp, and save to write frame to output
            timestamp = '{:.3f}'.format(get_time(cap)/1000)
            cv2.putText(frame, subtitle + f" ({timestamp} s)", (int(0.1*frame_width), int(0.1*frame_height)), font, 2, (0, 0, 255), 2, cv2.LINE_4)
            cv2.putText(frame, explanation_subtitle, (int(0.1 * frame_width), int(0.13 * frame_height)), font,
                        2, (0, 0, 255), 2, cv2.LINE_4)
            out.write(frame)

            # (optional) display the resulting frame
            if SHOW_FRAMES:
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)

            if SHOW_PROGRESS:
                print(timestamp, end="\r")

            # Stop processing if we exceed max_time
            if get_time(cap) > MAX_TIME:
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Specify in and out paths:
    # 1) When using the command line:

    # parser = argparse.ArgumentParser(description='OpenCV video processing')
    # parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    # parser.add_argument('-o', "--output", help='full path for saving processed video output')
    # args = parser.parse_args()
    #
    # if args.input is None or args.output is None:
    #     sys.exit("Please provide path to input and output video files! See --help")
    #
    # input_file  = args.input
    # output_file = args.output

    # 2) When using Pycharm:

    input_file  = os.path.normpath("D:\\Coding\\MAI\\CV\\Assignment_1\\in.MOV")
    output_file = os.path.normpath("D:\\Coding\\MAI\\CV\\Assignment_1\\out.mp4")

    # Then, run the above code
    main(input_file, output_file)
