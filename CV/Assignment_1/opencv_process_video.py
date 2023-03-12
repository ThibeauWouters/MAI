"""
Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import os
import sys
import numpy as np


# helper function to get the time in seconds
def get_time(cap):
    """
    Returns the time of the current frame in milliseconds.
    :param cap: VideoCapture
    :return: Time of frame in milliseconds.
    """
    return int(cap.get(cv2.CAP_PROP_POS_MSEC))


# helper function to change what you do based on video seconds
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

    # Specify hyperparameters (for during working on frames)
    show = True  # show the frame as well
    min_time = 35000
    max_time = 40000  # max time that we are going to process
    # Make sure max_time is at most 60000, i.e. 1 minute
    max_time = min(60000, max_time)
    print(f"Procesing frames between {min_time/1000} seconds and {max_time/1000} seconds.")
    font = cv2.FONT_HERSHEY_PLAIN

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            subtitle = ""
            explanation_subtitle = ""

            # If before min time, we don't handle those frames (used when constructing video)
            if between(cap, 0, min_time):
                continue

            "0 -- 4 seconds: color to grayscale"
            if between(cap, 0, 4000):
                subtitle = "Switch between greyscale and color"
            if between(cap, 0, 1000) or between(cap, 2000, 3000):
                # Convert to greyscale, but make sure we still have three channels (copy one channel to others)
                frame[:, :, 1] = frame[:, :, 0]
                frame[:, :, 2] = frame[:, :, 0]

            "4 -- 8 seconds: Gaussian blurring"
            if between(cap, 4000, 5000):
                kernel = (31, 31)
                sigma = 0
                frame = cv2.GaussianBlur(frame, kernel, sigma)

                subtitle = f"Gaussian filter: kernel {kernel}, sigma: {sigma}"

            if between(cap, 5000, 6000):
                kernel = (51, 51)
                sigma = 0
                frame = cv2.GaussianBlur(frame, kernel, sigma)

                subtitle = f"Gaussian filter: kernel {kernel}, sigma: {sigma}"

            if between(cap, 6000, 7000):
                kernel = (51, 51)
                sigma = 10
                frame = cv2.GaussianBlur(frame, kernel, sigma)

                subtitle = f"Gaussian filter: kernel {kernel}, sigma: {sigma}"

            if between(cap, 7000, 8000):
                kernel = (51, 51)
                sigma = 20
                frame = cv2.GaussianBlur(frame, kernel, sigma)

                subtitle = f"Gaussian filter: kernel {kernel}, sigma: {sigma}"

            "8 -- 12 seconds: bilateral filter"
            if between(cap, 8000, 10000):
                d = 15
                sigma_color = 10
                sigma_space = 10
                frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

                subtitle = f"Bilateral filter: d: {d}, sigma color {sigma_color}, sigma space: {sigma_space}"

            if between(cap, 10000, 12000):
                d = 15
                sigma_color = 50
                sigma_space = 50
                frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

                subtitle = f"Bilateral filter: d: {d}, sigma color {sigma_color}, sigma space: {sigma_space}"

            "12 -- 16 seconds: simple grabbing by thresholding"
            if between(cap, 12000, 16000):
                # Convert to HSV space
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Blur
                hsv = cv2.GaussianBlur(hsv, (13, 13), 0)
                # Specify threshold ranges
                lower = np.array([0, 150, 0])
                upper = np.array([179, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                # # Bitwise-AND mask and original image
                white = 255*np.ones_like(hsv)
                frame = cv2.bitwise_and(white, white, mask=mask)
                subtitle = f"Grabbing (threshold in HSV space)"

            "16 -- 20 seconds: improved grabbing"
            if between(cap, 16000, 20000):
                # Convert to HSV space
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                # Blur
                hsv = cv2.GaussianBlur(hsv, (13, 13), 0)
                # Specify threshold ranges
                lower = np.array([0, 150, 0])
                upper = np.array([179, 255, 255])
                mask = cv2.inRange(hsv, lower, upper)
                # Show new improvements in blue
                white = 255 * np.ones_like(hsv)
                blue = white
                blue[:, :, 1] = 0
                blue[:, :, 2] = 0
                # Bitwise-AND mask and original image
                frame = cv2.bitwise_and(white, white, mask=mask)
                # Now improve with morphological operations
                kernel_size = (21, 21)
                kernel = np.ones(kernel_size, np.uint8)
                # frame = cv2.erode(frame, kernel, iterations=1)
                # frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
                # kernel = np.ones((3, 3), np.uint8)
                # frame = cv2.dilate(frame, kernel, iterations=1)

                subtitle = f"Grabbing and closing (kernel: {kernel_size})"

            "20 -- 25 seconds: Sobel filter"
            if between(cap, 20000, 25000):
                # Blur for better results
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                # We vary the Sobel parameters according to time in these 5 seconds:
                ddepth = cv2.CV_16S
                # Default params
                scale = 1
                delta = 0
                x_val = 1
                y_val = 0
                if between(cap, 20000, 21000):
                    # Sobel parameters
                    scale = 1
                    delta = 0
                    x_val = 1
                    y_val = 0
                    # Subtitle
                    subtitle = f"Sobel filter: horizontal, scale: {scale}, delta: {delta}"

                elif between(cap, 21000, 22000):
                    # Sobel parameters
                    scale = 5
                    delta = 0
                    x_val = 1
                    y_val = 0
                    subtitle = f"Sobel filter: horizontal, scale: {scale}, delta: {delta}"

                elif between(cap, 22000, 23000):
                    # Sobel parameters
                    scale = 1
                    delta = 5
                    x_val = 0
                    y_val = 1
                    subtitle = f"Sobel filter: vertical, scale: {scale}, delta: {delta}"

                elif between(cap, 23000, 24000):
                    # Sobel parameters
                    scale = 1
                    delta = 10
                    x_val = 0
                    y_val = 1
                    subtitle = f"Sobel filter: vertical, scale: {scale}, delta: {delta}"

                elif between(cap, 24000, 25000):
                    # Sobel parameters
                    scale = 1
                    delta = 10
                    x_val = 0
                    y_val = 1
                    subtitle = f"Sobel filter: vertical, scale: {scale}, delta: {delta}"

                # We are going to apply the Sobel operator on each channel, split them:
                channels = cv2.split(frame)
                grads = []
                for channel in channels:
                    grad_x = cv2.Sobel(channel, ddepth, x_val, y_val, ksize=5, scale=scale, delta=delta,
                                       borderType=cv2.BORDER_DEFAULT)
                    grads.append(grad_x)
                # Merge back into one picture
                frame = cv2.merge(grads)
                # Convert the scale again
                frame = cv2.convertScaleAbs(frame)

            "25 -- 30s: Hough transform to detect circles"
            if between(cap, 25000, 35000):
                # Blur for better results
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                # Convert image to grayscale for Hough circles method:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Default params:
                minDist = gray.shape[0] / 16
                param1 = 150
                param2 = 50
                minR = 120
                maxR = 500

                # Parameters for each second:
                if between(cap, 25000, 27000):
                    param2 = 35
                    subtitle  = f"Hough transform: param1: {param1}, param2: {param2}, minRadius: {minR}, param2: {maxR}"
                    explanation_subtitle = "Gradients and hence circles easily pass threshold"

                elif between(cap, 27000, 29000):

                    subtitle  = f"Hough transform: param1: {param1}, param2: {param2}, minRadius: {minR}, param2: {maxR}"
                    explanation_subtitle = "Gradients and hence circles less easily pass threshold"

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
                print(np.shape(frame))
                # Read in a template
                template = cv2.imread("template_apple.png")

                # Blur a bit
                frame = cv2.GaussianBlur(frame, (3, 3), 0)
                template = cv2.GaussianBlur(template, (3, 3), 0)
                # Convert the images to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                # Get the dimensions of our template
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
                    # Normalize the values to grayscale
                    result = cv2.normalize(result, None, 0, 1, cv2.NORM_MINMAX)
                    # Convert to grayscale
                    result = (255 * result).astype(np.uint8)
                    # Resize the result variable to the same dimensions as the original frames
                    result = cv2.resize(result, (frame_width, frame_height))
                    # Merge into 3 color channels into frame to be written
                    frame = cv2.merge((result, result, result))
                subtitle = f"Template matching"

            # Write subtitle, with a timestamp, and save to write frame to output
            timestamp = '{:.3f}'.format(get_time(cap)/1000)
            cv2.putText(frame, subtitle + f" ({timestamp} s)", (int(0.1*frame_width), int(0.1*frame_height)), font, 2, (0, 0, 255), 2, cv2.LINE_4)
            cv2.putText(frame, explanation_subtitle, (int(0.1 * frame_width), int(0.13 * frame_height)), font,
                        2, (0, 0, 255), 2, cv2.LINE_4)
            out.write(frame)

            # (optional) display the resulting frame
            if show:
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)

            # Print progress of compiling new video
            # print(f"Progress: {min(int(round(100*get_time(cap)/max_time)), 100)} %", end="\r")
            # (optional) Stop processing if we exceed max_time
            if get_time(cap) > max_time:
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
    # Command line:

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

    # Inside Pycharm Project:

    input_file  = os.path.normpath("D:\\Coding\\MAI\\CV\\Assignment_1\\video2.MOV")
    output_file = os.path.normpath("D:\\Coding\\MAI\\CV\\Assignment_1\\out.mp4")

    main(input_file, output_file)
