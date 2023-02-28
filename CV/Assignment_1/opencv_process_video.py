"""
Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
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
    min_time = 12000
    max_time = 16000  # max time that we are going to process
    # Make sure max_time is at most 60000, i.e. 1 minute
    max_time = min(60000, max_time)
    print(f"Procesing frames between {min_time/1000} seconds and {max_time/1000} seconds.")
    font = cv2.FONT_HERSHEY_PLAIN
    subtitle = ""

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:

            # If before min time, we don't handle those frames (used when constructing video)
            if between(cap, 0, min_time):
                continue

            "0 -- 4 seconds"
            if between(cap, 0, 1000) or between(cap, 2000, 3000):
                # Convert to greyscale, but make sure we still have three channels (copy one channel to others)
                frame[:, :, 1] = frame[:, :, 0]
                frame[:, :, 2] = frame[:, :, 0]
                subtitle = "Switch between greyscale and color"

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

            # Write subtitle, with a timestamp, and save to write frame to output
            timestamp = '{:.3f}'.format(get_time(cap)/1000)
            cv2.putText(frame, subtitle + f" ({timestamp} s)", (int(0.1*frame_width), int(0.1*frame_height)), font, 2, (0, 0, 255), 2, cv2.LINE_4)
            out.write(frame)

            # (optional) display the resulting frame
            if show:
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)

            # Print progress of compiling new video
            print(f"Progress: {min(int(round(100*get_time(cap)/max_time)), 100)} %", end="\r")
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
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
