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
    max_time = 5000  # max time that we are going to process
    # Make sure max_time is at most 60000, or 1 minute
    max_time = min(60000, max_time)
    print(f"Max time is {max_time/1000} seconds.")

    while cap.isOpened():
        # Press a certain button to go to the next frame

        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_PLAIN

        if ret:
            if between(cap, 0, 10000):
                pass
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Add subtitle
            subtitle = "AAAAAAAA"
            cv2.putText(frame, subtitle, (int(0.5*frame_width), int(0.1*frame_height)), font, 2, (0, 0, 255), 2, cv2.LINE_4)

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            if show:
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
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
