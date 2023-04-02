"""
Code for the carte blanche part of the video submitted for individual assignment 1 of Computer Vision course.
By Thibeau Wouters
"""
import argparse
import math
import cv2
import os
import sys
import numpy as np

# Specify hyperparameters that are helpful when working on the videos
SHOW_FRAMES = True  # show the frames when running this script
MIN_TIME = 0  # start time of video to be processed
MAX_TIME = 20000  # max time that we are going to process
# Make sure max time is at most 20000 for this second part of the video
MAX_TIME = min(20000, MAX_TIME)


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


def dist(x1, y1, x2, y2):
    """
    Computes Euclidean distance between two points
    :param x1: x-coordinate first point
    :param y1: y-coordinate first point
    :param x2: x-coordinate second point
    :param y2: y-coordinate second point
    :return:
    """
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # Note: we take the transpose
    frame_width, frame_height = frame_height, frame_width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Hyperparameters for detections:
    # Define threshold for areas of dice in our video
    threshold_area_dice = 400
    # Threshold value for getting binary frames:
    threshold_value = 5

    # Create a SIFT object for object recognition
    sift = cv2.SIFT_create()

    # Read in templates of the six faces of the dice and process them
    templates, gray_templates = [], []
    widths, heights = [], []
    sift_templates = []
    for i in range(1, 7):
        # Read in template and get grayscale version
        name = f"template_{i}.jpg"
        template = cv2.imread(name)
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        # Compute features and descriptors of the template
        kp, des = sift.detectAndCompute(gray_template, None)
        # Save all info into lists
        templates.append(template)
        gray_templates.append(gray_template)
        sift_templates.append((kp, des))
        w, h = gray_template.shape[::-1]
        widths.append(w)
        heights.append(h)

    print(f"Processing frames between {MIN_TIME/1000} seconds and {MAX_TIME/1000} seconds.")
    font = cv2.FONT_HERSHEY_PLAIN

    while cap.isOpened():
        ret, frame = cap.read()
        # Initialize our subtitles
        subtitle = ""
        explanation_subtitle = ""
        # Rotate each frame to have the same dimensions as first part video
        frame = cv2.transpose(frame)
        if ret:
            cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Frame', (frame_width, frame_height))

            # If before min time, we don't handle those frames (used when constructing video)
            if between(cap, 0, MIN_TIME):
                continue

            # Set minimum and max HSV values for dices
            lower = np.array([55, 2, 180])
            upper = np.array([130, 237, 255])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)

            # Detect the circle of Yahtzee 'arena' (green area)
            lower_circle = np.array([36, 80, 0])
            upper_circle = np.array([154, 255, 198])

            mask_circle = cv2.inRange(hsv, lower_circle, upper_circle)
            output_circle = cv2.bitwise_and(frame, frame, mask=mask_circle)

            # Detect the circles with Hough transform
            gray_circle = cv2.cvtColor(output_circle, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray_circle, cv2.HOUGH_GRADIENT, 1, minDist=frame_width, param1=20, param2=20,
                                       minRadius=200, maxRadius=600)

            # In case no circles detected, estimate circle around 350 and center = center frame
            center_x, center_y = frame_width//2, frame_height//2
            radius = 350
            # Draw the circle on the original frame (if detected)
            if circles is not None:
                # Get first circle, skip others
                circles = np.uint16(np.around(circles))
                circle = circles[0, 0]
                radius = circle[2]
                center_x, center_y = circle[0], circle[1]

            # Now, get the contours for dices
            ret, thresh = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
            # Find contours in the binary image
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Loop through each contour and filter out the dices
            dice_x, dice_y, dice_widths, dice_heights = [], [], [], []
            for contour in contours:
                # Calculate the area of the contour
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                if dist(center_x, center_y, x, y) < radius and area > threshold_area_dice:
                    # This contour is a dice, save it
                    dice_x.append(x)
                    dice_y.append(y)
                    dice_widths.append(w)
                    dice_heights.append(h)

            # Next, we detect the face of the dice. Initialize list where we will save these as labels
            dice_labels = [0 for _ in range(len(dice_x))]

            "40 -- 50 s: Detect the face using feature descriptors"
            if between(cap, 0, 10000):
                # Iterate over all the dices:
                subtitle = "Detecting faces of dice"
                explanation_subtitle = "Using SIFT feature descriptors (results are not so reliable)"

                for i in range(len(dice_x)):
                    # Initialize variables
                    best_label = "???"
                    best_num_matches = 0
                    # Extract the dice's image
                    dice_image = frame[dice_y[i]:dice_y[i]+dice_heights[i], dice_x[i]:dice_x[i]+dice_widths[i]]
                    # Convert to gray scale
                    gray = cv2.cvtColor(dice_image, cv2.COLOR_BGR2GRAY)
                    # Find the keypoints and descriptors of this dice with SIFT
                    kp, des = sift.detectAndCompute(gray, None)

                    for j, (kp_template, des_template) in enumerate(sift_templates):
                        # BFMatcher with default params
                        bf = cv2.BFMatcher()
                        matches = bf.knnMatch(des, des_template, k=2)
                        # Apply ratio test to determine good matches
                        good = []
                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                good.append([m])
                        num_matches = len(good)
                        # See if we improved dice face estimate
                        if num_matches > best_num_matches:
                            best_label = str(j+1)
                            best_num_matches = num_matches
                    # At the end, save best label
                    dice_labels[i] = best_label

            "50 -- 60 s: Detect the face by detecting contours"
            if between(cap, 10000, 20000):
                subtitle = "Detecting faces of dice"
                explanation_subtitle = "Using contour detections (robust results)"

                # Now, count the number of contours within each dice contour
                for contour in contours:
                    # Calculate the area and get specifications
                    area = cv2.contourArea(contour)
                    x, y, w, h = cv2.boundingRect(contour)
                    # Check if it is located inside the game area and a "small contour"
                    if dist(center_x, center_y, x, y) < radius and threshold_area_dice > area > 60:
                        # This is a dot on a dice face, draw it:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # Find in which dice this contour is located. Compute distance to all dice centers
                        distances = [
                            dist(dice_x[i] + dice_widths[i] / 2, dice_y[i] + dice_heights[i] / 2, x + w / 2, y + h / 2)
                            for i in range(len(dice_x))]
                        # Increment counter of closest dice center
                        if len(distances) == 0:
                            continue
                        else:
                            index = np.argmin(distances)
                            dice_labels[index] += 1

            # Draw all the annotations that we gathered:
            # Draw outer circle of Yahtzee 'arena'
            cv2.circle(frame, (center_x, center_y), radius, (0, 0, 255), 2)
            # Draw inner circle of Yahtzee 'arena'
            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 3)
            # Draw rectangles around dice, and show their labels
            for i in range(len(dice_x)):
                cv2.rectangle(frame, (dice_x[i], dice_y[i]),
                          (dice_x[i] + dice_widths[i], dice_y[i] + dice_heights[i]),
                          (0, 0, 255), 2)
                cv2.putText(frame, str(dice_labels[i]), (dice_x[i], dice_y[i] - 10), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 255), 4)

            # Write subtitle, with a timestamp, write frame to output
            timestamp = '{:.3f}'.format((40000+get_time(cap))/1000)
            cv2.putText(frame, subtitle + f" ({timestamp} s)", (int(0.05*frame_width), int(0.05*frame_height)), font, 2, (0, 0, 255), 2, cv2.LINE_4)
            cv2.putText(frame, explanation_subtitle, (int(0.05 * frame_width), int(0.08 * frame_height)), font,
                        2, (0, 0, 255), 2, cv2.LINE_4)
            out.write(frame)

            # (optional) display the resulting frame
            if SHOW_FRAMES:
                cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)

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

    input_file  = os.path.normpath("D:\\Coding\\MAI\\CV\\Assignment_1\\dice.MOV")
    output_file = os.path.normpath("D:\\Coding\\MAI\\CV\\Assignment_1\\dice_out.mp4")

    # Then, run the above code
    main(input_file, output_file)
