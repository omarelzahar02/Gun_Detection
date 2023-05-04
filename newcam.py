# USAGE
# python main.py --video ex.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import math
import serial
import time
import cv2
import cvzone

# construct the argument parse and parse the arguments
a2 = 0
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
port="COM11" #This will be different for various devices and on windows it will probably be a COM port.
ser = serial.Serial()
ser.baudrate = 9600#the baud rate over which the arduino and python will communicate
ser.port = 'COM5' # change it for your owm com port
ser.open()

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
pts = deque([], maxlen=64)
# if a video path was not supplied, grab the reference
# to the webcam

camera = cv2.VideoCapture(0)
cv2.namedWindow('image')
cv2.createTrackbar('h lower', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('h upper', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('s lower', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('s upper', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('v lower', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('v upper', 'image', 0, 255, lambda x: None)

cv2.createTrackbar('h2 lower', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('h2 upper', 'image', 0, 255, lambda x: None)
# otherwise, grab a reference to the video file

# keep looping
while True:
    # grab the current frame
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    # resize the frame, blur it, and convert it to the HSV
    # color space
    # frame = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # 179,152,169
    #27,75,253
    RedLower = (15,65,240)
    RedUpper = (40,255,255)
    RedLower2 = (172,134,138)
    RedUpper2 = (240,255,255)
    mask1 = cv2.inRange(hsv, RedLower, RedUpper)
    mask2=cv2.inRange(hsv,RedLower2,RedUpper2)
    mask=mask1
    cv2.imshow("before erode", mask)
    mask = cv2.erode(mask, None, iterations=2)
    # cv2.imshow("after erode",mask)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("after dilate", mask)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    imgContour,contours= cvzone.findContours(frame,mask,minArea=10)
    center = None

    # only proceed if at least one contour was found
    if contours:
        x= contours[0]['center'][0]
        y=contours[0]['center'][1]
        area=contours[0]['area']
        area=area/3.14
        area=np.sqrt(area)
        cv2.circle(frame,(x,y),(int)(area),(0, 255, 255), 2)
        # only proceed if the radius meets a minimum size

        ############################################################################
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        radii = []
        radii.append(float(radius))
        avg_radius = np.average(radii)
        print(avg_radius)
        c1 = bytearray()
        # try:
        a = x / frame.shape[1] * 1.75
        a1 = math.ceil(math.degrees(math.atan(a))) + 70
        # except ZeroDivisionError:
        #   a1 = 90
        # if (x > 300):
        #   a1 = 180 - a1
        c1.append(a1)
        ser.write(str(a1).encode() + '\n'.encode())  # write the angle values on the X axis servo
        ser.write('a'.encode() + '\n'.encode())
        #print(a1)
        b = y / frame.shape[0]
        a2 = math.ceil(math.degrees(math.atan(b))) + 45
        ser.write(str(a2).encode() + '\n'.encode())  # write the angle values on the Y axis servo
        ser.write('b'.encode() + '\n'.encode())  # write 'b' to distinguish the angle value for Y axis only
        # input_data=bluetooth.readline()#This reads the incoming data. In this particular example it will be the "Hello from Blue" line
        # print(input_data.decode())#These are bytes coming in so a decode is needed
        time.sleep(0.1)  # A pause between bursts
        M = cv2.moments(c)
        ############################################################################

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue

        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
