# USAGE
# python main.py --video ex.mp4
# python ball_tracking.py

# import the necessary packages
import math
import serial
import time
from collections import deque
from numpy import sqrt
import argparse
import imutils
import cv2

a2 = 0
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
port="COM11" #This will be different for various devices and on windows it will probably be a COM port.
ser = serial.Serial()
ser.baudrate = 9600#the baud rate over which the arduino and python will communicate
ser.port = 'COM3' # change it for your owm com port
ser.open()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
#greenLower = (29, 86, 6)
#greenUpper = (64, 255, 255)
#greenLower = (26, 139, 132)
#greenUpper = (191, 215, 255)
greenLower = (109, 49, 144)
greenUpper = (255, 255, 255)

cv2.namedWindow('image')
cv2.createTrackbar('h lower', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('h upper', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('s lower', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('s upper', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('v lower', 'image', 0, 255, lambda x: None)
cv2.createTrackbar('v upper', 'image', 0, 255, lambda x: None)
pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
# otherwise, grab a reference to the video file
else:
    camera = cv2.VideoCapture(args["video"])
# keep looping
while True:
    # grab the current frame
    (grabbed, frameinv) = camera.read()
    frame = cv2.flip(frameinv, 1)
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    greenLower = (146, 43, 54)
    greenUpper = (196, 165, 128)

    if args.get("video") and not grabbed:

        break
    # resize the frame, blur it, and convert it to the HSV
    # color space
    frame = imutils.resize(frame, width=600)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, greenLower, greenUpper)
    cv2.imshow('detect',mask)
    mask = cv2.erode(mask, None, iterations=3)    ## iteration = 3 --------- 2 when testing
    mask = cv2.dilate(mask, None, iterations=4)
    cv2.imshow('nonoise', mask)
    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        c1 = bytearray()
        # try:
        a = x/frame.shape[1] * 1.75
        a1 = math.ceil(math.degrees(math.atan(a / 1.75))) + 70
        # except ZeroDivisionError:
        #   a1 = 90
        # if (x > 300):
        #   a1 = 180 - a1
        c1.append(a1)
        ser.write(str(a1).encode() + '\n'.encode())  # write the angle values on the X axis servo
        ser.write('a'.encode() + '\n'.encode())
        #print(a1)
        b = y/frame.shape[0]
        a2 = math.ceil(math.degrees(math.atan(b))) + 45
        ser.write(str(a2).encode() + '\n'.encode())  # write the angle values on the Y axis servo
        ser.write('b'.encode() + '\n'.encode())  # write 'b' to distinguish the angle value for Y axis only
        # input_data=bluetooth.readline()#This reads the incoming data. In this particular example it will be the "Hello from Blue" line
	    #print(input_data.decode())#These are bytes coming in so a decode is needed
        time.sleep(0.1) #A pause between bursts
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 0.5:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(frame, (int(x), int(y)), int(radius),
                       (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
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
        thickness = int(sqrt(args["buffer"] / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
    # show the frame to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
# bluetooth.close()
cv2.destroyAllWindows()
