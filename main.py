# USAGE
# python main.py --video ex.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np
import argparse
import imutils
import cv2
import cvzone
import math
import serial

# construct the argument parse and parse the arguments

camera = cv2.VideoCapture(0)
ser = serial.Serial()
ser.baudrate = 250000#the baud rate over which the arduino and python will communicate
ser.port = 'COM5' # change it for your owm com port
ser.open()
pre_servo_lower=0
pre_servo_upper=0
pre_servo_lower1=0
pre_servo_upper1=0
pre_center_x=0
pre_center_y=0
start_time=0
kernel = np.ones((7,7),np.uint8)
global center_x,center_y
def nothing(x):
	pass
# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
def calibrateColor(color,def_range):
    global kernel
    greenLower = (109, 49, 144)
    greenUpper = (255, 255, 255)
    pts = deque([], maxlen=64)
    # if a video path was not supplied, grab the reference
    # to the webcam

    cv2.namedWindow('image')
    cv2.createTrackbar('h lower', 'image', 0, 255, lambda x: None)
    cv2.createTrackbar('h upper', 'image', 0, 255, lambda x: None)
    cv2.createTrackbar('s lower', 'image', 0, 255, lambda x: None)
    cv2.createTrackbar('s upper', 'image', 0, 255, lambda x: None)
    cv2.createTrackbar('v lower', 'image', 0, 255, lambda x: None)
    cv2.createTrackbar('v upper', 'image', 0, 255, lambda x: None)
    # otherwise, grab a reference to the video file

    # keep looping
    while True:
        # grab the current frame
        (grabbed, frame) = camera.read()
        greenLower=(cv2.getTrackbarPos('h lower', 'image'),cv2.getTrackbarPos('s lower', 'image'),cv2.getTrackbarPos('v lower', 'image'))
        greenUpper = (cv2.getTrackbarPos('h upper', 'image'), cv2.getTrackbarPos('s upper', 'image'),cv2.getTrackbarPos('v upper', 'image'))
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        # resize the frame, blur it, and convert it to the HSV
        # color space
        # frame = cv2.GaussianBlur(frame, (11, 11), 0)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        cv2.imshow("before erode", mask)
        mask = cv2.erode(mask, None, iterations=2)
        # cv2.imshow("after erode",mask)
        mask = cv2.dilate(mask, None, iterations=2)
        # cv2.imshow("after dilate", mask)
        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        imgContour,contours= cvzone.findContours(frame,mask,minArea=100)
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
        if key == ord(' '):
            cv2.destroyWindow('image')
            return np.array([[greenLower], [greenUpper]])

red_range = np.array([[158,85,72],[180,255,255]])
red_range = calibrateColor('red', red_range)
while(1):
    _,img=camera.read()
    img=cv2.flip(img,1)#to get a flipped image
    height,width,depth=img.shape
    img=cv2.resize(img,(height*3,int(width*1.1)))
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    kernal = np.ones((5, 5), "uint8")
    mask = cv2.inRange(hsv, red_range[0], red_range[1])
    #mask1=cv2.(hsv,blue_range)
    eroded = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(eroded, kernel, iterations=1)

    (contours,hierarchy)=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for pic,contour in enumerate(contours):
        area=cv2.contourArea(contour)
        if(area>100):
            x,y,w,h=cv2.boundingRect(contour)
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.putText(img,"",(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
            center_x=x+(w/2)        #calculate the center X coordinate of the detected object
            center_y=y+(h/2)        #calculate the center Y coordinate of the detected object
            #print(center_x)
            servo_x=math.ceil(math.degrees(math.atan(center_x/600)))# convert the pixel coordinate into angles
            if servo_x>pre_servo_lower and servo_x<pre_servo_upper :

                pass
            else :
                #print(servo_x)
                ser.write(str(servo_x).encode()+'\n'.encode())   #write the angle values on the X axis servo
                ser.write('a'.encode()+'\n'.encode())           #write 'a' to distinguish the anfle values for the X axis only
                pre_servo_lower=servo_x-3                       #takes a range of -3 to +3 pixels to reduce the number of values written
                pre_servo_upper=servo_x+3
            servo_y = math.ceil(math.degrees(math.atan(center_y/400))) # convert the pixel coordinate into angles
            if servo_y>pre_servo_lower1 and servo_y<pre_servo_upper1 :
                if (center_x - pre_center_x) ** 2 <= 50 and (center_y - pre_center_y) ** 2 <= 50:
                    start_time = start_time+1   #it counts the time for which the object remained stationary
                #print('no values written for y')
            else :
                ser.write(str(servo_y).encode()+'\n'.encode())   #write the angle values on the Y axis servo
                ser.write('b'.encode()+'\n'.encode())           # write 'b' to distinguish the angle value for Y axis only
                pre_servo_lower1=servo_y-3
                pre_servo_upper1=servo_y+3
                start_time=0;
            pre_center_x=center_x
            pre_center_y=center_y
            if start_time>=15:
                ser.write('c'.encode() + '\n'.encode())
                print("shoot")
                start_time=0
    cv2.imshow('Color Tracking',img)
    k=cv2.waitKey(50)& 0xFF
    if k==27:   #press esc key to close the window
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
