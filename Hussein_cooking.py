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
from imutils import paths


def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image

	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)


def find_marker2(image2):
    # convert the image to grayscale, blur it, and detect edges
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
    edged2 = cv2.Canny(gray2, 35, 125)
    cnts2 = cv2.findContours(edged2.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image

    c = max(cnts2, key=cv2.contourArea)
    # compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# USAGE
# python main.py --video ex.mp4
# python ball_tracking.py

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
dist = 100.0
# keep looping
while True:
    calibrator = 0.9 #NOT = 1 less than 1
    Screen_centerx=320
    Screen_centery=240
    #mesured 320 , 249
    # grab the current frame
    (grabbed, frame) = camera.read()
    (grabbed, frame2) = camera.read()
    frame = cv2.flip(frame, 1)
    frame2 = cv2.flip(frame2, 1)
    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    # resize the frame, blur it, and convert it to the HSV
    # color space
    # frame = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # 179,152,169
    # green 33 91    58
    # blue 104 60 70
    #laser 69,31,94
    RedLower = (25, 70, 40)
    RedUpper = (50, 255, 255)
    RedLower2 = (95,50,60)
    RedUpper2 = (115,255,255)
    mask1 = cv2.inRange(hsv, RedLower, RedUpper)
    mask2=cv2.inRange(hsv2,RedLower2,RedUpper2)
    mask=mask1
    cv2.imshow("before erode", mask)
    mask = cv2.erode(mask, None, iterations=2)
    # cv2.imshow("after erode",mask)
    mask = cv2.dilate(mask, None, iterations=2)
    cv2.imshow("after dilate", mask)

    mask_blue = mask2
    cv2.imshow("before erode blue", mask_blue)
    mask_blue = cv2.erode(mask_blue, None, iterations=2)
    # cv2.imshow("after erode",mask)
    mask_blue = cv2.dilate(mask_blue, None , iterations = 2)
    cv2.imshow("after dilate blue", mask_blue)
    # (x, y) center of the ball
    imgContour,contours= cvzone.findContours(frame,mask,minArea=10)
    imgContour2, contours2 = cvzone.findContours(frame2, mask2, minArea=10)
    center = None

    KNOWN_DISTANCE = 160
    # initialize the known object width, which in this case, the piece of
    # paper is 12 inches wide
    KNOWN_WIDTH = 4.0
    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    marker = find_marker(frame)

    marker2 = find_marker2(frame2)
    focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
    #print(marker[1][0])
    #print(focalLength)

    focal = 4189.416571
    min_hypotenuse = math.sqrt((100**2)+(150**2))
    max_hypotenuse = math.sqrt((100**2)+(200**2))
    dis2 = (KNOWN_WIDTH*focal) / marker[1][0]
    if dis2 >= min_hypotenuse and dis2 <= max_hypotenuse:
        dist = dis2
    #if(dist > 200):
    #    dist = 200
    #elif (dist<150):
    #    dist = 150
    #print(dist)

    if contours2:
        x_blue = contours2[0]['center'][0]
        y_blue = contours2[0]['center'][1]
        cnts_blue = cv2.findContours(mask_blue.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
        #print(cnts_blue)
        c_blue = max(cnts_blue, key=cv2.contourArea)
        ((x_blue, y_blue), radius_blue) = cv2.minEnclosingCircle(c_blue)
        radii_blue = []
        radii_blue.append(float(radius_blue))
        avg_radius_blue = np.average(radii_blue)
        #print (f'radius of blue circle={avg_radius_blue}')

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
        #
        image_width_on_screen = avg_radius
        real_width = 13.6
        ratio = real_width/image_width_on_screen

        real_width_blue = 5.5
        ratio_blue = real_width_blue/avg_radius_blue
        ratio_romberg = (4*ratio - ratio_blue) / 3
        #print(contours2[0])
        #=-3.36952*(x)+325.284 ----A4
        #-2.47130111175*(x)+341.987533551------A3
        #the linearized equation
        #hypotunus from radius

        #print(avg_radius)
        #print(avg_radius_blue)
        c1 = bytearray()
        # try:
        #hetety ana deh
        height_of_target_irl=50
        atunator=-2.67731209107*(avg_radius)+352.948646254
        atunator_blue=-9.08432994228*(avg_radius_blue)+344.467290424
        max_hypotenuse1 = math.sqrt(height_of_target_irl**2+200**2)
        min_hypotenuse1 = math.sqrt(height_of_target_irl ** 2 + 150 ** 2)
        if atunator>max_hypotenuse1:
            atunator = max_hypotenuse1
        elif atunator<min_hypotenuse1:
            atunator = min_hypotenuse1
        if(x+y)-height_of_target_irl/ratio_romberg==0:
                x=x+0.0000001
        elif(x+y-height_of_target_irl/ratio_romberg==1):
                x=x+0.0000001
        if atunator_blue>max_hypotenuse1:
            atunator_blue = max_hypotenuse1
        elif atunator_blue<min_hypotenuse1:
            atunator_blue = min_hypotenuse1
        #print(atunator)
        #print(atunator_blue)
        atunator_romberg = (4*atunator-atunator_blue)/3
        #print(atunator_romberg)
        #hypot = 175*(1-calibrator)*(1-1/(x+y))+atunator_romberg*(calibrator)/(x+y)
        hypot = atunator_romberg
        #print(hypot)
        xreal = -(x-Screen_centerx)*ratio_romberg

        yreal = -(y-Screen_centery)*ratio_romberg
        #########################################################
        ###                 Note To self                      ###
        ###use height_of_target_irl intead of y-screen_centery###
        #########################################################
        proj = math.sqrt((dist**2)-(100**2))
        #print(proj)
        Projection = math.sqrt((hypot ** 2) - (yreal ** 2))
        if((abs(yreal)<hypot )| (abs(xreal) < Projection)):
            #BIG PROBLEM HERE SWAP SIN WITH ACOS or atleast asin
            a1 = (math.degrees(math.sin(yreal/hypot)))#vertical angle
            #y=hyp(cos(90-a1)) -----> 90-a1=acos(y/hyp)
            #print(a1)
            a2 = (math.degrees(math.asin(xreal/Projection))) #horizontal angle
            #x=hyp sin(90-a1)*cos(a2)
            #Or using projection #projection also = hyp sin(90-a1)
            #print(a2)
        else:
            a1 = 60
            a2 = 60
        #le7ad hena
        # except ZeroDivisionError:
        #   a1 = 90q
        # if (x > 300):
        #   a1 = 180 - a1
        #c1.append(a1)
        #ser.write(str(a1).encode() + '\n'.encode())  # write the angle values on the X axis servo
        #ser.write('a'.encode() + '\n'.encode())
        #print(a1)
        #b = y / frame.shape[0]

        #a2 = math.ceil(math.degrees(math.asin(100.0/dist)))
        if (a1 > 0):
            a1send = (60-math.ceil(a1))
        else:
            a1send = 60 + abs(math.ceil(a1))

        if (a2 > 0):
            a2send = 60-math.ceil(a2)
        else:
            a2send = 60+abs(math.ceil(a2))
        #print(a2send)
        #print(a1send)
        ser.write(str(a2send).encode() + '\n'.encode())  # write the angle values on the Y axis servo
        ser.write('a'.encode() + '\n'.encode())  # write 'b' to distinguish the angle value for Y axis only

        ser.write(str(a1send).encode() + '\n'.encode())  # write the angle values on the Y axis servo
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
    cv2.imshow("Frame2",frame2)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, stop the loop
    if key == ord("q"):
        break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

