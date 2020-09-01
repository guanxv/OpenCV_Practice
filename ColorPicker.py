import cv2
from stackimg import stackImages
import numpy as np

# setup the camera
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 130)

# prepar track bar function
def empty(a):
    pass


# setup track bar window
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 320)

# prepare track bar
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 179, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 223, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 40, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)

while True:
    success, img = cap.read()
    
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # get track bar value
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # set up the mask
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    # stack all images
    imgStack = stackImages(0.8, ([img, imgHSV], [mask, imgResult]))
    
    #cv2.imshow("Result", img)
    cv2.imshow("Stacked image", imgStack)

    # setup break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
