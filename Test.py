# -#-#-#-#- test - 1 #-#-#-#-#-#-#-#-

# import cv2
#
# img = cv2.imread('Resources/HPIM1217.JPG')
#
# cv2.imshow("Output", img)
# cv2.waitKey(0)

# -#-#-#-#- test - 2 #-#-#-#-#-#-#-#-

# vid = cv2.VideoCapture('Resources/HPIM2657.mpg')

"""
while True:

    success, img = vid.read()
    cv2.imshow('Video', img)

    if cv2.waitKey(50) & 0XFF == ord('q'):
        break
"""

# -#-#-#-#- test - 3 #-#-#-#-#-#-#-#-
"""
vid = cv2.VideoCapture(0)
vid.set(3,640)
vid.set(4,480)
vid.set(10,100)

while True:

    success, img = vid.read()
    cv2.imshow('Video', img)

    if cv2.waitKey(50) & 0XFF == ord('q'):
        break
        """

# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-
"""
import numpy as np

img = cv2.imread('Resources/HPIM1217.JPG')

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3,3),2)
imgCanny = cv2.Canny(img, 100, 100)

kernel = np.ones((2,2),np.uint8)
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1 )
imgEroded = cv2.erode(imgDialation, kernel, iterations= 1)

cv2.imshow("Gray Image", imgBlur)
cv2.imshow("Blur Image", imgBlur)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dialation Image", imgDialation)
cv2.imshow("Eroded Image", imgEroded)
cv2.waitKey(0)
"""
# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-
"""
import cv2

img = cv2.imread("Resources/HPIM1217.JPG")

imgResize = cv2.resize(img, (640,480))
imgReResize = cv2.resize(imgResize, (1280,960))
imgCropped = img[0:1000, 0:500] # only use matrix function to crop the image, Height come first

cv2.imshow("image", img)
cv2.imshow("imageResize", imgResize)
cv2.imshow("imageReResize", imgReResize)
cv2.imshow('imgCropped', imgCropped)


print(img.shape)
print(imgResize.shape)


cv2.waitKey(0)

"""

# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-
"""
import cv2
import numpy as np

#img = np.ones((512,512))
img_gray = np.zeros((512,512))
#print(img_gray)

img_bgr = np.zeros((512,512,3),np.uint8)
#print(img_bgr)

#img_bgr[:] = 255,0,0 # change whole pic to blue
#img_bgr[200:300 , 400: 500] = 255,0,0 #change area 200 :300 height to 400 : 500 Weidth to blue

#cv2.line(img_bgr, (0,0), (300,300), (0,255,0),3)
#cv2.line(img_bgr, (0,0), (img_bgr.shape[1],img_bgr.shape[0]), (0,255,0),3) # draw a diagnal line

#cv2.rectangle(img_bgr,(0,0),(200,200),(255,255,0),3)
#cv2.rectangle(img_bgr,(0,0),(200,200),(255,255,0),cv2.FILLED)
#cv2.circle(img_bgr, (225,225),50,(0,255,255),3)

cv2.putText(img_bgr,"Test Test",(0,220),cv2.FONT_ITALIC,1,(0,150,500),3)

cv2.imshow("zeros_gray", img_gray)
cv2.imshow("zeros_bgr", img_bgr)
cv2.waitKey(0)
"""
# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-
"""
import cv2
import numpy as np

img = cv2.imread("Resources/HPIM1265.JPG")

width, height = 350, 350

pts1 = np.float32([[1468,129],[1996,2],[1656,378],[2121,283]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgOutput = cv2.warpPerspective(img,matrix,(width,height))

cv2.imshow('image', img)
cv2.imshow('Output', imgOutput)
cv2.waitKey(0)
"""
# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-
"""
import cv2
import numpy as np

img1 = cv2.imread('Resources/HPIM1217.JPG')
img2 = cv2.imread('Resources/HPIM1265.JPG')

img1_resize = cv2.resize(img1, (640,480))
img2_resize = cv2.resize(img2, (640,480))

hor = np.hstack((img1_resize,img2_resize))
ver = np.vstack((img1_resize,img2_resize))

print(img1_resize)

cv2.imshow("Hori Stack", hor)
cv2.imshow("Veri Stack", ver)
cv2.waitKey(0)

# source picture need to be same channels to be able to merge.
# can not resize the image.
"""
# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-
"""
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y], (0, 0), None, scale, scale
                    )
                else:
                    imgArray[x][y] = cv2.resize(
                        imgArray[x][y],
                        (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                        None,
                        scale,
                        scale,
                    )
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(
                    imgArray[x],
                    (imgArray[0].shape[1], imgArray[0].shape[0]),
                    None,
                    scale,
                    scale,
                )
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
"""

"""

import cv2
import numpy as np

img1 = cv2.imread('Resources/HPIM1217.JPG')
img2 = cv2.imread('Resources/HPIM1265.JPG')

img1_resize = cv2.resize(img1, (640,480))
img2_resize = cv2.resize(img2, (640,480))

imgStack = stackImages(1, ([img1_resize,img2_resize]))  #horizontal Stack
imgStack = stackImages(1, ([img1_resize],[img2_resize])) #vertical stack

# this function can handle img in different channel, and size.

cv2.imshow("image Stack", imgStack)
cv2.waitKey(0)
"""

# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-
"""
import cv2
import numpy as np

img = cv2.imread('Resources/HPIM1265.JPG')
img = cv2.resize(img, (640,480))
imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#prepar track bar
def empty(a):
    pass

cv2.namedWindow('TrackBars')
cv2.resizeWindow("TrackBars", 640 , 320)

#prepare track bar
cv2.createTrackbar("Hue Min", "TrackBars",0,179,empty)
cv2.createTrackbar("Hue Max", "TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min", "TrackBars",0,255,empty)
cv2.createTrackbar("Sat Max", "TrackBars",223,255,empty)
cv2.createTrackbar("Val Min", "TrackBars",40,255,empty)
cv2.createTrackbar("Val Max", "TrackBars",255,255,empty)

while True:
    # get track bar value
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    # set up the mask
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask = mask)
    # stack all images
    imgStack = stackImages(0.8, ([img, imgHSV],[mask, imgResult]))

    # show image
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    cv2.imshow("Stacked Image", imgStack)
    cv2.waitKey(1)
"""
# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-

"""
import cv2
import numpy as np


def getContours(img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 800:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 1)
            cv2.drawContours(imgBlank, cnt, -1, (255, 255, 0), 2, None, None, 1)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            print(area, peri, len(approx))
            objCor = len(approx)
            print(type(objCor))
            x, y, w, h = cv2.boundingRect(approx)
            objType = ""
            if objCor == 3:
                objType = "Tri"
            elif objCor == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.95 and aspRatio < 1.05:
                    objType = "Square"
                else:
                    objType = "Rect"
            elif objCor > 4:
                objType = "Circle"

            cv2.rectangle(
                imgContour, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 1
            )
            cv2.putText(
                imgContour,
                objType,
                ((x + (w // 2) - 10), (y + (h // 2)) + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 0),
            )


path = "Resources/"
file = "shapes.jpg"

img = cv2.imread(path + file)
imgContour = img.copy()

imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 4)
imgBlank = np.zeros_like(img)

# prepare for the sliding bar


def empty(a):
    pass


cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 320)
cv2.createTrackbar("Canny Min", "TrackBars", 80, 300, empty)
cv2.createTrackbar("Canny Max", "TrackBars", 150, 300, empty)


while True:
    # get track bar value
    imgBlank = np.zeros_like(img)
    canny_min = cv2.getTrackbarPos("Canny Min", "TrackBars")
    canny_max = cv2.getTrackbarPos("Canny Max", "TrackBars")
    imgCanny = cv2.Canny(imgBlur, canny_min, canny_max)
    # print(canny_min, canny_max)

    getContours(imgCanny)
    imgStacked = stackImages(
        1, ([img, imgGray, imgBlur], [imgCanny, imgContour, imgBlank])
    )
    # cv2.imshow("Image", img)
    cv2.imshow("Stacked Image", imgStacked)
    cv2.waitKey(1)

"""
# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-
"""
import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier(
    "venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
)

original = cv2.imread('Resources/IMG_1355.jpg')
# print(original.shape)

img = cv2.resize(original,None, fx = 0.25 , fy= 0.25 , interpolation=cv2.INTER_AREA)
#img = cv2.imread("Resources/IMG_1355.jpg")
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(imgGray, 1.02, 1, None, (2, 2), (600, 600))

# scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
# Basically the scale factor is used to create your scale pyramid. More explanation can be found here. In short, as described here, your model has a fixed size defined during training, which is visible in the xml. This means that this size of face is detected in the image if present. However, by rescaling the input image, you can resize a larger face to a smaller one, making it detectable by the algorithm.
# 1.05 is a good possible value for this, which means you use a small step for resizing, i.e. reduce size by 5%, you increase the chance of a matching size with the model for detection is found. This also means that the algorithm works slower since it is more thorough. You may increase it to as much as 1.4 for faster detection, with the risk of missing some faces altogether.

# minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
# This parameter will affect the quality of the detected faces. Higher value results in less detections but with higher quality. 3~6 is a good value for it.

# minSize – Minimum possible object size. Objects smaller than that are ignored.Usually, [30, 30] is a good start for face detection.

# maxSize – Maximum possible object size. Objects bigger than this are ignored.

print(type(faces))
print(faces)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)


cv2.imshow("img", img)
cv2.waitKey(0)
"""
# -#-#-#-#- test - 4 #-#-#-#-#-#-#-#-

import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 130)

myColors = [[0, 104, 164, 13, 218, 255]]


def findColor(img, myColors):

    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array(myColors[0][0:3])
    upper = np.array(myColors[0][3:])
    mask = cv2.inRange(imgHSV, lower, upper)
    getContours(mask)
    # cv2.imshow("img", mask)


def getContours(img):
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > 5:
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)


while True:
    success, img = cap.read()
    imgResult = img.copy()
    findColor(img, myColors)

    #cv2.imshow("WebCam", img)
    cv2.imshow("Result", imgResult)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

