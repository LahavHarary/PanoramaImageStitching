import cv2
from extract import *
from stitcher import *


# Read the video from specified path
#camera = cv2.VideoCapture("C:\\Users\\lahav\\PycharmProjects\\PanoramaImageStitching\\videoItaly.mp4")
#videoToFrames(camera, 0, 8, str(input('Enter name for folder ')))

# Load our images
img1 = cv2.imread("frames/folder/frame0.jpg")
img2 = cv2.imread("frames/folder/frame1.jpg")

stitchImages(img1,img2)

