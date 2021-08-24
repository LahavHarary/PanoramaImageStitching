from extract import *
from stitcher import *
from stitcher import *
from stitcherUtils import *
from cutter import *
from enhancer import *

# Read the video from specified path
#camera = cv2.VideoCapture("C:\\Users\\rotem\\PycharmProjects\\PanoramaImageStitching\\videoItaly.mp4")
#videoToFrames(camera, 0, 8, str(input('Enter name for folder ')))


img1 = cv2.imread('frames/data/frame1.jpg')
img2 = cv2.imread('frames/data/frame2.jpg')

stitchedCuttedEnhancedImage = (stitchImages(img1, img2))
stitchedCuttedEnhancedImage = cutImage(stitchedCuttedEnhancedImage)
stitchedCuttedEnhancedImage = enhanceImage(stitchedCuttedEnhancedImage)
showImages(stitchedCuttedEnhancedImage)

#stitchImages(cv2.imread('stitchedOutputProcessed.png'),img3)

