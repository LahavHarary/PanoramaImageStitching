import cv2
from matplotlib import pyplot as plt

# Every util function that might be needed

def showImages(img):
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

def drawMatches(img1, keypoints1, img2, keypoints2, bestMatches):
    imgWithMatches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, bestMatches, None)
    showImages(imgWithMatches)

def drawKeypoints(img, keypoints):
    imgWithKeypoints = cv2.drawKeypoints(img, keypoints, None, flags=None)
    showImages(imgWithKeypoints)