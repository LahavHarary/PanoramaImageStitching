import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
from PIL import Image, ImageEnhance
import stitcherUtils

"""

1.Import 2 images
2.convert to gray scale
3.Initiate ORB detector
4.Find key points and describe them
5.Match keypoints- Brute force matcher
6.RANSAC(reject bad keypoints)
7. Register two images (use homography) 

"""

"""
    Stitcher:

    1. Detect distinctive keypoints:
     use ORB detector

    2. Match the points between two images:
    Descriptors are arrays of numbers that define keypoints. 
    In order to match we will compare descriptors from the first with descriptors from the second image.
    we will sort matching points by their distance in order to find the closest ones. 
    # CHECK
    We will use Brute Force matcher to match (knnMatch). 

    3. Keep the strong matches that are located in the overlapping region.
    We will use David Lowe’s ratio test, Lowe proposed this test in order to increase the robustness of the algo
    the goal is to get rid of the points that are not distinct enough.
    We will discard the matches where the ratio of the distance to the nearest and the second nearest 
    neighbor is greater then a threshold (that way we will preserve only good matches).

    4. Stitch the image:
    we need to apply a technique called "feature-based image alignment" it is the computation of 2D and 3D 
    transformations
    that map features in one image to another.
    This technique consists of two steps:
    a. to apply RANSAC algorithm to evaluate a homography matrix.
    b. use the matrix to calculate the warping transformation based on matched feature

    ***HOMOGRAPHY MATRIX EXPLANATION BELOW***



"""

"""
***HOMOGRAPHY MATRIX EXPLANATION***
    in the field of computer vision, any two images of the same scene are related by a homography.
    it is a transformation that maps the points in one image to the corresponding points in the other image.
    the two images can lay on the same surface in space or they are taken by rotating the camera along its optical axis.

    The essence of the homography is the simple 3X3 matrix called the homography matrix
    H =[ [h11,h12,h13]
         [h21,h22,h23]
         [h31,h32,h33] ]

    we can apply this matrix to any point in the image.
    For example if we take a point A(x1,y1) in the first image we can use a homography matrix to map this point
    A to the corresponding point B(x2,y2) in the second image.

    s [x'] = H [x] = [h11,h12,h13] [x]
      [y']     [y]   [h21,h22,h23] [y]
      [ 1]     [1]   [h31,h32,h33] [1]

    Now, using this technique we can easily stitch our images together.

    It is important to note that when we match feature points between two images we only accept those 
    matches that fall on the corresponding epipolar lines.
    We need these good matches to estimate the homography matrix. We detected a large number of keypoints and we 
    need to reject some of them to retain the best ones.

"""

def stitchImages(img1, img2):

    # get keypoints and descriptors using orb detector
    keypoints1,keypoints2,descriptors1,descriptors2 = getKeyPointsAndDescriptors(img1,img2)

    # Finding the best matches using David Lowe’s ratio test
    bestMatches = getBestMatches(descriptors1,descriptors2)

    # Minimum amount of matches in order to stitch the images.
    minMatchCount = 4

    # Check if bestMatches has the minimum amount of keypoints required for stitching.
    if len(bestMatches) > minMatchCount:

        # compute homography and then use it in order to warp the images together
        homography = computeHomography(keypoints1,keypoints2,bestMatches)
        result = warpImages(img2, img1, homography)
        return result

def warpImages(img1, img2, homography):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
    temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, homography)

    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    # Ravel is a np function that takes a 2 or more dim array and changes it to flattend array.
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel())
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel())

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]],
                              [0, 1, translation_dist[1]],
                              [0, 0, 1]])

    # x_max - x_min, y_max - y_min = size of the image
    output_img = cv2.warpPerspective(img2, H_translation.dot(homography), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img

def getKeyPointsAndDescriptors(img1,img2):
    # Convert images to gray scale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Create orb detector and detect keypoint and descriptors
    orb = cv2.ORB_create(nfeatures=2000)

    # Find keypoints and descriptors with orb
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    return keypoints1,keypoints2,descriptors1,descriptors2

def getBestMatches(descriptors1, descriptors2, k=2):
    # Create instance of  Brute Force Matcher (BFMatcher), it will find all the matching keypoints on two images.
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # Find matching points using knn algorithm which finds the keypoints
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Finding the best matches using David Lowe’s ratio test
    bestMatches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            bestMatches.append(m)

    return bestMatches

def computeHomography(keypoints1,keypoints2,bestMatches):
    # Convert keypoints to an argument for findHomography.
    # match.queryIdx give the index of the descriptor in the list of query descriptors
    # the list of the query descriptors is of the image we would like to spread
    # the others is the training descriptors for training, this is the routine before homography

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in bestMatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in bestMatches]).reshape(-1, 1, 2)

    # Establish a homography
    # we will use RANSAC to reject "bad keypoints"
    homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    return homography
