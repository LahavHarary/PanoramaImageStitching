import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import imutils

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
    We will discard the matches where the ratio of the distance to the nearest and the second nearest neighbor is greater
    then a threshold (that way we will preserve only good matches).
     
    4. Stitch the image:
    we need to apply a technique called "feature-based image alignment" it is the computation of 2D and 3D transformations
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

def stitchImages():
  # Load our images
  img1 = cv2.imread("frames/folder/frame0.jpg")
  img2 = cv2.imread("frames/folder/frame1.jpg")

  # Change images to gray
  img1_gray = cv2.cvtColor(img1.copy(), cv2.COLOR_BGR2GRAY)
  img2_gray = cv2.cvtColor(img2.copy(), cv2.COLOR_BGR2GRAY)

  """
  # Check if images are ok
  #cv2.imshow("first image",img1_gray)
  #cv2.imshow("second image",img2_gray)
  #cv2.waitKey(0)
  #cv2.destroyAllWindows();
  """

  # Create orb detector and detect keypoint and descriptors
  orb = cv2.ORB_create(nfeatures=2000)

  # Find keypoints and descriptors with orb
  keypoints1 , descriptors1 = orb.detectAndCompute(img1_gray,None)
  keypoints2 , descriptors2 = orb.detectAndCompute(img2_gray,None)

  """
  # Check that keypoints exists
  
  cv2.imshow("Image with keypoints",cv2.drawKeypoints(img1,keypoint1,None,(255,0,255)))
  cv2.waitKey(0)
  cv2.destroyAllWindows();
  """

  # Create BFMatcher obejct, it will find all the matching keypoints on two images.
  # NORM_HAMMING is a normType that specifies the distance as a measurement of similarity
  bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

  #Find matching poitns
  matches = bf.knnMatch(descriptors1, descriptors2,k=2)

  """
  # Check for keypoints and Descriptors

  print(keypoints1[0].pt)
  print(keypoints1[0].size)
  print("Descriptor of the first keypoint: ")
  print(descriptors2[0])
  """

  """
  # Show matches:

  img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
  img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
  matches = sorted(matches, key = lambda x:x.distance)
  img3 = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:20], None ,flags=2)
  plt.imshow(img3)
  plt.show()


  """

  """
  # Show the image with matches
  
  img3 = draw_matches(img1_gray,keypoints1,img2_gray,keypoints2,all_matches[:30])
  cv2.imshow("image",cv2.resize(img3,dsize=(1200,1200),interpolation=cv2.INTER_CUBIC))
  cv2.waitKey(0)
  """

  # Finding the best matches
  good = []
  for m, n in matches:
      if m.distance < 0.6 * n.distance:
        good.append(m)

  # Set minimum match condition
  MIN_MATCH_COUNT = 10

  if len(good) > MIN_MATCH_COUNT:
    # Convert keypoints to an argument for findHomography
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Establish a homography
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    result = warpImages(img2, img1, M)
    #result = trim(result)

    plt.figure()
    plt.imshow(result[:,:,::-1])
    plt.axis('off')
    plt.show()

    cv2.imshow("result" ,result)
    cv2.waitKey(0)

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
  r, c = img1.shape[:2]
  r1,c1 = img2.shape[:2]

  # Create a blank image with the size of the first image + second image
  output_img = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
  output_img[:r,:c,:] = np.dstack([img1,img1,img1])
  output_img[:r1,c:c+c1,:] = np.dstack([img2,img2,img2])

  # Go over all of the matching points and extract them for match in matches:

  for match in matches:
    img1_idx = match.queryIdx
    img2_idx = match.queryIdx
    (x1,y1) = keypoints1[img1_idx].pt
    (x2,y2) = keypoints2[img2_idx].pt

    #Draw circles on the keypoints
    cv2.circle(output_img, (int(x1),int(y1)),4,(0,255,255),1)
    cv2.circle(output_img, (int(x2) + c, int(y2)), 4, (0, 255, 255), 1)

    # Connect the same keypoints
    cv2.line(output_img,(int(x1),int(y1)),(int(x2) + c ,int(y2)),(0,255,255),1)

  return output_img;

def warpImages(img1, img2, H):
  rows1, cols1 = img1.shape[:2]
  rows2, cols2 = img2.shape[:2]

  list_of_points_1 = np.float32([[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
  temp_points = np.float32([[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

  # When we have established a homography we need to warp perspective
  # Change field of view
  list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

  list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

  [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
  [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

  translation_dist = [-x_min, -y_min]

  H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

  output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
  output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

  return output_img

def trim(stitched):
  # convert to grayscale
  gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)

  # The shit that is around our picture is now in thresh
  print("[INFO] cropping...")
  stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))
  # convert the stitched image to grayscale and threshold it
  # such that all pixels greater than zero are set to 255
  # (foreground) while all others remain 0 (background)
  gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]


  # Max size of rectangle
  cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                          cv2.CHAIN_APPROX_SIMPLE)
  cnts = imutils.grab_contours(cnts)
  c = max(cnts, key=cv2.contourArea)
  # allocate memory for the mask which will contain the
  # rectangular bounding box of the stitched image region
  mask = np.zeros(thresh.shape, dtype="uint8")
  (x, y, w, h) = cv2.boundingRect(c)
  cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

  # create two copies of the mask: one to serve as our actual
  # minimum rectangular region and another to serve as a counter
  # for how many pixels need to be removed to form the minimum
  # rectangular region
  minRect = mask.copy()
  sub = mask.copy()
  # keep looping until there are no non-zero pixels left in the
  # subtracted image
  while cv2.countNonZero(sub) > 0:
    # erode the minimum rectangular mask and then subtract
    # the thresholded image from the minimum rectangular mask
    # so we can count if there are any non-zero pixels left
    minRect = cv2.erode(minRect, None)
    sub = cv2.subtract(minRect, thresh)

    # find contours in the minimum rectangular mask and then
    # extract the bounding box (x, y)-coordinates
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    # use the bounding box coordinates to extract the our final
    # stitched image

    #stitched = stitched[y:y + h, x:x + w]

  return stitched