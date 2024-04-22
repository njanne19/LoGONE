import cv2
import numpy as np

def resize_to_match(image1, image2, max_dimensions=(1000, 1000)):
    # We need to scale the smallest image to be the same size as the largest
    scale1 = max(max_dimensions[0] / image1.shape[1], max_dimensions[1] / image1.shape[0])
    scale2 = max(max_dimensions[0] / image2.shape[1], max_dimensions[1] / image2.shape[0])

    if scale1 < 1:
        image1 = cv2.resize(image1, None, fx=scale1, fy=scale1, interpolation=cv2.INTER_AREA)
    if scale2 < 1:
        image2 = cv2.resize(image2, None, fx=scale2, fy=scale2, interpolation=cv2.INTER_AREA)
    
    return image1, image2

def align_images(image1, image2):
    image1, image2 = resize_to_match(image1, image2)

    #SIFT works better with greyscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    try:
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    except:
        return None

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    try:
        homography, _ = cv2.findHomography(points1, points2, cv2.RANSAC)
    except:
        return None

    return homography