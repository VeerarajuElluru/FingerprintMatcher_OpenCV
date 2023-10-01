# Goal is to find the best matching fingerprints. We would take altered fingerprints and match them with original ones
# THe dataset 'Altered' consists of altered images, while the 'Real' contains the original fingerprints
# We use OpenCV

import cv2
import os

sample = cv2.imread("SOCOFing/Altered/Altered-Hard/150__M_Right_index_finger_Obl.BMP")   # to load an image
"""the above dummy sample has the obliteration and the central rotation as its alterations,
it is also quite small"""
# sample = cv2.resize(sample, None, fx=2.5, fy=2.5)   # fx and fy are the scaling factors
# cv2.imshow('Sample', sample)    # disappears immediately
# cv2.waitKey(0)
# cv2.destroyAllWindows()
"""now we basically compare the key points of this image with the key points of all the real data samples"""

"""Section 1: Initialising Variables required"""

best_score = 0
filename = None
image = None    # to store the best match

kp1, kp2, matchPoints = None, None, None # this is to identify and plot diff matching points and similarities


"""Section 2: The core algorithm"""

# tracker
counter = 0
# loop through all the real images in the directory
for file in [file for file in os.listdir("SOCOFing/Real")][:1000]:
    if counter % 10 == 0:
        print(counter, file)
    counter += 1
    finger_print = cv2.imread("SOCOFing/Real/"+file)    # load the real image in every iteration

    # creation of SIFT objects-go thru terminologies file
    sift = cv2.SIFT_create()
    key_points1, descriptors_1 = sift.detectAndCompute(sample, None)
    key_points2, descriptors_2 = sift.detectAndCompute(finger_print, None)

    # find all the matching key points, the '1' in algorithm argument corresponds to KD Tree
    index_param = dict(algorithm=1, trees=4)
    matches = cv2.FlannBasedMatcher(index_param, {}).knnMatch(
        descriptors_1,
        descriptors_2,
        k=2
    )
    # the above gives us all matches

    mp = []     # to store valid matches
    for p,q in matches:
        if p.distance < 0.1*q.distance:     # case of valid matching
            mp.append(p)

    key_points = min(len(key_points1), len(key_points2))
    if len(mp)/key_points * 100 > best_score:
        best_score = len(mp)/key_points * 100
        filename = file
        image = finger_print
        kp1, kp2, matchPoints = key_points1, key_points2, mp

print("Best Match:" + filename)
print("Score:" + str(best_score))

result = cv2.drawMatches(sample, key_points1, image, key_points2, mp, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Results",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
