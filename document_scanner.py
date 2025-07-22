from imagepython.transform import top_down_view
from imagepython.scanning_tools import *
from skimage.filters import threshold_local
import numpy as np
import cv2 as cv
import argparse #  As a substitute for sys, it is more user friendly
import imutils

# Get arguments
parser = argparse.ArgumentParser(description="This script scans documents and outputs a well-adjusted version of them.")
arguments = get_arguments(parser)

# Start with edge detection
# First, let's load the image
image = cv.imread(arguments["image"])
# For edge detection, we'll resize the image (make it smaller) so the process becomes faster
# Sure, that would mean losing information but it's temporary — just to get the edges (which aren't affected by such loss)
copy_image = image.copy()
copy_image = imutils.resize(image, height=500)  # Don't worry, the width is automatically scaled too
scaling_ratio = image.shape[0] / 500  # This gives us the height ratio of the image
edged_image, gray_image = edge_detection(copy_image)

# Now for corner detection. This is just a step I added to the script thinking it
# would help me find coordinates for the 4 specific corners defining our document,
# but it turned out to be complicated since filtering the rest of the corners is hard
copy_corners = corner_detection(gray_image, copy_image)

# Instead, we'll do that through contour detection
document_contour = contours_detection(edged_image)
cv.drawContours(copy_image, [document_contour], -1, (255, 0, 0), 3)  # -1: draw all given contours (index -1 means all), 3 is the thickness

# Now we'll build a top-down view of the image
warped_image = top_down_view(image, document_contour.reshape(4, 2) * scaling_ratio)

# Convert to grayscale
warped_image_gray = warped_image.copy()
warped_image_gray = cv.cvtColor(warped_image_gray, cv.COLOR_BGR2GRAY)

# Adaptive thresholds
thresholds_gray = threshold_local(warped_image_gray, 11, offset=10, method="gaussian")  # We use the Gaussian filter for a more pixel-adapted thresholding
                                                                                        # offset=10 is subtracted from the threshold to make it stricter
                                                                                        # 11 is the size of the window for thresholding
thresholds = threshold_local(warped_image, 11, offset=10, method="gaussian")
warped_image_gray = (warped_image_gray > thresholds_gray).astype("uint8") * 255

# Show the original image
cv.imshow("Original image", imutils.resize(image, height=1012))  # Fits on most screens: A height of 650 pixels typically fits well in a standard 1080p monitor window

# Show the edged image
""" print("Edge Detection")
cv.imshow("Edged image", edged_image) """

# Highlight corners (pretty much useless as I explained above)
""" print("Corner Detection")
cv.imshow("Corners detected", copy_corners) """

# Show contours
""" print("Contour Detection")
cv.imshow("Contour", copy_image) """

# Show the scanned image
print("Scanned image")
form = int(input("If you prefer this document in grayscale, type 0. If you want to keep the colors, type 1: "))
if form == 1:
    cv.imshow("Scanned image", imutils.resize(warped_image, height=1012))
else:
    cv.imshow("Scanned image in grayscale", imutils.resize(warped_image_gray, height=1012))
cv.waitKey(0)
cv.destroyAllWindows()
