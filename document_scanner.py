from imagepython.transform import top_down_view
from skimage.filters import threshold_local
import numpy as np
import cv2 as cv
import argparse #  As a substitute for sys, it is more user friendly
import imutils

def edge_detection(copy_image):
    "A function that detects edges and returns a grayscale image where they are well highlighted"
    # Let's convert our image to grayscale to avoid computing all of red, blue, and green
    gray_image = cv.cvtColor(copy_image, cv.COLOR_BGR2GRAY)
    # Now, for smoothing, we'll use a Gaussian filter to remove noise.
    # This is done through a convolution, which corresponds to a multiplication
    # in the frequency domain that attenuates high-frequency noise.
    gray_image = cv.GaussianBlur(gray_image, (5,5), 0)  # (5, 5) is the kernel size: how much neighbors influence each other
    # Finally, for edge detection we call Canny for gradient calculations (Sobel)
    # (I'm not sure if it doesn't use Laplacian too in some scenarios)
    edge_image = cv.Canny(gray_image, 75, 200)  # 75 and 200 are the thresholds
    return edge_image, gray_image

def get_arguments(parser):
    "A function that adds some manually given arguments to the parser passed as parameter"
    # We'll put a requirement on the input just to be safe
    parser.add_argument("--image", required=True, help="path to image")
    # Get arguments in the form of a dictionary
    arguments = vars(parser.parse_args())
    return arguments

def corner_detection(gray_image, copy_image):
    "A function that detects corners and returns an image where they are highlighted in red"
    gray_image_float = np.array(gray_image, dtype="float32")
    # Calculate the matrix R = det(M) - k.(tr(M))² where M is the second moment matrix
    harris_response = cv.cornerHarris(gray_image_float, blockSize=2, ksize=3, k=0.04)  # ksize is the Sobel kernel size
                                                                                       # and k is just a constant usually between 0.04 and 0.06
    kernel = np.ones((3, 3), np.uint8)
    # We dilate the corner regions so they are more visible
    # The kernel is (3,3) by default so we could've just written None instead, but it's all for learning purposes
    harris_response = cv.dilate(harris_response, kernel)
    # Threshold
    maximum_response = harris_response.max()
    copy_corners = copy_image.copy()
    for i in range(copy_corners.shape[0]):
        for j in range(copy_corners.shape[1]):
            if harris_response[i][j] > 0.01 * maximum_response:
                copy_corners[i][j] = [0, 0, 255]  # mark it in red 
    return copy_corners

def contours_detection(edged_image):
    "A function that detects contours"
    # RETR_LIST gives out all contours, and CHAIN_APPROX_SIMPLE removes intermediate points in a segment
    contours_image = cv.findContours(edged_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # findContours returns either (image, contours, hierarchy) or (contours, hierarchy), either way we only want contours
    contours_image = imutils.grab_contours(contours_image)
    contours_image = sorted(contours_image, key=cv.contourArea, reverse=True)[:5]  # Pick the five largest contours by area
    document_contour = None
    for contour in contours_image:
        # Compute the perimeter of the contour (its length)
        perimeter = cv.arcLength(contour, True)
        # Now we'll use the perimeter to indicate a tolerance to the Douglas-Peucker algorithm
        # This algorithm simplifies the contour by eliminating some of its points while keeping the main shape
        approximate_polygone = cv.approxPolyDP(contour, 0.01 * perimeter, True)
        # If our contour has approximately 4 corners, it's likely a rectangle or a square
        # and we already know it's big enough through the sorting, so it might be our document
        if len(approximate_polygone) == 4:
            document_contour = approximate_polygone
            break
    if document_contour is None:
        raise Exception("Could not find document contour")
    return document_contour

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
