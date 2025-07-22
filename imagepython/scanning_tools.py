import numpy as np
import cv2 as cv
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
