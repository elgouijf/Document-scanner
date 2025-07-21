from imagepython.transform import top_down_view
from skimage.filters import threshold_local
import numpy as np
import cv2 as cv
import argparse #  As a substitute for sys, it is more user friendly
import imutils

def edge_detection(copy_image):
    "a function that detects edges and returns an image in gray scale where they are well highlighted"
    # let's convert our image to graysclae, to a void comuting all of red, blue and green
    gray_image = cv.cvtColor(copy_image, cv.COLOR_BGR2GRAY)
    #  Now, for smoothing, we'll use a Gaussian filter to remove noise.
    #  This is done through a convolution, which corresponds to a multiplication
    #  in the frequency domain that attenuates high-frequency noise.
    gray_image = cv.GaussianBlur(gray_image, (5,5), 0) # (5, 5) is the kernel size : how much nekghbors influence each other

    # finally for edge detection we call canny for gardient calculations(sobel) (i am not sure if it doesn't use the laplacian too in some scenarios)

    edge_image = cv.Canny(gray_image, 75, 200) # 75 and 200 are the thresholds
    return edge_image, gray_image

def get_arguments(parser):
    "a function that adds some manually given arguments to the parser in parameters"
    #  we'll put a requirement on the input just to be save
    parser.add_argument("--image", required= True, help = "path to image")
    #  get arguments in form of a dictionnary
    arguments = vars(parser.parse_args())
    return arguments


def corner_detection(gray_image, copy_image):
    "a function that detects corners, and returns an image in which they are highlighted with red"
    gray_image_float = np.array(gray_image, dtype = "float32")

    # Calculate the Matrix of R = det(M) - k.(tr(M))² where M is second moment matrix
    harris_response = cv.cornerHarris(gray_image_float, blockSize = 2, ksize = 3,k = 0.04) # ksize is for the size of the sobel kernel size
                                                                                           #  and k is just a constant that can go from 0.04 to 0.06

    kernel = np.ones((3, 3), np.uint8)
    # we dialate corner regions so they are clearer
    # the kernel is (3,3) by default so we could've just written None instead but it's all for learning purposes
    harris_response = cv.dilate(harris_response, kernel)

    # Treshold
    maximum_response = harris_response.max()
    copy_corners = copy_image.copy()
    for i in range (copy_corners.shape[0]):
        for j in range (copy_corners.shape[1]):
            if harris_response[i][j] > 0.01*maximum_response :
                copy_corners[i][j] = [0,0,255] # mark it in red 
    return copy_corners


def contours_detection(edged_image):
    "a function that detects contours"
    # RETR_LIST gives out all contours, and CHAIN_APPROX_SIMPLE removes intermediate points in a srgment
    contours_image = cv.findContours(edged_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    #  the findContours returns either (image, contours, hierarchy) or (contours, hierarchy), either way we only want contours
    contours_image = imutils.grab_contours(contours_image)
    contours_image = sorted(contours_image, key = cv.contourArea, reverse = True)[:5] # pick the five biggest contours by area

    document_contour = None
    for contour in contours_image:
        #  compute the perimeter of the contour (its lenght)
        perimeter = cv.arcLength(contour, True)
        #  now we'll use the perimeter to indicate a tolerance to Douglas-Peucker algorithm
        #  this algorithm simplifies the contour by eliminating some of its points while keeping the main shape
        approximate_polygone = cv.approxPolyDP(contour, 0.01*perimeter, True)

        # if our contour has approximatly 4 corners it is likely a rectangle or a square 
        # and we already know it is big enough through sorted sp it might be our document
        if len(approximate_polygone) == 4: 
            document_contour = approximate_polygone
            break
    if document_contour is None:
        raise Exception("Could not find document contour")

    return document_contour


#  get arguments
parser = argparse.ArgumentParser(description= "this script scanns documents giving in output a well adjusted version of them ")
#  get arguments in form of a dictionnary
arguments = get_arguments(parser)

# now we start with the edge detection
# first let's load the image
image = cv.imread(arguments["image"])

# now for edge detection we'll receize the image (make it smaller) so the process becomes faster 
# sure that would mean loosing information but it's temporarly to get the edges (which is not concerned with such lose)
copy_image = image.copy()
copy_image = imutils.resize(image, height = 500) # don't worry the width is automatically put to scale too
scaling_ratio = image.shape[0]/500 # this gets us the hight of the image

edged_image, gray_image = edge_detection(copy_image)

# now for corners detection, this is just a step that i added to the script thinkng it
# would help me find coordinates for the 4 specific corners defining our document, but 
# it turned out to be complicated as filtering the rest of corners is hard
copy_corners = corner_detection(gray_image, copy_image) 

# instead we'll do that through Countour detection
document_contour = contours_detection(edged_image)
cv.drawContours(copy_image, [document_contour], -1, (255, 0, 0) , 3) #  -1 : all given contours (it is the index of the contour 
                                                                     #  to draw but -1 is for all), 3 is for thickness
# now we'll build a top-down view of the image
warped_image = top_down_view(image, document_contour.reshape(4,2) * scaling_ratio)

# convert to gray scale
warped_image_gray = warped_image.copy() 
warped_image_gray = cv.cvtColor(warped_image_gray, cv.COLOR_BGR2GRAY)

#Adaptive tresholds
tresholds = threshold_local(warped_image_gray, 11, offset = 10, method = "gaussian") # we use the gaussian filter for a more pixel adapted tresholding
                                                                                     # offset = 10 is substracted from the treshold to make stricter
                                                                                     # 11 is the size of the window for trshholding
warped_image_gray = (warped_image_gray > tresholds).astype("uint8")*255


# show the original image
cv.imshow("Original image", image)

# show the edged image
""" print(" Edge Detection ")
cv.imshow("Edged image", edged_image) """

# highlight corners (pretty much useless as i explained above)
""" print("Corners detection")
cv.imshow("Corners detected", copy_corners) """

# show contours
""" print("Contours detection")
cv.imshow("Contour", copy_image) """

#show the scanned image
print("Scanned image")
cv.imshow("Scanned image", warped_image_gray)
cv.waitKey(0)
cv.destroyAllWindows()