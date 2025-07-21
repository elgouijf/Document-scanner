from imagepython.transform import four_points_transform
import numpy as np
import cv2 as cv
import argparse # As a substitute for sys, it is more user friendly
import imutils

# get arguments
parser = argparse.ArgumentParser(description= "this script scanns documents giving in output a well adjusted version of them ")
# we'll put a requirement on the input just to be save
parser.add_argument("--image", required= True, help = "path to image")
# get arguments in form of a dictionnary
arguments = vars(parser.parse_args())

#now we start with the edge detection
#first let's load the image
image = cv.imread(arguments["image"])

#now for edge detection we'll receize the image (make it smaller) so the process becomes faster 
#sure that would mean loosing information but it's temporarly to get the edges (which is not concerned with such lose)

scaling_ratio = image.shape[0]/500 #this gets us the hight of the image

copy_image = image.copy()
copy = imutils.resize(image, height = 500) #don't worry the width is automatically put to scale too

#let's convert our image to graysclae, to a void comuting all of red, blue and green
gray_image = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
# Now, for smoothing, we'll use a Gaussian filter to remove noise.
# This is done through a convolution, which corresponds to a multiplication
# in the frequency domain that attenuates high-frequency noise.
gray_image = cv.GaussianBlur(gray_image, (5,5), 0) #(5, 5) is the kernel size : how much nekghbors influence each other

#finally for edge detection we call canny for gardient calculations(sobel) (i am not sure if it doesn't use the laplacian too in some scenarios)
edge_image = cv.Canny(gray_image, 75, 200) #75 and 200 are the thresholds

#show the original and edged image
print(" Edge Detection ")
cv.imshow("original image", image)
cv.imshow("Edged image", edge_image)
cv.waitKey(0)
input("Press Enter to exit..")
cv.destroyAllWindows()
