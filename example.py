from imagepython.transform import four_points_transform
import numpy as np
import cv2 as cv
import argparse # As a substitute for sys, it is more user friendly

def parse_coords_float(string):
    parts = string.split(',') 
    # through map we'll apply the function float to our list parts
    # be warned map only works with itterables (even though one element is still an iterable)
    points = list(map(float, parts))
    # since the coordinates are given in the order indicated in help reshape would keep things in check
    return np.array(points, dtype = "float32").reshape((4,2)) 
parser = argparse.ArgumentParser(description = "This script is a test to see whether our transform file works or not")

parser.add_argument("--image", help = "path to image")  #the arguments are strings by default
# her we'll specify the type (which is just a function that would transform our string into the right format)
parser.add_argument("--coordinates", type = parse_coords_float, help = "coordinates of the four points separated with a comma in this order : x1,y1,x2,y2, etc ...")

#we'll read the arguments past by the user through pars_args and we'll convert them into a dictionnary using vars
arguments = vars(parser.parse_args()) 

# as i've seen in other versions we could just use eval insteadof parse_coords_float like this:
#parser.add_arguments("--coordinates", help = "coordinates of the four points separated with a comma in this order : x1,y1,x2,y2, etc ...")
#points = np.array(eval(args["coordinates"], dtype = "float32"))
#but then the user should enter coordinates in this form : (x1,y1), (x2,y2) etc which makes more sense, but i am trying to keep original 

points = arguments["coordinates"]
image = cv.imread(arguments["image"])
if image is None:
    print("Failed to load image:", arguments["image"])
    exit(1)
ß
#get the top-down view (also called birds eye view)
warped_image = four_points_transform(image, points)

#show original and wraped images

cv.imshow("original image", image)
# impose a delay of 0, meaning the image will stay opened until a key is pressed
cv.waitKey(0)
input("Press Enter to exit..")
cv.destroyAllWindows()
