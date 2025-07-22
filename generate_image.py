import cv2
import numpy as np

# Create a white background
img = np.ones((500, 500, 3), dtype=np.uint8) * 255  

# Define the 4 corners of a "document" (intentionally skewed)
corners = np.array([[120, 80], [380, 60], [420, 380], [80, 400]], dtype=np.int32)  

# Draw the document border
cv2.polylines(img, [corners], isClosed=True, color=(0, 0, 0), thickness=2)  

# Add some text inside
cv2.putText(img, "TEST DOCUMENT", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)  
cv2.putText(img, "Line 1: Perspective correction", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  
cv2.putText(img, "Line 2: Manual coordinates test", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  

cv2.imwrite("test_doc.png", img)  
cv2.imshow("Test Image", img)  
cv2.waitKey(0) 
