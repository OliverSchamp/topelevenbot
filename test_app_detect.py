import cv2
import numpy as np

# 1. Load the image and convert to grayscale
img = cv2.imread('/mnt/SF_NAS/Oliver/homescreen.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Apply blur to reduce noise (prevents false circle detections)
gray_blurred = cv2.medianBlur(gray, 5)

# 3. Apply the Circular Hough Transform
# dp=1: same resolution as image
# minDist=100: min distance between centers (prevents overlapping circles)
# param1=50: Upper threshold for internal Canny edge detector
# param2=30: Threshold for center detection (lower = more circles)
circles = cv2.HoughCircles(gray_blurred, 
                            cv2.HOUGH_GRADIENT, 
                            dp=1.2, 
                            minDist=50,
                            param1=40, 
                            param2=18, 
                            minRadius=27, 
                            maxRadius=30)

if circles is not None:
    # Convert coordinates and radius to integers
    circles = np.uint16(np.around(circles))
    print(circles)
    
    # 4. Find the top-left icon by minimizing the sum of x and y
    # This is a robust heuristic for "top-left" in a coordinate system starting at (0,0)
    top_left_icon = min(circles[0, :], key=lambda c: c[0] + c[1])

    for i in circles[0, :]:
        # Draw all detected circles in green
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of each circle
        cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    # 5. Specifically highlight the top-left icon in Blue
    cv2.circle(img, (top_left_icon[0], top_left_icon[1]), top_left_icon[2] + 5, (255, 0, 0), 3)
    
    print(f"Top-Left Icon located at: X={top_left_icon[0]}, Y={top_left_icon[1]}")

    # Display the result
    cv2.imwrite('/mnt/SF_NAS/Oliver/homescreen_annotated.jpg', img)