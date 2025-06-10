import cv2
import numpy as np

# Load the image
image = cv2.imread('img/contour_match_test/test_img.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# #DRILL HEADERS

# # Apply threshold to get image with only black and white
# _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
# # invert the image
# thresh = cv2.bitwise_not(thresh)

# cv2.imshow('Thresh', thresh)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Find contours
# contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# # Store header information
# headers = []

# # First pass: collect all valid headers
# for contour in contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     # Filter for rectangle-like shapes (aspect ratio and size)
#     if w > 300 and y < 300 and y > 100:
#         headers.append({'x': x, 'y': y, 'w': w, 'h': h})
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# # Calculate average dimensions
# if headers:
#     avg_width = sum(header['w'] for header in headers) / len(headers)
#     avg_height = sum(header['h'] for header in headers) / len(headers)
    
#     # Sort headers by y-coordinate
#     headers.sort(key=lambda x: x['x'])

#     gaps = [headers[i]['x'] - (headers[i-1]['x']+headers[i-1]['w']) for i in range(1, len(headers))]
#     print(gaps)
#     median_gap = np.median(gaps)
#     print(median_gap)
#     print(avg_width + int(median_gap))
    
#     # Find gaps in the sequence
#     for i in range(len(headers) - 1):
#         gap = headers[i + 1]['x'] - headers[i]['x']
#         if gap > avg_width*1.1:  # If gap is large enough to fit another header
#             # Calculate position for missing header
#             missing_y = headers[i]['y']
#             missing_x = headers[i]['x'] + headers[i]['w'] + int(median_gap)
            
#             # Draw box for missing header
#             print(f"Drawing missing header at {missing_x}, {missing_y} with width {avg_width} and height {avg_height}")
#             cv2.rectangle(image, 
#                         (missing_x, missing_y), 
#                         (missing_x + int(avg_width), missing_y + int(avg_height)), 
#                         (0, 255, 0), 2)  # Red color for missing header

# cv2.imshow('Detected Drill Headers', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# DRILLS

# Apply threshold to get image with only black and white
_, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

cv2.imshow('Thresh', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes around contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Filter for rectangle-like shapes (aspect ratio and size)
    if w > 300 and h > 100 and y < 500:
        x1 = x + int(w*0.05)
        y1 = y + int(h*0.45)
        x2 = x1 + int(w*0.9)
        y2 = y1 + int(50)
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)


# Save or display the result
# cv2.imwrite('output_image.jpg', image)
cv2.imshow('Detected Drills', image)
cv2.waitKey(0)
cv2.destroyAllWindows()