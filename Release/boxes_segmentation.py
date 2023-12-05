import cv2
import numpy as np
import json
import sys
from functions import check_points_in_boxes_with_rect, do_nothing

# Load images
image_path = 'Release/input_images/Student_ticket.jpg'
gray_image = cv2.imread(image_path,0)

# Load template images
templates_imgs = [cv2.imread('Release/templates/Stud_1.jpg',0), cv2.imread('Release/templates/Stud_2.jpg',0), cv2.imread('Release/templates/Stud_3.jpg',0)]

# List of paths to JSON files
json_list = ['Release/labeling/Stud.json', 'Release/labeling/Stud.json', 'Release/labeling/School.json']

# Initialize feature detector and descriptor
sift = cv2.xfeatures2d.SIFT_create()

# Calculate keypoints and descriptors for input image
kp_input, des_input = sift.detectAndCompute(gray_image,None)

# Initialize matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) 
matcher = cv2.FlannBasedMatcher(index_params, search_params)

# Variables for storing best match info
max_matches = 0
document_in_camera = None
best_match = None
best_kp_document = None
best_kp_camera = None

# Set minimum number of matches
min_matches = 20

# Iterate through each template image
for i, document_img in enumerate(templates_imgs):

    # Calculate keypoints and descriptors for document image
    kp_document, des_document = sift.detectAndCompute(document_img,None)
    matches = matcher.knnMatch(des_document, des_input, k=2)

    # Store all the good matches as per Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good_matches.append(m)

    if len(good_matches)>max_matches:
        max_matches = len(good_matches)
        json_path = json_list[i]
        best_match = good_matches
        best_kp_document = kp_document
        best_kp_camera = kp_input

# If max matches is less than the minimum required, stop the execution
if max_matches < min_matches:
    print(f"Number of matches {max_matches} is less than the minimum required {min_matches}. Stopping execution.")
    sys.exit()

print("The document in the camera image is:", json_path)


# Function to save image parts based on bounding boxes and interest points
def visualize_and_print(json_file, contours, image):
    # Load json file
    with open(json_file) as f:
        data = json.load(f)

    # Extract bounding boxes from contours
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    print("Bounding Boxes:", bounding_boxes)
    print("Interest Points:", data['interest_points'])

    # Draw boxes and corresponding point on the image
    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check for each point
        for point_name, points in data['interest_points'].items():
            points = points if isinstance(points[0], list) else [points]
            for point in points:
                px, py = point
                if x <= px <= x+w and y <= py <= y+h:  # point is inside the bbox
                    # Draw point
                    cv2.circle(image, (px, py), radius=5, color=(0, 0, 255), thickness=-1)
                    print(f"Point {point_name} is inside box {i}")

    cv2.imshow("Image with boxes and points", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#Load JSON data 
with open(json_path) as f:
    data = json.load(f)

# Extract image size from JSON data and resize image accordingly   
w, h, s =  data['img_size']
image = cv2.imread(image_path)
image = cv2.resize(image, (h, w))

# Create a window for trackbars
cv2.namedWindow('image')

# create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,do_nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,do_nothing)
cv2.createTrackbar('VMin','image',0,255,do_nothing)
cv2.createTrackbar('HMax','image',0,179,do_nothing)
cv2.createTrackbar('SMax','image',0,255,do_nothing)
cv2.createTrackbar('VMax','image',0,255,do_nothing)

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMax', 'image', 179)
cv2.setTrackbarPos('SMax', 'image', 255)
cv2.setTrackbarPos('VMax', 'image', 255)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0

# Initialize outupe image and wait time for button pressing
output = image
wait_time = 33

# Extract the 'interest_points'
interest_points = data['interest_points']

num_interest_points = len(interest_points)
# Convert the interest points to a list of tuples
interest_points_list = [{k: tuple(v)} for k, v in interest_points.items()]

# Main loop for adjusting thresholds and finding contours
while(1):
    # Create a copy of the original image
    img = image.copy()

    # Get the current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert the image to HSV and apply the threshold
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image,image, mask= mask)
    mask = cv2.bitwise_not(mask)

    # Apply erosion to the mask
    dilation = cv2.erode(mask,(3,3),iterations = 3)

    # Find contours in the eroded image
    contours, hierarchy = cv2.findContours(dilation, 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cntr = 0
    obtained_contours = []
    # Iterate through each contour and draw bounding rectangle if it satisfies conditions
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 25 or rect[3] < 25 or cv2.contourArea(c) < 2000: continue
        x,y,w,h = rect
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cntr += 1
        obtained_contours.append(c)

    print('Cntrs', cntr)
    # If the number of contours matches the number of interest points, check and visualize results
    if cntr == num_interest_points:
        result = check_points_in_boxes_with_rect(json_path, obtained_contours)
        print(result)
        if result : 
            cv2.imshow('image',output)
            cv2.imshow('box', img)
            cv2.imshow('mask', mask)

            # Save image parts if 'q' is pressed
            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    # Increment 'SMin' trackbar position until all boxes are found or limit is reached
    if sMin < 29:
        sMin +=14
    elif sMin < 75:
        sMin += 8
    elif sMin < 245: 
        sMin +=5
    else:
        print("Not all boxes found")
        break

    cv2.setTrackbarPos('SMin', 'image', sMin)

    # Display output images
    cv2.imshow('image',output)
    cv2.imshow('box', img)
    cv2.imshow('mask', mask)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()