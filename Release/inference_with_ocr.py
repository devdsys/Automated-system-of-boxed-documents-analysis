import cv2
import numpy as np
import json
import os 
import requests
import base64
import time
import sys

# Load images
image_path = 'Release/input_images/Student_ticket.jpg'

# OCR API key obtained from 'https://onlineocrconverter.com/profile'
api_key = 'DnmyqT84I5p5AxxoZ1nF3NoCwv2iZ9M605JxfXw1WfaGOUYvrLRL_veP8sq_ldw7BDY'

# Folder to save cropped parts of the image
parts_folder = 'Release/cropped_parts'
os.makedirs(parts_folder, exist_ok=True)

# Load the input image in gray scale
gray_image = cv2.imread(image_path,0)

# Load the template images in gray scale
templates_imgs = [cv2.imread('Release/templates/Stud_1.jpg',0), cv2.imread('Release/templates/Stud_2.jpg',0), cv2.imread('Release/templates/Stud_3.jpg',0)]
json_list = ['Release/labeling/Stud.json', 'Release/labeling/Stud.json', 'Release/labeling/School.json']

# Initialize feature detector and descriptor
sift = cv2.xfeatures2d.SIFT_create()

# Calculate keypoints and descriptors for input image
kp_input, des_input = sift.detectAndCompute(gray_image,None)

# Create FLANN matcher
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

    # If the number of good matches is higher than the current maximum, update the maximum
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

# Function that does nothing, used as a placeholder for trackbar callback
def nothing(x):
    pass

# Check if all interest points are inside bounding boxes and each box contains a point
def check_points_in_boxes_with_rect(json_file, contours):
    # Load json file
    with open(json_file) as f:
        data = json.load(f)
    
    # Prepare a dictionary to record if a point has been assigned to a box.
    assigned_points = {key: [False]*len(val["coordinates"]) for key, val in data['interest_points'].items()}

    # Prepare a list to record if a box has been assigned a point.
    box_assigned = [False]*len(contours)

    # Extract bounding boxes from contours
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Check for each box
    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        # Check for each point
        for point_name, point_info in data['interest_points'].items():
            points = point_info["coordinates"]
            for j, point in enumerate(points):
                px, py = point
                if x <= px <= x+w and y <= py <= y+h:  # point is inside the bbox
                    # Check if this point is already assigned to a box.
                    if assigned_points[point_name][j]:
                        return False
                    assigned_points[point_name][j] = True
                    box_assigned[i] = True

    # Check if there are points not assigned to any box.
    for point_name, assigned in assigned_points.items():
        if not all(assigned):
            return False

    # Check if there are boxes not assigned any point.
    for i, assigned in enumerate(box_assigned):
        if not assigned:
            return False

    return True


# Save image parts based on bounding boxes and interest points
def save_image_parts(json_file, contours, image):
    # Load json file
    with open(json_file) as f:
        data = json.load(f)

    # Extract bounding boxes from contours and create mapping with points
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Check for each box
    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox

        # Get cropped region
        cropped = image[y:y+h, x:x+w]

        # Check for each point
        for point_name, point_info in data['interest_points'].items():
            points = point_info["coordinates"]
            px, py = points[0]  # consider only the first pair
            if x <= px <= x+w and y <= py <= y+h:  # point is inside the bbox

                # Save cropped region with the name of point
                filename = f"Release/cropped_parts/{point_name}.jpg"
                cv2.imwrite(filename, cropped)
        

# Call OCR API to recognize text in cropped image parts
def call_ocr_api(image_folder, json_file, api_key, language='ukr'):
    # Load JSON data
    with open(json_file) as f:
        data = json.load(f)


    responses = {}
    for field, details in data['interest_points'].items():
        # Check if field_type is 'text', process the image
        if details['field_type'] == 'text':
            # Construct image file path
            image_path = os.path.join(image_folder, field + ".jpg")
            # Validate if the image exists
            if os.path.isfile(image_path):
                # Read image file
                with open(image_path, 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode()

                # API endpoint
                url = 'https://api.onlineocrconverter.com/api/image'
                
                # Headers for request
                headers = {
                    'key': api_key
                }

                # Data for request
                api_data = {
                    'base64': encoded_string,
                    'language': language
                }

                # Make a post request to the API
                response = requests.post(url, headers=headers, json=api_data)

                # Check the status code of the response
                if response.status_code == 200:
                    response_data = response.json()
                    if 'text' in response_data:
                        # Store the response
                        responses[field] = response_data['text'].strip().replace('\n', ' ')
                    else:
                        print(f"'text' not found in response for field '{field}'. Response: {response_data}")
                else:
                    print(f"Request for field '{field}' failed with status code {response.status_code}")

                # Sleep for 1 second to avoid hitting rate limits
                time.sleep(1)

    return responses

# Load JSON data 
with open(json_path) as f:
    data = json.load(f)

# Extract image size from JSON data and resize image accordingly
w, h, s =  data['img_size']
image = cv2.imread(image_path)
image = cv2.resize(image, (h, w))

# Create a window for trackbars
cv2.namedWindow('image')

# Create trackbars for color change
cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

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
while(True):
    # Copy the original image
    img = image.copy()

    # Get current positions of all trackbars
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

    # Initialize counter and list of obtained contours
    cntr = 0
    obtained_contours = []

    # Iterate through each contour and draw bounding rectangle 
    # if it satisfies certain conditions
    for c in contours:
        rect = cv2.boundingRect(c)
        if rect[2] < 25 or rect[3] < 25 or cv2.contourArea(c) < 2000: continue
        x,y,w,h = rect
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cntr += 1
        obtained_contours.append(c)

    # If the number of contours matches the number of interest points, 
    # check if all interest points are inside bounding rectangles
    if cntr == num_interest_points:
        result = check_points_in_boxes_with_rect(json_path, obtained_contours)
        # If the result is true, save the image parts and break the loop
        if result : 
            cv2.destroyAllWindows()
            save_image_parts(json_path, obtained_contours, image)
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

    # Break loop on 'q' key press
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()

# Call the OCR API to recognize text in the image parts
responses = call_ocr_api(parts_folder, json_path, api_key)
print(responses)

