# Import the necessary libraries
import cv2
import json
import os

# Function that does nothing, to be used as a callback for the trackbar
def do_nothing(x):
    pass

# Function to handle mouse click events
def handle_click_event(event, x, y, flags, params):
    # Access global variables
    global points, max_points, current_point_counter, current_point_name

    # If right button clicked
    if event == cv2.EVENT_RBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Draw a circle at clicked point
        cv2.circle(sized_img, (x, y), 5, (255,0,0), -1)

        # If point counter is less than the max_points, process the click
        if current_point_counter < max_points:
            if current_point_name:
                # Add coordinates to points dictionary
                points[current_point_name]["coordinates"].append((x, y))

                # Display name of the point on image
                cv2.putText(sized_img, current_point_name, (x, y), font, 0.75, (0, 0, 255), 2)

            # Increment the counter
            current_point_counter += 1

        else:
            # If max_points is reached, get user input for new point name
            user_input = input("Enter your value: ")
            if user_input:
                # Display new point name on image
                cv2.putText(sized_img, user_input, (x, y), font, 0.75, (0, 0, 255), 2)

                # Get user input for field type and reset point counter and name
                field_type = input("Enter field type: ")
                current_point_counter = 1
                current_point_name = user_input

                # Initialize new point in points dictionary
                points[user_input] = {"coordinates": [(x, y)], "field_type": field_type}
        user_input = ''

# Input for file name and number of points per interest point
file_name =  input("Enter file name: ")
max_points = int(input("Enter number of points for each interest point: "))

# Initialize counter and point name
current_point_counter = max_points
current_point_name = ""

# Read image to label, and get its dimensions
image = cv2.imread('Release/input_images/Student_ID.jpg')
height, width, _ = image.shape

# Create window and trackbars for adjusting width and height
cv2.namedWindow('image')
cv2.createTrackbar('width', 'image', 100, 1000, do_nothing)
cv2.createTrackbar('height', 'image', 100, 1000, do_nothing)
cv2.setTrackbarPos('width', 'image', int(width/2))
cv2.setTrackbarPos('height', 'image', int(height/2))
previous_width = int(width/2)
previous_height = int(height/2)

# Resize image according to trackbar positions
resized_img = cv2.resize(image, (previous_width, previous_height))
size_previous = (height, width, _)

# Main loop to display and resize the image
while True:
    # Get current positions of trackbars
    width_trackbar = cv2.getTrackbarPos('width','image')
    height_trackbar = cv2.getTrackbarPos('height','image')

    # If either trackbar is moved, resize the image
    if( (previous_width != width_trackbar) | (previous_height != height_trackbar) ):
        previous_width = width_trackbar
        previous_height = height_trackbar
        resized_img = cv2.resize(image, (width_trackbar, height_trackbar))

    # If 'n' is pressed, store image size and break the loop
    if cv2.waitKey(30) & 0xFF == ord('n'):
        img_size = list(image.shape)
        img_size[0] = height_trackbar
        img_size[1] = width_trackbar
        break
    
    # Show image
    cv2.imshow("image", resized_img)

    # If 'q' is pressed, break the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Close all windows
cv2.destroyAllWindows()

# Make a copy of the resized image and initialize points dictionary
sized_img = resized_img.copy()
points = {}

# Main loop for setting labels
while True:
    # Display image
    cv2.imshow('Set labels', sized_img)

    # Set mouse callback
    s = cv2.setMouseCallback('Set labels', handle_click_event)

    # If 'q' is pressed, break the loop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Get input for processing type
processing_type = input("Enter processing type: ")

# Prepare json data
json_data = {
    'file_name': file_name,
    'img_size': img_size,
    'interest_points': points,
    'processing_type': processing_type
}

# Prepare file path
current_directory = os.path.dirname(os.path.abspath(__file__))
labeling_folder_path = os.path.join(current_directory, 'labeling')

# Ensure the directory exists
os.makedirs(labeling_folder_path, exist_ok=True) 

# Generate json file path
json_file_path = os.path.join(labeling_folder_path, f"{json_data['file_name']}.json")

# Save to json file
with open(json_file_path, 'w+') as json_file:
    json.dump(json_data, json_file)

# Close all windows
cv2.destroyAllWindows()