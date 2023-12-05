import json
import cv2

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


# Function to save image parts based on bounding boxes and interest points
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

# Function that does nothing, used as a placeholder for trackbar callback
def do_nothing(x):
    pass