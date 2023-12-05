
import cv2
import numpy as np

# Load the images
img1 = cv2.imread('path_to_manually_obtained_mask', 0)
img2 = cv2.imread('path_to_algorithm_segmented_mask', 0)

# Apply thresholds
_, img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
_, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)

# Calculate True positive, True negative, False positive, False negative
TP = np.logical_and(img1, img2)
TN = np.logical_and(np.logical_not(img1), np.logical_not(img2))
FP = np.logical_and(np.logical_not(img1), img2)
FN = np.logical_and(img1, np.logical_not(img2))

# Calculate total accuracy
accuracy = (np.sum(TP) + np.sum(TN)) / (np.sum(TP) + np.sum(TN) + np.sum(FP) + np.sum(FN))

# Calculate Precision and Recall
precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))

f1_score = 2 * ((precision * recall) / (precision + recall))

# Create the final image 
final_image = np.zeros((img1.shape[0], img1.shape[1], 3), dtype=np.uint8)
final_image[TP] = [255, 255, 255]  # Set True positive to white
final_image[TN] = [0, 0, 0]  # Set True negative to black
final_image[FP] = [0, 0, 255]  # Set False positive to red
final_image[FN] = [0, 0, 255]  # Set False negative to red

# Convert masks to color images
TP_img = np.zeros_like(final_image)
TP_img[TP] = [0, 0, 255]

TN_img = np.zeros_like(final_image)
TN_img[TN] = [0, 0, 255]

FP_img = np.zeros_like(final_image)
FP_img[FP] = [0, 0, 255]

FN_img = np.zeros_like(final_image)
FN_img[FN] = [0, 0, 255]

# Show the images
cv2.imshow('True Positive', TP_img)
cv2.imshow('Tue Negative', TN_img)
cv2.imshow('False Positive', FP_img)
cv2.imshow('False Negative', FN_img)
cv2.imshow('Final Image', final_image)

# Define a mask for red color
mask = (final_image[:,:,2] == 255) & (final_image[:,:,1] == 0) & (final_image[:,:,0] == 0)

# Count the number of red pixels
red_pixels = np.sum(mask)

# Calculate the percentage of red pixels
red_percentage = (red_pixels / (final_image.shape[0] * final_image.shape[1])) * 100

print(f'Red pixels percentage: {red_percentage}%')

# Print the accuracy
print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision * 100:.2f}%')
print(f'Recall: {recall * 100:.2f}%')
print(f'F1 Score: {f1_score * 100:.2f}%')

cv2.waitKey(0)
cv2.destroyAllWindows()

