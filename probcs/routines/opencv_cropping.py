import numpy as np
import cv2
from pathlib import Path


# Function to display the image and get the cropping coordinates
def crop_image(image):
    clone = image.copy()

    roi = []

    def select_roi(event, x, y, flags, param):
        nonlocal roi
        if event == cv2.EVENT_LBUTTONDOWN:
            roi = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            roi.append((x, y))
            cv2.rectangle(image, roi[0], roi[1], (0, 255, 0), 2)
            cv2.imshow("Image", image)

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", select_roi)

    # Keep looping until 'q' is pressed
    while True:
        cv2.imshow("Image", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Close the open window
    cv2.destroyAllWindows()

    if len(roi) == 2:
        return clone[roi[0][1] : roi[1][1], roi[0][0] : roi[1][0]]
    else:
        return None


# Path to your .npy file
basepath = Path("/src/project/data/experiment")
exc_fname = basepath / "exc_images.npy"

# load the images
images = np.load(exc_fname, allow_pickle=True)
image_id = 0
image = images[image_id]

# Crop the image
cropped_image = crop_image(image)

# Check if the cropping was done and save the result
if cropped_image is not None:
    # Save the cropped image as a .npy file
    np.save("cropped_image.npy", cropped_image)
    print("Cropped image saved as .npy file.")
else:
    print("Cropping was cancelled.")
