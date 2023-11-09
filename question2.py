# Etape 2: Segmentation par seuillage de couleur
# - Operer une segmentation pour 2 images couleurs (Domaine HSV doit etre utiliser pour 1)

import cv2
import numpy as np

baboonRGB = cv2.imread('BaboonRGB.bmp')
kiwiRGB = cv2.imread('kiwi.jpg')

# Convert 1 image to HSV
baboonHSV = cv2.cvtColor(baboonRGB, cv2.COLOR_BGR2HSV)

# Color Threshold
## This mask would be representing the originally red section
## The HSV image, having different color representation, will display the same section with a different color
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

lower_green = np.array([0, 60, 0])
upper_green = np.array([70, 255, 70])

# Create masks from threshold
mask1 = cv2.inRange(baboonHSV, lower_red, upper_red)
mask2 = cv2.inRange(kiwiRGB, lower_green, upper_green)

# Applying mask to images
segmented_image1 = cv2.bitwise_and(baboonHSV, baboonHSV, mask=mask1)
segmented_image2 = cv2.bitwise_and(kiwiRGB, kiwiRGB, mask=mask2)

# Displaying images
cv2.imshow('Original Kiwi RGB', kiwiRGB)
cv2.imshow('Original Baboon RGB', baboonRGB)
cv2.imshow('Baboon HSV', baboonHSV)
cv2.imshow('Segmented Image 1', segmented_image1)
cv2.imshow('Segmented Image 2', segmented_image2)

cv2.waitKey(0)
cv2.destroyAllWindows()

