import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.util as skim


colored_image = cv2.imread('baboonRGB.bmp')

############################################################################################

# Etape 1: Detection de contours
# - Appliquer deux filtres passe-hauts
# - Bruiter l'image en niveau de gris (pourcentage enrte 15% et 25%)
# - Detection de contours
#       - Passe-bas (DoG = Difference of Gaussian)
#       - Passe-bas + Passe-haut (Gaussian + Sobel). Varier parametres et noter

############################################################################################

### Appliquer filtres passe-hauts

# Laplacian
laplacian_image = cv2.Laplacian(colored_image, cv2.CV_64F)

# Prewitt
kernelX = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernelY = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
PrewittX = cv2.filter2D(colored_image, -1, kernelX)
PrewittY = cv2.filter2D(colored_image, -1, kernelY)
prewitt_image = PrewittX + PrewittY

############################################################################################

### Bruiter les images (gris)

# Images en niveaux de gris
laplacian_gray_scale = cv2.cvtColor(laplacian_image, cv2.COLOR_RGB2GRAY)
prewitt_gray_scale = cv2.cvtColor(prewitt_image, cv2.COLOR_RGB2GRAY)

# Bruitage S&P sur Laplacian (15% and 25%)
laplacian_sp_15 = skim.random_noise(laplacian_gray_scale, node='s&p', amount=0.15)
laplacian_sp_25 = skim.random_noise(laplacian_gray_scale, node='s&p', amount=0.25)

# Bruitage S&P sur Prewwit (15% and 25%)
prewitt_sp_15 = skim.random_noise(prewitt_gray_scale, node='s&p', amount=0.15)
prewitt_sp_25 = skim.random_noise(prewitt_gray_scale, node='s&p', amount=0.25)

############################################################################################

# Détection contours (DoG)



# Détection contours (Gaussian + Sobel)

############################################################################################

# Display images
cv2.imshow("Initial Image", colored_image)
cv2.imshow("Laplacian Image", laplacian_image)
cv2.imshow("Prewitt Image", prewitt_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
