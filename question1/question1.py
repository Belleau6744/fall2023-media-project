import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
colored_image = cv2.imread("laughing_cow.png")

# Convertir l'image en niveaux de gris
gray_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2GRAY)

# Appliquer le filtre de gradient Roberts
roberts_x = cv2.filter2D(gray_image, -1, np.array([[1, 0], [0, -1]]))
roberts_y = cv2.filter2D(gray_image, -1, np.array([[0, 1], [-1, 0]]))
roberts_gradient = np.sqrt(np.square(roberts_x) + np.square(roberts_y))

# Appliquer le masque de Sobel
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
sobel_gradient = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

# Afficher les résultats
plt.figure(figsize=(12, 6))

plt.subplot(231), plt.imshow(colored_image[:, :, ::-1]), plt.title('Image originale')
plt.subplot(232), plt.imshow(gray_image, cmap='gray'), plt.title('Image en niveaux de gris')
plt.subplot(233), plt.imshow(roberts_gradient, cmap='gray'), plt.title('Contours Roberts')
plt.subplot(234), plt.imshow(sobel_gradient, cmap='gray'), plt.title('Contours Sobel')

# Bruiter l'image en niveaux de gris avec un certain pourcentage de bruit (entre 15% et 25%)
noise_percentage = 20
noise = np.random.normal(0, 1, gray_image.shape)
noisy_image = gray_image + noise_percentage * noise

# Clipper les valeurs pour s'assurer qu'elles restent dans la plage [0, 255]
noisy_image = np.clip(noisy_image, 0, 255)

# Convertir l'image bruitée en niveaux de gris
noisy_gray_image = np.uint8(noisy_image)

# Afficher l'image bruitée
plt.subplot(235), plt.imshow(noisy_gray_image, cmap='gray'), plt.title('Image bruitée')

# Appliquer le filtre passe-bas (Difference of Gaussian - DoG)
gaussian_blur1 = cv2.GaussianBlur(noisy_gray_image, (5, 5), 0)
gaussian_blur2 = cv2.GaussianBlur(noisy_gray_image, (9, 9), 0)
dog_image = gaussian_blur1 - gaussian_blur2

# Appliquer le filtre passe-bas + passe-haut (Gaussien + Sobel)
gaussian_blur = cv2.GaussianBlur(noisy_gray_image, (5, 5), 0)
sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
sobel_gradient_gaussian = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

# Afficher les résultats de la détection de contours avec les filtres passe-bas
plt.subplot(236), plt.imshow(dog_image, cmap='gray'), plt.title('Contours DoG')
plt.tight_layout()
plt.show()
