import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('..BmW.png', 0)

contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]

image_external = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i , 255, -1)
        
image_interna = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image_internal, contours, i, 255, -1)
        
plt.figure(figsize=(10,3))
plt.subplot(131)
pllt.axis('off')
pt.title('original')
plt.imshow(image, cmap='gray')
plt.subplot(132)
pllt.axis('off')
pt.title('external')
plt.imshow(imagem_external, cmap='gray')

plt.subplot(133)
pllt.axis('off')
pt.title('internal')
plt.imshow(imagem_internal, cmap='gray')
plt.tight_layout()
plt.show()