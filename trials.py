import cv2
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('sample/rxlr-4-2.jpg')

# Apply Canny
edges = cv2.Canny(img, 10, 80, 3, L2gradient=True)

plt.figure()
plt.title('Spider')
plt.imsave('dancing-spider-canny.png', edges, cmap='gray', format='png')
plt.imshow(edges, cmap='gray')
plt.show()