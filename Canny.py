import cv2
import numpy as np
import matplotlib.pyplot as plt

for fname in ['smoke1','smoke2','smoke3','smoke4']:
	# Load image and convert to black and white, make type=float32 
	img = cv2.imread('data/%s.jpeg' % fname, 0)
	H = img.shape[0]
	W = img.shape[1]
	for var1 in [50,100,200]:
		for var2 in [50,100,200]:
			# Find Canny edges
			edges = cv2.Canny(img,var1,var2)
			print edges.shape
			print np.sum(edges)
			print np.max(edges)
			# Plot results
			plt.subplot(121), plt.imshow(img,cmap = 'gray')
			plt.title('Original'), plt.xticks([]), plt.yticks([])
			plt.subplot(122), plt.imshow(edges,cmap = 'gray')
			plt.title('Edges with %d and %d' % (var1, var2))
			plt.xticks([]), plt.yticks([])
			plt.show()

