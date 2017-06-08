import cv2
import numpy as np

for fname in ['smoke1','smoke2','smoke3','smoke4']:
	# Load image and convert to black and white, make type=float32 
	img = cv2.imread('data/%s.jpeg' % fname)
	H = img.shape[0]
	W = img.shape[1]
	bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	bw = np.float32(bw)
	for blockSize in [7]:
		for ksize in [7]:
			for k in [0.01]:

				# Find Harris corners
				dst = cv2.cornerHarris(bw,blockSize,ksize,k)
				# Dilate result for appearance
				dst = cv2.dilate(dst,None)
				# Threshold imposition
				print dst
				img[np.logical_and(dst>0.00001*dst.max(),dst<0.0001*dst.max())]=[0,0,255]
				cv2.namedWindow('bSz %d ksz %d k %f' % (blockSize, ksize, k),cv2.WINDOW_NORMAL)
				imS = cv2.resize(img, (W/2,H/2))
				cv2.imshow('bSz %d ksz %d k %f' % (blockSize, ksize, k),imS)
				if cv2.waitKey(0) & 0xff == 27:
					cv2.destroyAllWindows()
