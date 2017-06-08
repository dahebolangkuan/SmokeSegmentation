import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.misc import imread
from scipy.spatial.distance import cdist
import cv2

# RECEIVES: the original RGB image 
# Uses pixel groupings in multiple directions to create texton features. Using max and min functions
# the algorithm can then select the neighbors most likely to characterize the pixel in question
# RETURNS: normalized feature vector of size (npixels,nfeatures)
def get_segmentation_features(img):

    # Apply 2D filters to image to obtain representations of texture 
    # (take into account some subset of nearby pixels)
    img_h = cv2.blur(img,(1,6))
    img_v = cv2.blur(img,(6,1))
    img_bilat = cv2.bilateralFilter(img,9,75,75)
    
    plt.subplot(222), plt.imshow(img_bilat)
    plt.title('Smoothed'), plt.xticks([]), plt.yticks([])

    num_features = 3
    features = np.zeros((img.shape[0] * img.shape[1], num_features))

    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            features[row*img.shape[1] + col, :] = np.array([img_bilat[row, col, 0], img_bilat[row, col, 1], img_bilat[row, col, 2]])

    norms = features.max(axis=0)
    # Normalize to weigh all features equally
    features_normalized = features / norms

    return features_normalized

# RECEIVES: the original RGB image and an image of the same size with segmentation region assignments
# RETURNS: array of size (nclusters,1) containing the ratio of harris corner pixels to total pixels in the boundary of each segment
def harris_smoke(img, clustered_img):
    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bw = np.float32(bw)
    num_clusters = np.max(clustered_img) # Should this be +1?
    smoke_clusters = np.empty((0,1))

    blockSize = 6
    ksize = 3
    k = 0.01

    # Find Harris corners
    dst = cv2.cornerHarris(bw,blockSize,ksize,k)
    
    for cluster in np.arange(num_clusters):
        cluster_pixels = (clustered_img==cluster)
        # Convert to float for dilate operation
        cluster_pixels = np.float32(cluster_pixels)
        cluster_pixels = cv2.dilate(cluster_pixels,None)
        # Change data type back to boolean
        cluster_pixels = (cluster_pixels==1)
        # Threshold imposition
        img_har = img
        img_har[np.logical_and(dst>0.0001*dst.max(),cluster_pixels)]
        har=np.sum(np.logical_and(dst>0.0001*dst.max(),cluster_pixels))
        tot=np.sum(cluster_pixels)
        ratio = np.double(har)/tot
        # Plot cluster of interest
        ROI_only = img
        ROI_only[cluster_pixels] = [0,0,0]
        cv2.imshow('ROI only',ROI_only)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
        if ratio < 0.29 and ratio > 0.15:
            smoke_clusters = np.vstack((smoke_clusters,cluster))

    return smoke_clusters.flatten()

# RECEIVES: the original RGB image and an image of the same size with segmentation region assignments
# RETURNS: array of size (nclusters,1) containing the ratio of canny edge pixels to total pixels in the boundary of each segment
def canny_smoke(img, clustered_img):
    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bw = np.float32(bw)
    num_clusters = np.max(clustered_img)
    smoke_clusters = np.empty((0,1))

    thresh1 = 50
    thresh2 = 50

    # Find Canny edges
    edges = cv2.Canny(img,thresh1,thresh2)
    
    for cluster in np.arange(num_clusters):
        cluster_pixels = (clustered_img==cluster)
        # Convert to float for dilate operation
        cluster_pixels = np.float32(cluster_pixels)
        cluster_pixels = cv2.dilate(cluster_pixels,None)
        # Change data type back to boolean
        cluster_pixels = (cluster_pixels==1)
        # Count Canny edges
        can=np.sum(np.logical_and(edges/255==1,cluster_pixels))
        tot=np.sum(cluster_pixels)
        ratio = np.double(can)/tot
        # Plot cluster of interest
        ROI_only = img
        ROI_only[cluster_pixels] = [0,0,0]
        cv2.imshow('ROI only',ROI_only)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
        if ratio < 0.29 and ratio > 0.15:
        	smoke_clusters = np.vstack((smoke_clusters,cluster))

    return smoke_clusters.flatten()

# RECEIVES: original rgb image, feature vector of size (npixels,nfeatures), and scalar bandwidth parameter\
# RETURNS: image of size (H,W) containing the segmentation assignment of each pixel
def meanshift_segmentation(im, features, bandwidth):
    H = im.shape[0]; 
    W = im.shape[1]; 
    num_pixels = features.shape[0]; 
    num_features = features.shape[1]
    centroids = np.empty((0,num_features))
    # Keep track of pixels to be tested
    pixels = np.arange(num_pixels)
    unseen = np.ones((num_pixels,),dtype=bool)
    # Until all pixels have been added to a cluster
    while(unseen.any()):
        # Randomly choose a *new* pixel to test
        pixel = np.random.choice(pixels[unseen])
        meanvec = features[pixel,:]
        oldmean = meanvec+bandwidth # Make sure to enter loop once
        while(np.linalg.norm(meanvec-oldmean)>0.01*bandwidth):
            oldmean = meanvec
            # Find all other pixels within bandwith distance of mean
            sqdist = np.sum(np.square(meanvec-features),axis=1)
            withinbw = np.array(sqdist<bandwidth**2)
            # Record that certain pixels have been clustered already
            unseen[withinbw] = False
            # Re-compute centroid of mean vector
            meanvec = np.mean(features[withinbw,:],axis=0)
        # Compare meanvec to others and merge if necessary
        newcluster = np.sum(np.square(centroids-meanvec),axis=1)
        newcluster = newcluster<(bandwidth/2)**2
        if(~newcluster.any()):
            meanvec = np.reshape(meanvec,(1,num_features))
            centroids = np.vstack((centroids,meanvec))
    # Vectorized: Assign all points to their nearest cluster by creating matrices of (npixels,nfeatures,nclusters)
    centroids = np.reshape(centroids,(centroids.shape[0],centroids.shape[1],1))
    centroids = np.swapaxes(centroids,0,2)
    centroids_mat = np.tile(centroids,(num_pixels,1,1))
    features = np.reshape(features,(features.shape[0],features.shape[1],1))
    features_mat = np.tile(features,(1,1,centroids_mat.shape[2]))
    # Assign based off of minimum euclidean distance (same as min of squared euclidean)
    distances = np.sum(np.square(features_mat-centroids_mat),axis=1)
    cluster_assignments = np.argmin(distances,axis=1)
    return np.reshape(cluster_assignments,(H,W))

if __name__ == '__main__':
    
    # Change these parameters to find favorite parameter
    bandwidths = [0.15, 0.20]

    for filename in ['smoke1','smoke2','smoke3','smoke4']:
        # Load image from file, resize to speed up segmentation, and plot it
        img = imread('data/%s.jpeg' % filename) 
        img = cv2.resize(img, (img.shape[1]/4,img.shape[0]/4))
        plt.subplot(221), plt.imshow(img)
        plt.title('Original'), plt.xticks([]), plt.yticks([])

        # Create the feature vector for the images using texture (various nearby RGB)
        features = get_segmentation_features(img)

        # Segmentation using meanshift
        for bandwidth in bandwidths:
            clustered_pixels = meanshift_segmentation(img, features, bandwidth)
            plt.subplot(223), plt.imshow(255.0*clustered_pixels/np.max(clustered_pixels))
            plt.title('Segmentation'), plt.xticks([]), plt.yticks([])
            plt.show()

            harris_edges = harris_smoke(img, clustered_pixels)
            canny_edges = canny_smoke(img, clustered_pixels)

            silhouette = np.zeros(clustered_pixels.shape)
            for clust in np.arange(harris_clusters.shape[0]):
                print clust
                silhouette = np.logical_or(clustered_pixels==harris_clusters[clust],silhouette)
            # mask = np.uint8(np.reshape(silhouette,(silhouette.shape[0],silhouette.shape[1],1)))
            plt.subplot(223), plt.imshow(silhouette*255) 
            plt.title('Harris-voted smoke'), plt.xticks([]), plt.yticks([])
            silhouette = np.zeros(clustered_pixels.shape)
            for clust in np.arange(canny_clusters.shape[0]):
                print clust
                silhouette = np.logical_or(clustered_pixels==canny_clusters[clust],silhouette)
            # mask = np.uint8(np.reshape(silhouette,(silhouette.shape[0],silhouette.shape[1],1)))
            plt.subplot(224), plt.imshow(silhouette*255) 
            plt.title('Canny-voted smoke'), plt.xticks([]), plt.yticks([])

