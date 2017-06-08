import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.misc import imread
from scipy.spatial.distance import cdist
import cv2

def find_shadows(img, clustered_img):
    num_clusters = np.max(clustered_img)
    means = np.zeros((num_clusters,1))

    for cluster in np.arange(num_clusters):
        cluster_pixels = (clustered_img==cluster)
        incluster = np.zeros((cluster_pixels.shape[0],cluster_pixels.shape[1],3))
        incluster[:,:,0] = cluster_pixels
        incluster[:,:,1] = cluster_pixels
        incluster[:,:,2] = cluster_pixels
        meanval = np.sum(img*incluster)/np.sum(incluster)
        means[cluster] = meanval

    threshold = 130
    print means
    return means>threshold


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

    num_features = 5
    features = np.zeros((img.shape[0] * img.shape[1], num_features))

    for row in xrange(img.shape[0]):
        for col in xrange(img.shape[1]):
            features[row*img.shape[1] + col, :] = np.array([img_bilat[row, col, 0], img_bilat[row, col, 1], 
                img_bilat[row, col, 2], row, col])

    norms = features.max(axis=0)
    # Normalize to weigh all features equally
    features_normalized = features / norms

    return features_normalized

# RECEIVES: the original RGB image and an image of the same size with segmentation region assignments
# RETURNS: array of size (nclusters,1) containing the ratio of harris corner pixels to total pixels in the boundary of each segment
def harris_smoke(img, clustered_img):
    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bw = np.float32(bw)
    num_clusters = np.max(clustered_img)
    ratios = np.zeros((num_clusters,1))

    # Parameters for Harris
    blockSize = 6
    ksize = 3
    k = 0.01

    # Kernel for dilation and erosion
    kernel = np.ones((5,5), np.uint8)

    # Find Harris corners
    har = cv2.cornerHarris(bw,blockSize,ksize,k)
    # har is array of size (H,W) of type float 32 from 1e-8 to 1e08
    
    for cluster in np.arange(num_clusters):
        cluster_pixels = (clustered_img==cluster)
        # Convert to float for dilate operation
        cluster_pixels = np.float32(cluster_pixels)
        # Eliminate noisy pixels that stand on their own
        cluster_pixels = 1-(cv2.dilate(1-cluster_pixels,kernel))

        dilated_pixels = cv2.dilate(cluster_pixels,kernel,iterations = 2)
        eroded_pixels = cv2.erode(cluster_pixels,kernel)
        # Change data type back to boolean
        border_pixels = (dilated_pixels-eroded_pixels)
        
        # Threshold imposition
        seg_har = np.logical_and(har>0.00001*har.max(),border_pixels)
        corners = np.sum(seg_har)
        tot = np.sum(border_pixels)
        ratio = np.double(corners)/tot

        ratios[cluster] = ratio

    return ratios

# RECEIVES: the original RGB image and an image of the same size with segmentation region assignments
# RETURNS: array of size (nclusters,1) containing the ratio of canny edge pixels to total pixels in the boundary of each segment
def canny_smoke(img, clustered_img):
    bw = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bw = np.float32(bw)
    num_clusters = np.max(clustered_img)
    ratios = np.zeros((num_clusters,1))


    # Find Canny edges
    thresh1 = 300
    thresh2 = 400
    edges = cv2.Canny(img,thresh1,thresh2)

    # Kernel for dilation and erosion
    kernel = np.ones((5,5), np.uint8)
    
    for cluster in np.arange(num_clusters):
        cluster_pixels = (clustered_img==cluster)
        # Convert to float for dilate operation
        cluster_pixels = np.float32(cluster_pixels)

        # Eliminate noisy pixels that stand on their own
        cluster_pixels = 1-(cv2.dilate(1-cluster_pixels,kernel))

        dilated_pixels = cv2.dilate(cluster_pixels,kernel,iterations = 2)
        eroded_pixels = cv2.erode(cluster_pixels,kernel)
        # Change data type back to boolean
        border_pixels = (dilated_pixels-eroded_pixels)
        
        # Threshold imposition
        seg_can = np.logical_and(edges/255==1,border_pixels)
        edgecount = np.sum(seg_can)
        tot = np.sum(border_pixels)
        ratio = np.double(edgecount)/tot

        plt.subplot(221), plt.imshow(img), 
        plt.title('Original'), plt.xticks([]), plt.yticks([])
        plt.subplot(222), plt.imshow(255.0*clustered_img/np.max(clustered_img))
        plt.title('Segmentation'), plt.xticks([]), plt.yticks([])
        plt.subplot(223), plt.imshow(edges), 
        plt.title('Canny Edges'), plt.xticks([]), plt.yticks([])
        plt.subplot(224), plt.imshow(border_pixels - 0.5*seg_can)
        plt.title('Edges on Boundary'), plt.xticks([]), plt.yticks([])

        ratios[cluster] = ratio

    return ratios

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
    # 0.10 seems good for no x, y in feature vector
    # 0.25 seems good for rgb of bilateral image and x and y
    bandwidths = [0.25]

    for filename in ['smoke1','smoke2','smoke3','smoke4']:
        print filename
        # Load image from file, resize to speed up segmentation, and plot it
        img = imread('data/%s.jpeg' % filename) 
        img = cv2.resize(img, (img.shape[1]/4,img.shape[0]/4))

        # Create the feature vector for the images using texture (various nearby RGB)
        features = get_segmentation_features(img)

        for bandwidth in bandwidths:
            # Segmentation using meanshift algorithm
            clustered_pixels = meanshift_segmentation(img, features, bandwidth)

            # Find frequency of edges and corners in border area of each segment
            clustercorner_ratios = harris_smoke(img, clustered_pixels)
            clusteredge_ratios = canny_smoke(img, clustered_pixels)

            # Find and eliminate dark segments (shadow filter)
            shadows = find_shadows(img, clustered_pixels)

            # Combine harris, canny, shadow for final answer
            smoke = np.logical_and(np.logical_and(clustercorner_ratios>0.2,clustercorner_ratios<0.6),
                np.logical_and(clusteredge_ratios<0.08,shadows))
            smoke = clustered_pixels==np.argmax(smoke)

            # Eliminate noisy pixels
            kernel = np.ones((5,5), np.uint8)
            final = 1-(cv2.dilate(1-np.float32(smoke),kernel))

            # Display Results
            print "Harris"
            print clustercorner_ratios
            print "Canny"
            print clusteredge_ratios
            plt.subplot(224), plt.imshow(final,cmap='gray'), plt.title('Final Results')
            plt.xticks([]), plt.yticks([])
            plt.show()



