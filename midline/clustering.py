from sklearn.cluster import KMeans, AffinityPropagation, MeanShift
import numpy as np
import pylab as pl
from scipy import ndimage
from skimage import filter

# Open, and convert the image to grayscale.
A = rgb2gray(pl.imread('the_letter_s.png'))
pl.imshow(A, cmap=pl.get_cmap('gray'))
#pl.imshow(ndimage.median_filter(A, 30), cmap=pl.get_cmap('pink'))
pl.imshow(filter.canny(A, 3), cmap=pl.get_cmap('pink'))

X = np.loadtxt('the_letter_s.txt')

def cluster_and_plot(algo, *args, **kwargs):
    algo.fit(X)
    centroids = algo.cluster_centers_
    pl.plot(centroids[:, 1], centroids[:, 0], *args, **kwargs)

cluster_and_plot(KMeans(init='random', n_clusters=10), 'r.')
#cluster_and_plot(MeanShift(), 'b.')

pl.show()

