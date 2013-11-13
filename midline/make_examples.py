import numpy as np
import pylab as pl

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

X = np.empty((0, 2))
(nr, nc) = A.shape
for i in range(nr):
    for j in range(nc):
        if A[i, j] < 1.0:
            X = np.append(X, [[i, j]], axis=0)

np.savetxt('the_letter_s.txt', X)
