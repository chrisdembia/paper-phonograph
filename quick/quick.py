"""Quick analysis of microscopy images, immediately after (during?) microscopy
session on 24 Oct 2013.

We use a support vector machine to predict frequency, using pixel intensities.
This is very similar to the scikit-learn tutorial on the 'digits' dataset.

"""

import os
import numpy
import Image
from pylab import imshow, gray

from sklearn import svm, metrics

# Data input.
# -----------
Y_train = [0, 0, 1]
Y_test = [0]

class_names = ['ambient', '300', '424', '1000', '1414']

data_dir = os.path.join(os.environ['PAPER_PHONOGRAPH_DROPBOX'], 'data',
        'microscopy-split')

# TODO imshow(os.path.join(data_dir, 'trial_49_C.tif'))
training_images_fnames = [
        'trial_49_C_0.tif',
        'trial_49_C_1.tif',
        'trial_49_C_2.tif',
        ]
test_images_fnames = [
        'trial_49_C_3.tif',
        ]

def prepare_X(image_fnames):
    # Each element is an r x c image.
    images_tuple = tuple( np.array(Image.open(os.path.join(data_dir, fname)))
                for fname in image_fnames)
    # If n is the number of images (length of this tuple), then the next line
    # creates a matrix that goes through the following shapes:
    # (r x c x n) --> (n x c x r) --> (n x r x c)
    images = np.dstack(images_tuple).swapaxes(0, 2).swapaxes(1, 2)
    n_images = len(image_fnames)
    return images.reshape((n_images, -1))


# Train.
# ------
X_train = prepare_X(training_images_fnames)
classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train, Y_train)


# Test.
# -----
X_test = prepare_X(test_images_fnames)
Y_predicted = classifier.predict(X_test)


# Report results.
# ---------------
print "Prediction:\n", Y_predicted
print "Classification report:\n", metrics.classification_report(Y_test,
        Y_predicted)
print "Confusion matrix:\n", metrics.confusion_matrix(Y_test, Y_predicted)
