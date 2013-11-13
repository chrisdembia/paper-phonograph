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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Binarizer
from sklearn import cross_validation


# Data input.
# -----------
class_names = ['ambient', '300', '424', '1000', '1414']

data_dir = os.path.join(os.environ['PAPER_PHONOGRAPH_DROPBOX'], 'data',
        'microscopy-split')

data = {
        'trial_19_D/trial_19_D_0.tif': 0,  
        'trial_19_D/trial_19_D_1.tif': 0,  
        'trial_19_D/trial_19_D_2.tif': 0,  
        'trial_19_D/trial_19_D_3.tif': 0,  
        'trial_19_D/trial_19_D_4.tif': 0,  
        'trial_19_D/trial_19_D_5.tif': 0,  
        'trial_19_D/trial_19_D_6.tif': 0,  
        'trial_19_D/trial_19_D_7.tif': 0,  
        'trial_19_D/trial_19_D_8.tif': 0,  
        'trial_19_D/trial_19_D_9.tif': 0,  
        'trial_19_D/trial_19_D_10.tif': 0, 
        'trial_19_D/trial_19_D_11.tif': 0, 
        'trial_19_D/trial_19_D_12.tif': 0, 
        'trial_19_D/trial_19_D_13.tif': 0, 
        'trial_19_D/trial_19_D_14.tif': 0, 
        'trial_19_D/trial_19_D_15.tif': 0, 
        'trial_19_D/trial_19_D_16.tif': 0, 
        'trial_19_D/trial_19_D_17.tif': 0, 
        'trial_19_D/trial_19_D_18.tif': 0, 
        'trial_19_D/trial_19_D_19.tif': 0, 
        'trial_19_D/trial_19_D_20.tif': 0, 
        'trial_19_D/trial_19_D_21.tif': 0, 
        'trial_19_D/trial_19_D_22.tif': 0, 
        'trial_21_C/trial_21_C_0.tif': 1,  
        'trial_21_C/trial_21_C_1.tif': 1,  
        'trial_21_C/trial_21_C_2.tif': 1,  
        'trial_21_C/trial_21_C_3.tif': 1,  
        'trial_21_C/trial_21_C_4.tif': 1,  
        'trial_21_C/trial_21_C_5.tif': 1,  
        'trial_21_C/trial_21_C_6.tif': 1,  
        'trial_21_C/trial_21_C_7.tif': 1,  
        'trial_21_C/trial_21_C_8.tif': 1,  
        'trial_21_C/trial_21_C_9.tif': 1,  
        'trial_21_C/trial_21_C_10.tif': 1, 
        'trial_21_C/trial_21_C_11.tif': 1, 
        'trial_21_C/trial_21_C_12.tif': 1, 
        'trial_21_C/trial_21_C_13.tif': 1, 
        'trial_21_C/trial_21_C_14.tif': 1, 
        'trial_21_C/trial_21_C_15.tif': 1, 
        'trial_21_C/trial_21_C_16.tif': 1, 
        'trial_21_C/trial_21_C_17.tif': 1, 
        'trial_21_C/trial_21_C_18.tif': 1, 
        'trial_21_C/trial_21_C_19.tif': 1, 
        'trial_21_C/trial_21_C_20.tif': 1, 
        'trial_21_C/trial_21_C_21.tif': 1, 
        'trial_21_C/trial_21_C_22.tif': 1, 
        'trial_21_C/trial_21_C_23.tif': 1, 
        'trial_21_C/trial_21_C_24.tif': 1, 
        'trial_21_C/trial_21_C_25.tif': 1, 
        }

all_image_fnames = data.keys()
y = np.array(data.values())

#binarizer = Binarizer(threshold=0.3)
def prepare_X(image_fnames):
    # Each element is an r x c image.
    images_tuple = tuple(np.array(Image.open(os.path.join(data_dir, fname)))
                for fname in image_fnames)
    # If n is the number of images (length of this tuple), then the next line
    # creates a matrix that goes through the following shapes:
    # (r x c x n) --> (n x c x r) --> (n x r x c)
    images = np.dstack(images_tuple).swapaxes(0, 2).swapaxes(1, 2)
    # Convert the data into a format that the algorithms prefer:
    # http://scikit-learn.org/stable/datasets/#sample-images
    images_floating_point = np.array(images, dtype=np.float64) / 255.
    n_images = len(image_fnames)
    images_2d = images_floating_point.reshape((n_images, -1))
    # Binarization took a long time.
    # X_transformed = binarizer.transform(images_floating_point)
    X_transformed = (images_2d > 0.3).astype(float)
    return X_transformed


# Prepare data.
#X_train = prepare_X(training_images_fnames)
#X_test = prepare_X(test_images_fnames)
X = prepare_X(all_image_fnames)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.4, random_state=0)

def quick_classification_test(name, classifier):
    print '\nClassifying using %s...' % name
    print '==================================================================='

    # Train.
    classifier.fit(X_train, y_train)
    # Test.
    y_predicted = classifier.predict(X_test)

    # Report results.
    # ---------------
    #print "Actual:\n", y_test
    #print "Prediction:\n", y_predicted
    #print "Classification report:\n", metrics.classification_report(y_test,
    #        y_predicted)
    #print "Confusion matrix:\n", metrics.confusion_matrix(y_test, y_predicted)
    err = sum(np.array(y_test) != y_predicted) / float(len(y_predicted))
    print "Error:", err

def cross_validate(name, classifier):
    cv = cross_validation.LeaveOneOut(len(y))
    scores = cross_validation.cross_val_score(classifier, X, y, cv=5)
    print '\nCross-validating using %s...' % name
    print '==================================================================='
    print 'Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std())


#cross_validate('SVM', svm.SVC())
#cross_validate('SVM gamma = 0.001', svm.SVC(gamma=0.001))
#cross_validate('SVM gamma = 0.01', svm.SVC(gamma=0.01))
#cross_validate('GaussianNB', GaussianNB())
cross_validate('K nearest neighbors', KNeighborsClassifier(n_neighbors=6))



