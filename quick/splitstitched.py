""" python splitstitched.py <trial-num> <letter> <pixels-off-top>

TODO 

"""

import os
import sys

import numpy as np
import Image

trial_num = sys.argv[1]
letter = sys.argv[2]
pixels_off_top = int(sys.argv[3])

# Dimensions of each output image.
output_height = 1500
output_width = 3000

data_dir = os.path.join(os.environ['PAPER_PHONOGRAPH_DROPBOX'], 'data',
        'microscopy-stitched')
splitdata_dir = os.path.join(os.environ['PAPER_PHONOGRAPH_DROPBOX'], 'data',
    'microscopy-split')

path_to_image = os.path.join(data_dir, 'trial_%s_%s.tif' % (trial_num, letter))
img = np.array(Image.open(path_to_image))
img_height = img.shape[0]
img_width = img.shape[1]

for i in range(int(img_width / output_width)):
    # Row indices.
    r_min = pixels_off_top
    r_max = pixels_off_top + output_height
    # Column indices.
    c_min = i * output_width
    c_max = (i + 1) * output_width
    
    # Get subset of image.
    img_subset = img[r_min:r_max, c_min:c_max]

    # Save to file.
    Image.fromarray(img_subset).save(os.path.join(splitdata_dir,
        'trial_%s_%s_%i.tif' % (trial_num, letter, i)))

