""" python splitstitched.py <trial-num> <letter> <pixels-off-top>

TODO 

"""

import os
import sys
import getpass

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

orig_fname = 'trial_%s_%s' % (trial_num, letter)
path_to_image = os.path.join(data_dir, '%s.tif' % orig_fname)
tif_image = Image.open(path_to_image)
img = np.array(tif_image)
img_height = img.shape[0]
img_width = img.shape[1]

# We'll save the split images in their own directory.
out_path = os.path.join(splitdata_dir, orig_fname)

# Make sure output path exists.
if not os.path.exists(out_path): os.makedirs(out_path)

# Write a log.
f = open(os.path.join(out_path, 'README.txt'), 'w')
f.write('Directory created by %s by running:\n' % getpass.getuser())
f.write('python splitstitched.py %s %s %i' % (trial_num, letter,
    pixels_off_top))
f.close()

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
    Image.fromarray(img_subset).save(os.path.join(out_path, '%s_%i.tif' %
        (orig_fname, i)))

