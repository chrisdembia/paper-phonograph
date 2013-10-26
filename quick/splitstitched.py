"""Interactively select how to split up a stitched image.

python splitstitched.py <trial-num> <letter>

For example:
    python splitstitched.py 49 C

"""

import os
import sys
import getpass

import numpy as np
import pylab as pl
from matplotlib import pyplot
import Image

trial_num = sys.argv[1]
letter = sys.argv[2]

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
f.write('python splitstitched.py %s %s\n' % (trial_num, letter))
f.write('Mouse inputs:\n')

fig = pl.figure()
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

# In the images shown in the interactive window, decrease resolution by the
# following integer factor:
dec_res = 20

num_images = int(img_width / output_width)
for i in range(num_images):

    # Column indices.
    c_min = i * output_width
    c_max = (i + 1) * output_width

    # Draw a slice of the image, and ask the user which subset to crop out.
    ax.clear()
    ax.set_title('Click to define top edge of the to-be-cropped-out '
            'image for this slice (%i/%i).' % (i + 1, num_images))
    ax.imshow(img[::dec_res, c_min:c_max:dec_res])
    pl.xticks([])
    pl.yticks([])
    # Need this so that the plot is updated IMMEDIATELY:
    pl.draw()
    coord = pl.ginput()[0]
    pixels_off_top = np.floor(coord[1] * dec_res)

    f.write('%i %i\n' % (i, pixels_off_top))

    # Row indices.
    r_min = pixels_off_top
    r_max = min(img_height, r_min + output_height)

    # Get subset of image.
    img_subset = img[r_min:r_max, c_min:c_max]

    # Show cropped image, but at a lower resolution for speed.
    ax2.clear()
    ax2.set_title('Previous image (%i), after cropping:' % (i + 1))
    ax2.imshow(img_subset[::dec_res, ::dec_res])
    pl.xticks([])
    pl.yticks([])
    # Need this so that the plot is updated IMMEDIATELY:
    pl.draw()

    # Save to file.
    Image.fromarray(img_subset).save(os.path.join(out_path, '%s_%i.tif' %
        (orig_fname, i)))

ax.clear()
ax2.clear()
ax.set_title('All done! You can close the plot window now.')
pl.draw()

pl.show()
f.close()
