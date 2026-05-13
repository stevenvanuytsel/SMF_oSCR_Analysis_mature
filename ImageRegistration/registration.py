# %%
import os
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation
from skimage import io
import scipy.ndimage
import numpy as np
import pandas as pd
%matplotlib qt
# %%
regpath = os.path.join('..', '..', 'registry.tif')
regstack = io.imread(regpath)

# There are 200 frames, we take the final 10 as everything has stabilised then
regframe = np.median(regstack[-10:], axis=0)

rows=61

# Separate the two channels.
green = regframe[1: rows]
red = regframe[-rows+1:]

green = green/np.max(green)
red = red/np.max(red)

# Find shift between both images, green is reference, accuracy 0.01 px
shift, error, diffphase = phase_cross_correlation(green, red, upsample_factor=100)
registration = pd.DataFrame({'shift_y': shift[0], 'shift_x':shift[1], 'error': error, 'diffphase':diffphase, 'upsample_factor':100, 'rows':rows},index=[0])
registration.to_csv('registration_params.csv')
print(shift, error)
red_aligned = scipy.ndimage.shift(red, shift, order=1, prefilter=False, mode='nearest')

# # Plot the overlay of the graticule and the ratio of the two images
fig, (ax1, ax2) = plt.subplots(1,2)

ax1.imshow(green, cmap='Greens', alpha=1)
ax1.imshow(red, cmap='Reds', alpha=0.5)
ax1.set_title('Unregistered', fontsize=18, fontname='Arial', fontweight='black')

ax2.imshow(green, cmap='Greens', alpha=1)
ax2.imshow(red_aligned, cmap='Reds', alpha=0.5)
ax2.set_title('Registered', fontsize=18, fontname='Arial', fontweight='black')

fig.savefig('graticule_after_registration_0.01px_acc.png')


# %%
# Sanity check
print(phase_cross_correlation(green, red_aligned, upsample_factor=1)[0])