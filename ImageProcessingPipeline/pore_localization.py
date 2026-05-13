# %%
# File handling
from glob import glob
import os

# I/O
from skimage import io, feature

# Processing
import pandas as pd
import time
import numpy as np
from scipy.signal import fftconvolve

from scipy.optimize import curve_fit
import napari

# Visualization
from matplotlib import rcParams
import matplotlib.pyplot as plt
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except Exception:
    pass
plt.style.reload_library()
plt.style.use('2025VanuytselNRJ')

# %%
def moffat_kernel(alpha_pix, beta, size=21):
    assert size %2 ==1 # Size is odd
    half = size//2
    y, x = np.mgrid[-half:half+1, -half:half+1]
    r2 = x**2 + y**2

    k = (1+(r2/alpha_pix**2))**(-beta)
    k /= k.sum() # normalize

    return k

def extract_patch(image, x, y, radius):
    x = int(round(x))
    y = int(round(y))
    if (
        x - radius < 0 or y - radius < 0 or
        x + radius >= image.shape[1] or
        y + radius >= image.shape[0]
    ):
        return None

    return image[y - radius : y + radius + 1, x - radius : x + radius + 1].copy()


def moffat_1d(x, amp, x0, alpha, beta, bkg):
    rr = (x-x0)**2
    return amp*(1+(rr/alpha**2))**(-beta)+bkg

def moffat_2d(xy, amp, x0, y0, alpha, beta, bkg):
    x, y = xy
    rr = ((x - x0)**2 + (y - y0)**2)
    return amp * (1 + (rr / alpha**2))**(-beta)+ bkg


def moffat_2d_flat(xy, amp, x0, y0, alpha, beta, bkg):
    return moffat_2d(xy, amp, x0, y0, alpha, beta, bkg).ravel()

def fit_2d_moffat(patch, alpha_pix_init=None, beta_init=None, bkg_init = None, bounds=None):
    h, w = patch.shape
    y, x = np.mgrid[0:h, 0:w]
    
    amp_init = patch.max() - patch.min()
    x0_init = w / 2
    y0_init = h / 2

    if alpha_pix_init is None:
        alpha_init = w / 4
    else:
        alpha_init = alpha_pix_init
    if beta_init is None:
        beta_init = 2.0  # Typical for PSFs
    else:
        beta_init = beta_init

    if bkg_init is None:
        bkg_init = patch.min()
    else:
        bkg_init = bkg_init

    p0 = [amp_init, x0_init, y0_init, alpha_init, beta_init, bkg_init]
    if bounds is None:
        bounds = ([0, 0, 0, 0, 0, 0], [np.inf]*len(p0))
    else:
        bounds = bounds
    
    try:
        popt, pcov = curve_fit(moffat_2d_flat, (x, y), patch.ravel(), p0=p0, bounds=bounds)
        y_fit = moffat_2d_flat((x, y), *popt)
        ss_res = np.sum((patch.ravel()-y_fit)**2)
        ss_tot = np.sum((patch.ravel()-np.mean(patch))**2)
        r2 = 1-ss_res/ss_tot
        return popt, pcov, r2  # amp, x0, y0, alpha, beta, bkg
    except RuntimeError:
        return None, None, None



# %%
start_global = time.perf_counter()

# %%
mpp = 10/29 #um

# %%
# Load in image stack and set up saving
datafolder = 'cleaned_data'
droplet = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
expname = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

savename = f'{expname}_{droplet}'
outdir = 'pore_localization'
os.makedirs(outdir, exist_ok=True)

# %%
stack = glob(os.path.join(f'{datafolder}', '*green.tif'))[0]
stack = io.imread(stack)

# %%
# First attempt to convolve with Moffat kernel
alpha_pix = 1.7/mpp # Found from the fits of the PSF
beta = 1.22

# Fit bounds for later
alpha_bound = 1e-6
beta_bound = 1e-6
print(alpha_pix, alpha_bound)
print(beta, beta_bound)
# %%

moffat_fwhm_pix = 2*alpha_pix*np.sqrt(2**(1/beta)-1)
border = radius = int(moffat_fwhm_pix//2)
min_distance = int(0.6*moffat_fwhm_pix)

# %%

# Build kernel and set threshold
mkernel = moffat_kernel(alpha_pix, beta*1.2, 21)
pad_y, pad_x = mkernel.shape[0]//2, mkernel.shape[1]//2
k=1.0

h = w = 2*radius+1
yy_p, xx_p = np.mgrid[0:h, 0:w]

mask_circ = ((xx_p - radius)**2 + (yy_p - radius)**2) <= radius**2
N_pix_ap = int(mask_circ.sum())

start = time.perf_counter()
all_results = []
for t in range(len(stack)):
    print(t)
    
    f = stack[t].copy()
    padded = np.pad(f, ((pad_y, pad_y), (pad_x, pad_x)), mode='median').astype(float)
    convolved = fftconvolve(padded, mkernel, mode='same')
    convolved = convolved[pad_y:-pad_y, pad_x:-pad_x]
    bg = np.median(convolved)
    c0 = convolved-bg
    mad = np.median(np.abs(c0)) # already bg subtracted so mean bg should be 0
    sigma = 1.4826*mad
    threshold = k*sigma

    coords = feature.peak_local_max(c0, threshold_abs=threshold, exclude_border = border, min_distance=min_distance)

    # Fit the coords we found (bright peaks)
    good_fits = []
    frame_results = []
    for (y0, x0) in coords:
        patch = extract_patch(f, x0, y0, radius)
        if patch is None:
            continue
        H, W = f.shape

        # bg
        r_in = border*3
        r_out = border*4
        x1 = max(0, int(x0 - r_out))
        x2 = min(W, int(x0 + r_out + 1))
        y1 = max(0, int(y0 - r_out))
        y2 = min(H, int(y0 + r_out + 1))

        patch_bg = f[y1:y2, x1:x2]
        yy_loc, xx_loc = np.mgrid[y1:y2, x1:x2]
        r2_loc = (yy_loc - y0)**2 + (xx_loc - x0)**2
        mask_bg = (r2_loc >= r_in**2) & (r2_loc <= r_out**2)
        bg_vals = patch_bg[mask_bg] 

        # Take the 20-th percentile and find a good spread     
        bg_init = np.percentile(bg_vals, 20)
        sigma_bg = 1e-6

        # Define bounds and Fit
        fitting_bounds = ([0, 0, 0, alpha_pix-alpha_bound, beta-beta_bound, bg_init-sigma], [np.inf, np.inf, np.inf, alpha_pix+alpha_bound, beta+beta_bound, bg_init+sigma])
        popt_moffat, pcov_moffat, r2_moffat = fit_2d_moffat(patch, alpha_pix_init=alpha_pix, beta_init=beta, bkg_init=bg_init, bounds=fitting_bounds)
        
        # We only care about the pores that are fitted reasonably well in this first pass
        if r2_moffat<0.5:
            continue

        amp, x0_fit, y0_fit, alpha_fit, beta_fit, bkg_fit = popt_moffat
        perr = np.sqrt(np.diag(pcov_moffat))
        _, x0_err, y0_err, _, _, _ = perr

        # Convert subpixel locations to image coordinates
        x_fit_img = (x0-radius)+x0_fit
        y_fit_img = (y0-radius)+y0_fit

        # Centroiding for track linking
        centroid_patch = patch-bg_init
        centroid_patch[centroid_patch<0] = 0 # clip to zero
        sum_w = centroid_patch.sum()
        if sum_w>0:
            x_cent = (xx_p*centroid_patch).sum()/sum_w
            y_cent = (yy_p*centroid_patch).sum()/sum_w
        else:
            x_cent = w/2.0
            y_cent = h/2.0
        
        x_cent_imag = (x0-radius)+x_cent
        y_cent_imag = (y0-radius)+y_cent

        # Calculate intensities from the fit (not used normally)
        model_2d = moffat_2d_flat((xx_p, yy_p), *popt_moffat).reshape(h, w)
        max_pixel = patch.max()

        model_sum = model_2d[mask_circ].sum()
        raw_sum = patch[mask_circ].sum()
        N_pix_patch = N_pix_ap  
        frame_results.append([t, y0, x0, y_cent_imag, x_cent_imag, y_fit_img, x_fit_img, raw_sum, model_sum, max_pixel, amp, alpha_fit, beta_fit, bkg_fit, r2_moffat, N_pix_patch, y0_err, x0_err])
        good_fits.append([amp, x_fit_img, y_fit_img, alpha_fit, beta_fit, bkg_fit])

    # Change all found particles for background after model subtraction)
    model = np.zeros_like(f, dtype=float)
    r_excl = moffat_fwhm_pix
    if good_fits:
        yy_img, xx_img = np.mgrid[0:f.shape[0], 0:f.shape[1]]
        
        for _fit in good_fits:
            amp, x_fit_img, y_fit_img, alpha_fit, beta_fit, bkg_fit = _fit
            # modified alpha and beta a little to suppress leftover 
            model+=moffat_2d_flat((xx_img, yy_img), amp, x_fit_img, y_fit_img, alpha_fit*0.9, beta_fit*1.1, 0).reshape(f.shape) # Don't add background!
        # Subtract model from data
        residual = f-model
        
        # Set all known particles to their background
        res_clean = residual.copy()

        # inverted_moffat rather than hard mask
        for _fit in good_fits:
            amp, x_fit_img, y_fit_img, alpha_fit, beta_fit, bkg_fit = _fit
            r2 = (yy_img-y_fit_img)**2+(xx_img-x_fit_img)**2
            mask = r2<=r_excl**2
            res_clean[mask] = bkg_fit
    else:
        res_clean = f.copy()

    padded_r = np.pad(res_clean, ((pad_y, pad_y), (pad_x, pad_x)), mode='median')
    conv_r = fftconvolve(padded_r, mkernel, mode='same')
    conv_r = conv_r[pad_y:-pad_y, pad_x:-pad_x]

    bg_r = np.median(conv_r)
    c0_r = conv_r - bg_r
    mad_r = np.median(np.abs(c0_r))
    sigma_r = 1.4826 * mad_r
    threshold_r = k * sigma_r  # maybe lower than first pass

    coords_r = feature.peak_local_max(
        c0_r,
        threshold_abs=threshold_r,
        exclude_border=border,
        min_distance=min_distance,
    )

    # Discard all the peaks found within the excluded zone
    coords_r_filtered = []
    for (y, x) in coords_r:
        keep = True
        for amp, x_fit_img, y_fit_img, alpha_fit, beta_fit, bkg_fit in good_fits:
            if (y - y_fit_img)**2 + (x - x_fit_img)**2 <= r_excl**2:
                keep = False
                break
        if keep:
            coords_r_filtered.append([y, x])

    coords_r = np.array(coords_r_filtered)

    # Fit these newly foudn particles
    for (y0, x0) in coords_r:
        patch = extract_patch(f, x0, y0, radius)
        if patch is None:
            continue
        H, W = f.shape

        # bg
        r_in = border*3
        r_out = border*4
        x1 = max(0, int(x0 - r_out))
        x2 = min(W, int(x0 + r_out + 1))
        y1 = max(0, int(y0 - r_out))
        y2 = min(H, int(y0 + r_out + 1))

        patch_bg = f[y1:y2, x1:x2]
        yy_loc, xx_loc = np.mgrid[y1:y2, x1:x2]
        r2_loc = (yy_loc - y0)**2 + (xx_loc - x0)**2
        mask_bg = (r2_loc >= r_in**2) & (r2_loc <= r_out**2)
        bg_vals = patch_bg[mask_bg] 

        # Take the 20-th percentile and find a good spread     
        bg_init = np.percentile(bg_vals, 20)
        sigma_bg = 1e-6

        # Define bounds and Fit
        fitting_bounds = ([0, 0, 0, alpha_pix-alpha_bound, beta-beta_bound, bg_init-sigma], [np.inf, np.inf, np.inf, alpha_pix+alpha_bound, beta+beta_bound, bg_init+sigma])
        popt_moffat, pcov_moffat, r2_moffat = fit_2d_moffat(patch, alpha_pix_init=alpha_pix, beta_init=beta, bkg_init=bg_init, bounds=fitting_bounds)
        
        amp, x0_fit, y0_fit, alpha_fit, beta_fit, bkg_fit = popt_moffat
        perr = np.sqrt(np.diag(pcov_moffat))
        _, x0_err, y0_err, _, _, _ = perr

        # Convert subpixel locations to image coordinates
        x_fit_img = (x0-radius)+x0_fit
        y_fit_img = (y0-radius)+y0_fit

        # Centroiding for track linking
        centroid_patch = patch-bg_init
        centroid_patch[centroid_patch<0] = 0 # clip to zero
        sum_w = centroid_patch.sum()
        if sum_w>0:
            x_cent = (xx_p*centroid_patch).sum()/sum_w
            y_cent = (yy_p*centroid_patch).sum()/sum_w
        else:
            x_cent = w/2.0
            y_cent = h/2.0
        
        x_cent_imag = (x0-radius)+x_cent
        y_cent_imag = (y0-radius)+y_cent

        # Calculate intensities from the fit (not used normally)
        model_2d = moffat_2d_flat((xx_p, yy_p), *popt_moffat).reshape(h, w)
        max_pixel = patch.max()

        model_sum = model_2d[mask_circ].sum()
        raw_sum = patch[mask_circ].sum()
        N_pix_patch = N_pix_ap  
        frame_results.append([t, y0, x0, y_cent_imag, x_cent_imag, y_fit_img, x_fit_img, raw_sum, model_sum, max_pixel, amp, alpha_fit, beta_fit, bkg_fit, r2_moffat, N_pix_patch, y0_err, x0_err])
    all_results.extend(frame_results)
end = time.perf_counter()
print(f'Localization took {end-start}s')
# %%
results = np.array(all_results)
cols = [
    "frame",
    "y_init", "x_init",
    "y_centr", "x_centr",
    "y_fit", "x_fit",
    "raw_sum",
    "fit_sum",
    "max_raw_pixel",
    "amp_fit",
    "alpha_fix",
    "beta_fix",
    "bkg_fix",
    "r2_moffat",
    "N_pix_patch",
    "y_fit_err",
    "x_fit_err",
]

df = pd.DataFrame(results, columns=cols)
df.to_csv(os.path.join(outdir, f"{savename}_pore_localization.csv"), index=False)

# %%
# # Visualize all features that were found
# # locs: columns [frame, y, x]
# frames = np.asarray(results[:, 0], int)
# ys     = np.asarray(results[:, 3], float)
# xs     = np.asarray(results[:, 4], float)

# # napari expects (t, y, x) for time series points
# points = np.column_stack([frames, ys, xs])

# viewer = napari.view_image(stack, name='data')

# viewer.add_points(
#     points,
#     name='localizations',
#     size=5,
#     face_color='translucent',
#     border_color='red',
#     border_width=0.05,
# )
# napari.run()



# %%
