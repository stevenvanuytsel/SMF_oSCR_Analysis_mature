# %%
import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from skimage import io
from collections import defaultdict
import time

import matplotlib.pyplot as plt
import matplotlib
import napari
try:
    get_ipython().run_line_magic('matplotlib', 'qt')
except Exception:
    pass

matplotlib.style.reload_library()
plt.style.use('2025VanuytselNRJ')

# %%
def moffat_1d(x, amp, x0, alpha, beta, bkg):
    rr = (x-x0)**2
    return amp*(1+(rr/alpha**2))**(-beta)+bkg

def moffat_2d(xy, amp, x0, y0, alpha, beta, bkg):
    x, y = xy
    rr = ((x - x0)**2 + (y - y0)**2)
    return amp * (1 + (rr / alpha**2))**(-beta)+ bkg


def moffat_2d_flat(xy, amp, x0, y0, alpha, beta, bkg):
    return moffat_2d(xy, amp, x0, y0, alpha, beta, bkg).ravel()

def fit_2d_moffat(patch):
    h, w = patch.shape
    y, x = np.mgrid[0:h, 0:w]
    
    amp_init = patch.max() - patch.min()
    x0_init = w / 2
    y0_init = h / 2
    alpha_init = w / 4
    beta_init = 2.0  # Typical for PSFs
    bkg_init = patch.min()

    p0 = [amp_init, x0_init, y0_init, alpha_init, beta_init, bkg_init]
    
    try:
        popt, pcov = curve_fit(moffat_2d_flat, (x, y), patch.ravel(), p0=p0)
        y_fit = moffat_2d_flat((x, y), *popt)
        ss_res = np.sum((patch.ravel()-y_fit)**2)
        ss_tot = np.sum((patch.ravel()-np.mean(patch))**2)
        r2 = 1-ss_res/ss_tot
        return popt, pcov, r2  # amp, x0, y0, alpha, beta, bkg
    except RuntimeError:
        return None, None, None

def gaussian_1d(x, amp, x0, sigma, bkg):
    rr = (x-x0)**2
    return amp*np.exp(-rr/(2*sigma**2))+bkg

def gaussian_2d(xy, amp, x0, y0, sigma, bkg):
    x, y = xy
    rr = ((x-x0)**2 + (y-y0)**2)
    return amp*np.exp(-rr/(2*sigma**2))+bkg

def gaussian_2d_flat(xy, amp, x0, y0, sigma, bkg):
    return gaussian_2d(xy, amp, x0, y0, sigma, bkg).ravel()

def fit_2d_gaussian(patch):
    h, w = patch.shape
    y, x = np.mgrid[0:h, 0:w]
    
    amp_init = patch.max() - patch.min()
    x0_init = w / 2
    y0_init = h / 2
    sigma_init = min(h, w)/4.0
    bkg_init = patch.min()

    p0 = [amp_init, x0_init, y0_init, sigma_init, bkg_init]
    
    try:
        popt, pcov = curve_fit(gaussian_2d_flat, (x, y), patch.ravel(), p0=p0)
        y_fit = gaussian_2d_flat((x, y), *popt)
        ss_res = np.sum((patch.ravel()-y_fit)**2)
        ss_tot = np.sum((patch.ravel()-np.mean(patch))**2)
        r2 = 1-ss_res/ss_tot
        return popt, pcov, r2  # amp, x0, y0, sigma, bkg
    except RuntimeError:
        return None, None, None

# %%
mpp = 13/29 # um

# %%
# Open pore
datasets = ['001', '003', '006', '007']
startframes = [[2938, 4899, 5041], [1486, 2290, 3740], [1, 764], [1468, 1500, 1535, 1605]] # FIJI indexed (start at 1 so subtract 1!)
endframes = [[3027, 5024, 5150], [1500, 2421, 3880], [44, 961], [1481, 1520, 1561, 1699]]
framerate = 485.44 # Hz
per_dataset = {
    ds: {
        "interval_counts": [e - s + 1 for s, e in zip(s_list, e_list)],
        "total": sum(e - s + 1 for s, e in zip(s_list, e_list)),
    }
    for ds, s_list, e_list in zip(datasets, startframes, endframes)
}

grand_total = sum(d["total"] for d in per_dataset.values())
print(grand_total)

# %%

sigma_small = 3
sigma_large = 15

half_w = 18

data_dict = defaultdict(list)
plotted = False

for idx, d in enumerate(datasets):
    print(f'dataset: {d}')
    stack = io.imread(os.path.join(f'D:/KCL_PhD_THESIS/R9.4.1_simulmeas/32bp_Cy5_32bp_BHQ2_0.1uM/20211010/data_cleaning+tracking/droplet_1_{d}_-100mV/cleaned_data/20211010_droplet_1_{d}_-100mV_green.tif'))
    for jdx, f in enumerate(startframes[idx]):
        print(f'frame: {jdx}')
        substack = stack[f-1:endframes[idx][jdx-1+1]].copy()

        for zdx, frame in enumerate(substack):
            print(f'frame: {zdx}')
            frame_small = gaussian_filter(frame.astype(np.float32), sigma = (sigma_small, sigma_small))
            frame_large = gaussian_filter(frame.astype(np.float32), sigma = (sigma_large, sigma_large))
            frame_dog = frame_small-frame_large
            frame_dog = np.clip(frame_dog, 0, None)
            flat_idx = np.argmax(frame_dog)
            y, x = np.unravel_index(flat_idx, frame.shape)
            
            roi = frame[y-half_w:y+half_w+1, x-half_w:x+half_w+1].copy()

            popt_moffat, pcov_moffat, r2_moffat = fit_2d_moffat(roi)
            perr_moffat = np.diag(pcov_moffat)
            popt_gaussian, pcov_gaussian, r2_gaussian = fit_2d_gaussian(roi)
            perr_gaussian = np.diag(pcov_gaussian)

            # Save the data
            data_dict['moffat_alpha_pix'].append(popt_moffat[3])
            data_dict['moffat_beta'].append(popt_moffat[4])
            data_dict['moffat_alpha_var_pix'].append(perr_moffat[3])
            data_dict['moffat_beta_var'].append(perr_moffat[4])
            data_dict['moffat_fwhm_pix'].append(2*popt_moffat[3]*np.sqrt(2**(1/popt_moffat[4])-1))
            data_dict['moffat_fwhm_um'].append(2*popt_moffat[3]*np.sqrt(2**(1/popt_moffat[4])-1)*mpp)
            data_dict['moffat_r2'].append(r2_moffat)

            data_dict['gaussian_sigma'].append(popt_gaussian[3])
            data_dict['gaussian_fwhm_pix'].append(2*np.sqrt(2*np.log2(2))*popt_gaussian[3])
            data_dict['gaussian_fwhm_um'].append(2*np.sqrt(2*np.log2(2))*popt_gaussian[3]*mpp)
            data_dict['gaussian_r2'].append(r2_gaussian)
            data_dict['moffat_sigma_var_pix'].append(perr_gaussian[3])

            # For plotting
            if jdx != 0 and r2_moffat > 0.97 and not plotted:
                print('Plotting')
                h, w = roi.shape

                center_y = h//2
                center_x = w //2

                data_cs = roi[center_y, :]

                # Define finer grid for y-coordinates to evaluate model on
                x_pix = np.arange(w, dtype=float) # We take the central row and all columns, so x
                x_fine = np.linspace(0, w-1, 10*(w-1)+1)

                # Shift so x0 is at 0
                x_pix_shifted_moffat = x_pix - popt_moffat[1]
                x_fine_shifted_moffat = x_fine - popt_moffat[1]

                x_pix_shifted_gaussian = x_pix - popt_gaussian[1]
                x_fine_shifted_gaussian = x_fine - popt_gaussian[1]

                model_moffat_1d = moffat_1d(x_fine, popt_moffat[0], popt_moffat[1], popt_moffat[3], popt_moffat[4], popt_moffat[5])
                model_gaussian_1d = gaussian_1d(x_fine, popt_gaussian[0], popt_gaussian[1], popt_gaussian[3], popt_gaussian[4])

                fig, ax = plt.subplots()
                ax.plot(x_pix_shifted_moffat*mpp, data_cs, ls='None', marker='.')
                ax.plot(x_fine_shifted_moffat*mpp, model_moffat_1d, ls='--')
                ax.set_xlabel(r'Distance ($\mathrm{\mu}$m)')
                ax.set_ylabel('F (counts)')
                fig.savefig('PSF_OpenPore_moffat_fit.svg', dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.plot(x_pix_shifted_gaussian*mpp, data_cs, ls='None', marker='.')
                ax.plot(x_fine_shifted_gaussian*mpp, model_gaussian_1d, ls='--')
                ax.set_xlabel(r'Distance ($\mathrm{\mu}$m)')
                ax.set_ylabel('F (counts)')
                fig.savefig('PSF_OpenPore_gaussian_fit.svg', dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.plot(x_pix_shifted_gaussian*mpp, data_cs, ls='None', marker='.')
                ax.plot(x_fine_shifted_gaussian*mpp, model_gaussian_1d, ls='--', label='Gaussian fit')
                ax.plot(x_fine_shifted_moffat*mpp, model_moffat_1d, ls='--', label='Moffat fit')
                ax.set_xlabel(r'Distance ($\mathrm{\mu}$m)')
                ax.set_ylabel('F (counts)')
                ax.legend(frameon=False)
                fig.savefig('PSF_OpenPore_gaussian_fit_moffat_fit.svg', dpi=300)
                plt.close(fig)    
                 
                # Make 3d plot
                Y, X = np.mgrid[0:h, 0:w]
                x_flat, y_flat = X.ravel(), Y.ravel()
                z_flat = np.zeros_like(x_flat)
                dz_flat = roi.ravel()
                dx=dy=1

                # Color
                norm = (dz_flat-dz_flat.min())/(dz_flat.max()-dz_flat.min()+1e-12)
                colors = matplotlib.cm.viridis(norm)

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.bar3d(x_flat, y_flat, z_flat, dx, dy, dz_flat, shade=False, color=colors, ec='black', linewidth=0.5)
                ax.view_init(elev=30, azim=45)
                ax.grid(False)
                ax.set_axis_off()

                fig.savefig('PSF_OpenPore_3Dprojection.svg', dpi=300)
                time.sleep(1)
                plt.close(fig)

                # Prevent any additional plotting from occurring
                plotted = True

            
# %%
np.savez('PSF_OpenPore_fitting_data.npz', **data_dict)

# %%
# Let's plot the distributions of alpha and beta values of the Moffat
data = np.load('PSF_OpenPore_fitting_data.npz')

# %%
# Mask out the ones that are less than 0.97 r2 and plot the distributions
alpha_pix = data['moffat_alpha_pix']
beta = data['moffat_beta']
moffat_r2 = data['moffat_r2']
alpha_var = data['moffat_alpha_var_pix']
beta_var = data['moffat_beta_var']
mask = moffat_r2 > 0.97

# Mask bad ones
alpha_pix = alpha_pix[mask]
beta = beta[mask]
alpha_var = alpha_var[mask]
beta_var = beta_var[mask]

# Calculate weights
w_alpha = 1.0/alpha_var
weighted_alpha_pix = np.average(alpha_pix, weights=w_alpha)
weighted_alpha_pix_var = np.average((alpha_pix-weighted_alpha_pix)**2, weights=w_alpha)
weighted_alpha_pix_std = np.sqrt(weighted_alpha_pix_var)

w_beta = 1.0/beta_var
weighted_beta = np.average(beta, weights = w_beta)
weighted_beta_var = np.average((beta-weighted_beta)**2, weights=w_beta)
weighted_beta_std = np.sqrt(weighted_beta_var)


# Alpha
fig, ax = plt.subplots()
ax.hist(alpha_pix*mpp, ec='black', bins=20)
ax.axvline(weighted_alpha_pix*mpp, ls='--', c='black')
ax.axvline((weighted_alpha_pix-weighted_alpha_pix_std)*mpp, ls='-.', c='black')
ax.axvline((weighted_alpha_pix+weighted_alpha_pix_std)*mpp, ls='-.', c='black')
ax.set_xlabel(r'$\alpha$ ($\mathrm{\mu}$m)')
ax.set_ylabel('Counts')
mu  = weighted_alpha_pix * mpp
sig = weighted_alpha_pix_std * mpp

title = rf"$\alpha = {mu:.2f} \pm {sig:.2f}\,\mathrm{{\mu m}}$"
ax.set_title(title)
fig.savefig('PSF_OpenPore_Moffat_AlphaDist.svg')
plt.close(fig)

# Beta
fig, ax = plt.subplots()
ax.hist(beta, ec='black', bins=20)
ax.axvline(weighted_beta, ls='--', c='black')
ax.axvline((weighted_beta-weighted_beta_std), ls='-.', c='black')
ax.axvline((weighted_beta+weighted_beta_std), ls='-.', c='black')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('Counts')
mu  = weighted_beta
sig = weighted_beta_std

title = rf"$\beta = {mu:.2f} \pm {sig:.2f}$"
ax.set_title(title)
fig.savefig('PSF_OpenPore_Moffat_BetaDist.svg')
plt.close(fig)


# %%
# Closed pore
datasets = ['001', '002', '003', '004', '006', '007', '008', '009']
startframes = [[2932, 3164, 4645], [116, 1587, 1811, 2151], [2025, 2065, 3655], [79, 269], [56], [1487, 1572], [1606, 3156, 4425, 5201, 5253], [4625, 4745, 4815]] # FIJI indexed (start at 1 so subtract 1!)
endframes = [[2934, 3215, 4795], [134, 1722, 1822, 2167], [2027, 2074, 3732], [140, 340], [58], [1491, 1582], [1735, 3176, 4515, 5218, 5267], [4686, 4790, 4858]]
per_dataset = {
    ds: {
        "interval_counts": [e - s + 1 for s, e in zip(s_list, e_list)],
        "total": sum(e - s + 1 for s, e in zip(s_list, e_list)),
    }
    for ds, s_list, e_list in zip(datasets, startframes, endframes)
}

grand_total = sum(d["total"] for d in per_dataset.values())
print(grand_total)
# %%

framerate = 485.44 # Hz

sigma_small = 3
sigma_large = 15

half_w = 18

data_dict = defaultdict(list)
plotted = False

for idx, d in enumerate(datasets):
    print(f'dataset: {d}')
    stack = io.imread(os.path.join(f'D:/KCL_PhD_THESIS/R9.4.1_simulmeas/32bp_Cy5_32bp_BHQ2_0.1uM/20211010/data_cleaning+tracking/droplet_1_{d}_-100mV/cleaned_data/20211010_droplet_1_{d}_-100mV_green.tif'))
    for jdx, f in enumerate(startframes[idx]):
        substack = stack[f-1:endframes[idx][jdx-1+1]].copy()

        for zdx, frame in enumerate(substack):
            print(f'frame: {zdx}')
            frame_small = gaussian_filter(frame.astype(np.float32), sigma = (sigma_small, sigma_small))
            frame_large = gaussian_filter(frame.astype(np.float32), sigma = (sigma_large, sigma_large))
            frame_dog = frame_small-frame_large
            frame_dog = np.clip(frame_dog, 0, None)
            flat_idx = np.argmax(frame_dog)
            y, x = np.unravel_index(flat_idx, frame.shape)

            expected_shape = (2*half_w+1, 2*half_w+1)
            
            roi = frame[y-half_w:y+half_w+1, x-half_w:x+half_w+1].copy()
            if roi.shape != expected_shape:
                continue

            popt_moffat, pcov_moffat, r2_moffat = fit_2d_moffat(roi)
            popt_gaussian, pcov_gaussian, r2_gaussian = fit_2d_gaussian(roi)

            if popt_moffat is not None and popt_gaussian is not None:
                perr_gaussian = np.diag(pcov_gaussian)
                perr_moffat = np.diag(pcov_moffat)
                # # Save the data
                data_dict['moffat_alpha_pix'].append(popt_moffat[3])
                data_dict['moffat_beta'].append(popt_moffat[4])
                data_dict['moffat_alpha_var_pix'].append(perr_moffat[3])
                data_dict['moffat_beta_var'].append(perr_moffat[4])
                data_dict['moffat_fwhm_pix'].append(2*popt_moffat[3]*np.sqrt(2**(1/popt_moffat[4])-1))
                data_dict['moffat_fwhm_um'].append(2*popt_moffat[3]*np.sqrt(2**(1/popt_moffat[4])-1)*mpp)
                data_dict['moffat_r2'].append(r2_moffat)

                data_dict['gaussian_sigma'].append(popt_gaussian[3])
                data_dict['gaussian_fwhm_pix'].append(2*np.sqrt(2*np.log2(2))*popt_gaussian[3])
                data_dict['gaussian_fwhm_um'].append(2*np.sqrt(2*np.log2(2))*popt_gaussian[3]*mpp)
                data_dict['gaussian_r2'].append(r2_gaussian)
                data_dict['moffat_sigma_var_pix'].append(perr_gaussian[3])

            # For plotting
            if zdx != 0 and not plotted:
            
                print('Plotting')
                h, w = roi.shape

                center_y = h//2
                center_x = w //2

                data_cs = roi[center_y, :]

                # Define finer grid for y-coordinates to evaluate model on
                x_pix = np.arange(w, dtype=float) # We take the central row and all columns, so x
                x_fine = np.linspace(0, w-1, 10*(w-1)+1)

                # Shift so x0 is at 0
                x_pix_shifted_moffat = x_pix - popt_moffat[1]
                x_fine_shifted_moffat = x_fine - popt_moffat[1]

                x_pix_shifted_gaussian = x_pix - popt_gaussian[1]
                x_fine_shifted_gaussian = x_fine - popt_gaussian[1]

                model_moffat_1d = moffat_1d(x_fine, popt_moffat[0], popt_moffat[1], popt_moffat[3], popt_moffat[4], popt_moffat[5])
                model_gaussian_1d = gaussian_1d(x_fine, popt_gaussian[0], popt_gaussian[1], popt_gaussian[3], popt_gaussian[4])

                fig, ax = plt.subplots()
                ax.plot(x_pix_shifted_moffat*mpp, data_cs, ls='None', marker='.')
                ax.plot(x_fine_shifted_moffat*mpp, model_moffat_1d, ls='--')
                ax.set_xlabel(r'Distance ($\mathrm{\mu}$m)')
                ax.set_ylabel('F (counts)')
                # plt.show(block=True)
                fig.savefig('PSF_BlockedPore_moffat_fit.svg', dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.plot(x_pix_shifted_gaussian*mpp, data_cs, ls='None', marker='.')
                ax.plot(x_fine_shifted_gaussian*mpp, model_gaussian_1d, ls='--')
                ax.set_xlabel(r'Distance ($\mathrm{\mu}$m)')
                ax.set_ylabel('F (counts)')
                # plt.show(block=True)
                fig.savefig('PSF_BlockedPore_gaussian_fit.svg', dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots()
                ax.plot(x_pix_shifted_gaussian*mpp, data_cs, ls='None', marker='.')
                ax.plot(x_fine_shifted_gaussian*mpp, model_gaussian_1d, ls='--', label='Gaussian fit')
                ax.plot(x_fine_shifted_moffat*mpp, model_moffat_1d, ls='--', label='Moffat fit')
                ax.set_xlabel(r'Distance ($\mathrm{\mu}$m)')
                ax.set_ylabel('F (counts)')
                ax.legend(frameon=False)
                # plt.show(block=True)
                fig.savefig('PSF_BlockedPore_gaussian_fit_moffat_fit.svg', dpi=300)
                plt.close(fig)    
                 
                # Make 3d plot
                Y, X = np.mgrid[0:h, 0:w]
                x_flat, y_flat = X.ravel(), Y.ravel()
                z_flat = np.zeros_like(x_flat)
                dz_flat = roi.ravel()
                dx=dy=1

                # Color
                norm = (dz_flat-dz_flat.min())/(dz_flat.max()-dz_flat.min()+1e-12)
                colors = matplotlib.cm.viridis(norm)

                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.bar3d(x_flat, y_flat, z_flat, dx, dy, dz_flat, shade=False, color=colors, ec='black', linewidth=0.5)
                ax.view_init(elev=30, azim=45)
                ax.grid(False)
                ax.set_axis_off()

                fig.savefig('PSF_BlockedPore_3Dprojection.svg', dpi=300)
                time.sleep(1)
                plt.close(fig)

                # Prevent any additional plotting from occurring
                plotted = True

# %%
np.savez('PSF_BlockedPore_fitting_data.npz', **data_dict)

# %%
# Let's plot the distributions of alpha and beta values of the Moffat
data = np.load('PSF_BlockedPore_fitting_data.npz')
# %%
# Mask out the ones that alpha < 0 (wrong fit) and r2 <0.5
alpha_pix = data['moffat_alpha_pix']
beta = data['moffat_beta']
moffat_r2 = data['moffat_r2']
alpha_var = data['moffat_alpha_var_pix']
beta_var = data['moffat_beta_var']

mask = (alpha_pix>0)&(beta>0)&(moffat_r2>0.5)

# Mask bad ones
alpha_pix = alpha_pix[mask]
beta = beta[mask]
alpha_var = alpha_var[mask]
beta_var = beta_var[mask]

# Calculate weights
w_alpha = 1.0/alpha_var
weighted_alpha_pix = np.average(alpha_pix, weights=w_alpha)
weighted_alpha_pix_var = np.average((alpha_pix-weighted_alpha_pix)**2, weights=w_alpha)
weighted_alpha_pix_std = np.sqrt(weighted_alpha_pix_var)

w_beta = 1.0/beta_var
weighted_beta = np.average(beta, weights = w_beta)
weighted_beta_var = np.average((beta-weighted_beta)**2, weights=w_beta)
weighted_beta_std = np.sqrt(weighted_beta_var)


# Alpha
fig, ax = plt.subplots()
ax.hist(alpha_pix*mpp, ec='black', bins=20)
ax.axvline(weighted_alpha_pix*mpp, ls='--', c='black')
ax.axvline((weighted_alpha_pix-weighted_alpha_pix_std)*mpp, ls='-.', c='black')
ax.axvline((weighted_alpha_pix+weighted_alpha_pix_std)*mpp, ls='-.', c='black')
ax.set_xlabel(r'$\alpha$ ($\mathrm{\mu}$m)')
ax.set_ylabel('Counts')
mu  = weighted_alpha_pix * mpp
sig = weighted_alpha_pix_std * mpp

title = rf"$\alpha = {mu:.2f} \pm {sig:.2f}\,\mathrm{{\mu m}}$"
ax.set_title(title)
# plt.show(block=True)
fig.savefig('PSF_BlockedPore_Moffat_AlphaDist.svg')
plt.close(fig)

# Beta
fig, ax = plt.subplots()
ax.hist(beta, ec='black', bins=20)
ax.axvline(weighted_beta, ls='--', c='black')
ax.axvline((weighted_beta-weighted_beta_std), ls='-.', c='black')
ax.axvline((weighted_beta+weighted_beta_std), ls='-.', c='black')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel('Counts')
mu  = weighted_beta
sig = weighted_beta_std

title = rf"$\beta = {mu:.2f} \pm {sig:.2f}$"
ax.set_title(title)
# plt.show(block=True)
fig.savefig('PSF_BlockedPore_Moffat_BetaDist.svg')
plt.close(fig)
# %%
