# %%
import glob
import os

# %%
fnames = glob.glob(os.path.join('..', '*.tif'))
# Filter out files starting with 'blank' or 'registry'
fnames = [f for f in fnames if not os.path.basename(f).startswith(('blank', 'registry'))]
print(fnames)
folders = [os.path.splitext(os.path.basename(x))[0] for x in fnames]
subdirs = ['cleaned_data', 'pore_tracks', 'pore_localization', 'pore_idealization']

for f in folders:
    print(f)
    os.makedirs(f, exist_ok=True)
    for s in subdirs:
        os.makedirs(os.path.join(f, s), exist_ok=True)

os.makedirs('registry', exist_ok=True)


# %%
