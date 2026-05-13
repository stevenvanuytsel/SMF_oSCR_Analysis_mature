# %%
import os
import time
import subprocess
import sys
import glob

# %%
# Get all subdirectories in the current directory (or change root as needed)
base_dir = os.path.abspath(os.path.dirname(__file__))
subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
subfolders = [f for f in subfolders if 'registry' not in f.lower()]
for folder in subfolders:
    name = 'pore+SM'
    [os.remove(f) for f in glob.glob(os.path.join(folder, name, "*.png"))]
# %%
