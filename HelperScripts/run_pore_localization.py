# %%
import os
import time
import subprocess

# %%
# Get list of subfolders (excluding registry)
subfolders = [f.path for f in os.scandir(path='.') if f.is_dir()]
subfolders = [f for f in subfolders if 'registry' not in f.lower()]

# %%
# Path to the data_cleaning script
script_name = "pore_localization.py"

# Get all subdirectories in the current directory (or change root as needed)
base_dir = os.path.abspath(os.path.dirname(__file__))
subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]

start = time.perf_counter()
for folder in subfolders:
    _start = time.perf_counter()
    script_path = os.path.join(folder, script_name)
    if os.path.isfile(script_path):
        print(f"\nRunning {script_name} in {folder}...")
        subprocess.run(["python", script_path], cwd=folder)
    else:
        print(f"Skipped {folder} — no {script_name} found.")
    _end = time.perf_counter()
    print(f'{_end-_start}s to complete')

end = time.perf_counter()
print(f'{end-start}s to finish everything')
# %%
