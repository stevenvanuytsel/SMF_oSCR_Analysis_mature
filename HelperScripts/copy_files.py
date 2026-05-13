# %%
import os
import shutil

# %%
subfolders = [f.path for f in os.scandir(path='.') if f.is_dir()]
subfolders = [f for f in subfolders if 'registry' not in f]

# %%
for f in subfolders:
    # shutil.copy('data_cleaning.py', f)
    # shutil.copy('pore_localization.py', f)
    # shutil.copy('pore_track_linking.py', f)
    # shutil.copy('pore_idealization.py', f)
    shutil.copy('pore_SM_signal_extraction.py', f)

# %%
