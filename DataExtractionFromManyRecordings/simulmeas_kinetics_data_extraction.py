# %%
import glob
import os

import pandas as pd

# %%
# load kinetics dataframe
kinetics_folder = 'D:/KCL_PhD_THESIS/R9.4.1_simulmeas/32bp_Cy5_32bp_BHQ2_0.1uM/oscr_kinetics_analysis'
fname = '32bp_Cy5_32bp_BHQ2_0.1uM_closed_frames.csv'
segments = pd.read_csv(os.path.join(kinetics_folder, fname))

# %%
# Loop over segments to find the closed states we care about and see if it has an associated simulmeas event
all_segdfs = []

for name in segments.savename.unique():
    print(name)
    date = name.split('_')[0]
    droplet = name.split(f'{date}_')[1]
    droplet = f'{droplet}_-100mV'
    sm_folder = 'pore+SM'
    basefolder = 'D:/KCL_PhD_THESIS/R9.4.1_simulmeas/32bp_Cy5_32bp_BHQ2_0.1uM/'
    df_name = f'{date}_{droplet}_pore_sm.csv'
    df = pd.read_csv(os.path.join(basefolder, date, 'data_cleaning+tracking', droplet, sm_folder, df_name))
    df_closed = df[df['state_seg']=='closed'].copy()
    df_closed.rename(columns={'track_id':'particle'}, inplace=True) # rename column so merge works
    gb_cols = ['particle', 'segment_id_refined']

    # Find segments that have sm event, also find the onset of the peak and the peak
    seg_has_sm = df_closed.groupby(gb_cols)['sm_events'].any().reset_index(name='has_sm_event')
    onsets = df_closed[df_closed['sm_events']==1].groupby(gb_cols)['frame'].min().reset_index(name='sm_onset_frame')

    # Overlay that with the known closed segments
    segdf = segments.loc[segments['savename']==name].copy()
    segdf = segdf.merge(seg_has_sm, on=['particle', 'segment_id_refined'], how='left')
    segdf = segdf.merge(onsets, on=['particle', 'segment_id_refined'], how='left')


    # Only keep the rows with a sm event
    segdf_sm = segdf[segdf['has_sm_event']].copy()
    
    # Throw out the k probabilities bc we don't need them
    cols_to_drop = ['p_k1', 'p_k2']
    segdf_final = segdf_sm.drop(columns=cols_to_drop, errors='ignore')

    # Append to list
    all_segdfs.append(segdf_final)
# %%
final_df = pd.concat(all_segdfs, ignore_index=True)

final_df.to_csv("32bp_Cy5_32bp_BHQ2_0.1uM_closed_sm_event_frames.csv", index=False)


# %%
