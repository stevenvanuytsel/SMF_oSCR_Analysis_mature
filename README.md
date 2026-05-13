# SMF_oSCR_Analysis_mature
Analysis pipeline for simultaneous oSCR and SMF

## Organization
Scripts are organized per folder for easy retrieval. The folders and their uses are as follow:
1. AnalysisScripts: contains scripts to analyze kinetics, psf, laser intensity and psf fit r2 versus intensity
2. DataExtractionFromManyRecordings: contains files to extrasct the required data across many recording folders, which is then used for the kinetics analysis, for example. The idea is that every recording has its own folder with data, which then needs to be collated into a big csv in order to extract kinetics across many of them.
3. HelperFiles: these are camera shutter and voltage fallback files in case WinEDR doesn't save them correctly. IMPORTANT: this only works if you use a standardized measurement protocol, and if it's different from mine, you need to update these helper files with your own
4. HelperScripts: contains scripts that help set up the pipeline with correct folders and scripts in every folder, which can then all be ran simultaneously using the scripts beginning with "run.... .py". The additional helper script called "delete_png_files.py" is used to delete images generated during the analysis if they need to be remade (imagine you've ran the pipeline but are not happy and run it again from scratch with different localization or tracking parameters - at that point, you'll get different particle names and as the images are saved with particle names, you might not overwrite the files, which turns into a mess. Best to first delete them if you start anew).
5. ImageProcessingPipeline: all scripts required to go from raw data to oscr and smf tracks
6. ImageRegistration: when running two-colour experiments, it's important to first run a registration so that you know the displacement of each channel wrt each other. This script does that.

## Workflow
I'll briefly walk you through the general order of operations when running this pipeline using my personal setup. This means that everything depends on the relative locations that I have stored my files and folders in. If you deviate, move things around (which you absolutely can do), there will be errors and you'll have to fix the relative paths in order for things to work again.

Imagine I have a folder with data that was recorded on a particular date, under which I would have a bunch of datafiles - the structure looks as follows:
.
├── 32bpX_32bpY_concentration
│   ├── Date1
│   │   ├── blank_xxxEM.tif
│   │   ├── droplet_1_001_-100mV.abf (electrical data in ABF format, from winEDR)
│   │   ├── droplet_1_001_-100mV.tif (images from Andor)
│   │   ├── droplet_1_001_-100mV.EDR (electrical data in native EDR, from winEDR)
│   │   ├── log.txt
│   │   └── registry.tif
            
The .abf format can de exported directly from winEDR and is the one we use for electrical traces. The .EDR file is the native winEDR save that always happens when you do a recording, which I keep as a fallback (.abf exports can fail in winEDR).
The log file contains experimental data that I wrote down during the experiment. The registry.tif file contains the two-colour image of a graticule, which is used for the spatial alignment of the colour channels.

1. The first thing I do is make a folder called "data_cleaning+tracking" - the name is not important, it's just the folder where we'll do the analysis. Then the files from the "HelperFiles", "HelperScripts", "ImageProcessingPipeline" and "ImageRegistration" are copied there. Now the folder looks like this:

.
├── 32bpX_32bpY_concentration
│   ├── Date1
│   │   ├── data_cleaning+tracking
│   │   │   ├── copy_files.py
│   │   │   ├── data_cleaning.py
│   │   │   ├── delete_png_files.py
│   │   │   ├── folder_creating.py
│   │   │   ├── pore_idealization.py
│   │   │   ├── pore_localization.py
│   │   │   ├── pore_SM_signal_extraction.py
│   │   │   ├── pore_track_linking.py
│   │   │   ├── registration.py
│   │   │   ├── run_data_cleaning.py
│   │   │   ├── run_pore_idealization.py
│   │   │   ├── run_pore_linking.py
│   │   │   ├── run_pore_localization.py
│   │   │   ├── run_pore_SM_signal_extraction.py
│   │   │   ├── shutter_fallback.csv
│   │   │   └── voltage_fallback.csv
│   │   ├── blank_xxxEM.tif
│   │   ├── droplet_1_001_-100mV.abf (electrical data in ABF format, from winEDR)
│   │   ├── droplet_1_001_-100mV.tif (images from Andor)
│   │   ├── droplet_1_001_-100mV.EDR (electrical data in native EDR, from winEDR)
│   │   ├── log.txt
│   │   ├── registry.tif

2. Run folder_creation.py
This script will ensure that there will be a separate folder for every ".tif" recording, with the folder name being equal to the recording name (you can amend this of course - if your names are too long, you might run into problems on window so try to be succinct and store the full name elsewhere, in the log, in a csv, ...)

.
├── 32bpX_32bpY_concentration
│   ├── Date1
│   │   ├── data_cleaning+tracking
│   │   │   ├── droplet_1_001_-100mV
│   │   │   ├── registry
│   │   │   ├── copy_files.py
│   │   │   ├── data_cleaning.py
│   │   │   ├── delete_png_files.py
│   │   │   ├── folder_creating.py
│   │   │   ├── pore_idealization.py
│   │   │   ├── pore_localization.py
│   │   │   ├── pore_SM_signal_extraction.py
│   │   │   ├── pore_track_linking.py
│   │   │   ├── registration.py
│   │   │   ├── run_data_cleaning.py
│   │   │   ├── run_pore_idealization.py
│   │   │   ├── run_pore_linking.py
│   │   │   ├── run_pore_localization.py
│   │   │   ├── run_pore_SM_signal_extraction.py
│   │   │   ├── shutter_fallback.csv
│   │   │   └── voltage_fallback.csv
│   │   ├── blank_xxxEM.tif
│   │   ├── droplet_1_001_-100mV.abf (electrical data in ABF format, from winEDR)
│   │   ├── droplet_1_001_-100mV.tif (images from Andor)
│   │   ├── droplet_1_001_-100mV.EDR (electrical data in native EDR, from winEDR)
│   │   ├── log.txt
│   │   └── registry.tif

3. Run "copy_files.py" (always open stuff first to see what's commented out etc, especially with copying and deleting)
This copies the image processing pipeline files into the folders (except for the registry folder)
As you see, there are many copies of e.g., data cleaning, on top of the one that lives in the main analysis folder. This is obviously redundant and can be changed. However, if you keep it this way, it is of paramount importance that after troubleshooting (which you'll do by running the specific script you're troubleshooting from within a "droplet folder"), you copy the amended file back to the "data_cleaning+tracking" folder and run the "copy_files.py" script again so that every droplet folder has the most recent version (this is exactly why it's brittle - it works but you need to be mindful and you can change this behaviour)

.
├── 32bpX_32bpY_concentration
│   ├── Date1
│   │   ├── data_cleaning+tracking
│   │   │   ├── droplet_1_001_-100mV
│   │   │   │   ├── data_cleaning.py
│   │   │   │   ├── pore_idealization.py
│   │   │   │   ├── pore_localization.py
│   │   │   │   ├── pore_SM_signal_extraction.py
│   │   │   │   └── pore_track_linking.py
│   │   │   ├── registry
│   │   │   ├── copy_files.py
│   │   │   ├── data_cleaning.py
│   │   │   ├── delete_png_files.py
│   │   │   ├── folder_creating.py
│   │   │   ├── pore_idealization.py
│   │   │   ├── pore_localization.py
│   │   │   ├── pore_SM_signal_extraction.py
│   │   │   ├── pore_track_linking.py
│   │   │   ├── registration.py
│   │   │   ├── run_data_cleaning.py
│   │   │   ├── run_pore_idealization.py
│   │   │   ├── run_pore_linking.py
│   │   │   ├── run_pore_localization.py
│   │   │   ├── run_pore_SM_signal_extraction.py
│   │   │   ├── shutter_fallback.csv
│   │   │   └── voltage_fallback.csv
│   │   ├── blank_xxxEM.tif
│   │   ├── droplet_1_001_-100mV.abf (electrical data in ABF format, from winEDR)
│   │   ├── droplet_1_001_-100mV.tif (images from Andor)
│   │   ├── droplet_1_001_-100mV.EDR (electrical data in native EDR, from winEDR)
│   │   ├── log.txt
│   │   └── registry.tif

4. Copy the registration.py script into the registry folder to run the registration and run the registration script (+ visually inspect it)
We've now aligned the colour channels to each other.

.
├── 32bpX_32bpY_concentration
│   ├── Date1
│   │   ├── data_cleaning+tracking
│   │   │   ├── droplet_1_001_-100mV
│   │   │   │   ├── data_cleaning.py
│   │   │   │   ├── pore_idealization.py
│   │   │   │   ├── pore_localization.py
│   │   │   │   ├── pore_SM_signal_extraction.py
│   │   │   │   └── pore_track_linking.py
│   │   │   ├── registry
│   │   │   │   └── registration.py
│   │   │   ├── copy_files.py
│   │   │   ├── data_cleaning.py
│   │   │   ├── delete_png_files.py
│   │   │   ├── folder_creating.py
│   │   │   ├── pore_idealization.py
│   │   │   ├── pore_localization.py
│   │   │   ├── pore_SM_signal_extraction.py
│   │   │   ├── pore_track_linking.py
│   │   │   ├── registration.py
│   │   │   ├── run_data_cleaning.py
│   │   │   ├── run_pore_idealization.py
│   │   │   ├── run_pore_linking.py
│   │   │   ├── run_pore_localization.py
│   │   │   ├── run_pore_SM_signal_extraction.py
│   │   │   ├── shutter_fallback.csv
│   │   │   └── voltage_fallback.csv
│   │   ├── blank_xxxEM.tif
│   │   ├── droplet_1_001_-100mV.abf (electrical data in ABF format, from winEDR)
│   │   ├── droplet_1_001_-100mV.tif (images from Andor)
│   │   ├── droplet_1_001_-100mV.EDR (electrical data in native EDR, from winEDR)
│   │   ├── log.txt
│   │   └── registry.tif

5. From here you have two options:
   5.1 Manually run each of the scripts in the droplet folder in this order: data_cleaning.py, pore_localization.py, pore_track_linking.py, pore_idealization.py and pore_SM_signal_extraction.py
       I strongly recommend doing this at the start to ensure that you are optimizing all parameters for your specific experimental case, and to understand how each script works and what it does.
   5.2 Once you are happy with how each script works, you can run the scripts starting with "run_xxx.py" in the same order as the one I gave above. This will run the associated script in every droplet folder - I use this because i have loads of recordings that are all processed identically and I don't want to do this manually.

Running these scripts will automatically spawn new folders with the results embedded. The folder structure will now look like this:

.
├── 32bpX_32bpY_concentration
│   ├── Date1
│   │   ├── data_cleaning+tracking
│   │   │   ├── droplet_1_001_-100mV
│   │   │   │   ├── cleaned_data
│   │   │   │   │   ├── cleaned_data.csv
│   │   │   │   │   ├── cleaned_data_ch1.tif
│   │   │   │   │   └── cleaned_data_ch2.tif
│   │   │   │   ├── pore_idealization
│   │   │   │   │   ├── idealized_data.csv
│   │   │   │   │   └── image_pore1.png
│   │   │   │   ├── pore_localization
│   │   │   │   │   └── localization_data.csv
│   │   │   │   ├── pore_tracks
│   │   │   │   │   └── pore_tracking_data.csv
│   │   │   │   ├── pore+SM
│   │   │   │   │   └── pore+SM_data.csv
│   │   │   │   ├── data_cleaning.py
│   │   │   │   ├── pore_idealization.py
│   │   │   │   ├── pore_localization.py
│   │   │   │   ├── pore_SM_signal_extraction.py
│   │   │   │   └── pore_track_linking.py
│   │   │   ├── registry
│   │   │   │   └── registration.py
│   │   │   ├── copy_files.py
│   │   │   ├── data_cleaning.py
│   │   │   ├── delete_png_files.py
│   │   │   ├── folder_creating.py
│   │   │   ├── pore_idealization.py
│   │   │   ├── pore_localization.py
│   │   │   ├── pore_SM_signal_extraction.py
│   │   │   ├── pore_track_linking.py
│   │   │   ├── registration.py
│   │   │   ├── run_data_cleaning.py
│   │   │   ├── run_pore_idealization.py
│   │   │   ├── run_pore_linking.py
│   │   │   ├── run_pore_localization.py
│   │   │   ├── run_pore_SM_signal_extraction.py
│   │   │   ├── shutter_fallback.csv
│   │   │   └── voltage_fallback.csv
│   │   ├── blank_xxxEM.tif
│   │   ├── droplet_1_001_-100mV.abf (electrical data in ABF format, from winEDR)
│   │   ├── droplet_1_001_-100mV.tif (images from Andor)
│   │   ├── droplet_1_001_-100mV.EDR (electrical data in native EDR, from winEDR)
│   │   ├── log.txt
│   │   └── registry.tif

NOTES: 
1. some of the scripts require parameter optimization (as mentioned above). Sometimes these are heuristic values that you obtain by looking at your data, intermediate plots (which you need to put in yourself when going through the script as I've removed them), and then tweak the parameters to obtain the desired result. These are usually parameters related to thresholds etc.
2. In the pore localization script there are 2 values used for the fitting of the pore (alpha_pix and beta). They are obtained by running the psf_analysis.py script in the AnalysisScripts folder. You can obviously amend the PSF model etcetera etcetera, which you can do from the psf_analysis.py script and then you'd have to change the model used in "pore_localization.py" and copy it again to all relevant folders.

6. Once you have ran the entire pipeline, you need to extract the data from all recordings in order to get kinetics measurements. This is done using the scripts that live in the "DataExtractionFromManyRecordings" folder. In order for everything to work easily (with relative folder locations), your file structure should now look something like this, after you've made a folder for oscr_kinetics_analysis and one for simulmeas_kinetics_analysis, and copied the scripts mentioned in the previous sentence into them.

.
├── 32bpX_32bpY_concentration
│   ├── Date1
│   │   └── ...
│   ├── Date2
│   │   └── ...
│   ├── Date3
│   │   └── ...
│   ├── oscr_kinetics_analysis
│   │   └── oSCR_kinetics_data_extraction.py
│   └── simulmeas_kinetics_analysis
│       └── simulmeas_kinetics_data_extraction.py

Those scripts will go through each droplet folder for every date and extract the relevant data from the relevant csv to allow you to analyze kinetics. Again, this can be amended of course (both where it lives and what it does).

7. Once you have the kinetics csv extracted, the folder tree looks like this:

.
├── 32bpX_32bpY_concentration
│   ├── Date1
│   │   └── ...
│   ├── Date2
│   │   └── ...
│   ├── Date3
│   │   └── ...
│   ├── oscr_kinetics_analysis
│   │   ├── oSCR_kinetics_data_extraction.py
│   │   ├── closed_frames.csv (these are dwell times/frames - unzipping rate)
│   │   └── open_frames.csv (these are open times/frames - capture rate)
│   └── simulmeas_kinetics_analysis
│       ├── simulmeas_kinetics_data_extraction.py
│       └── simulmeas_closed_frames.csv (these are dwell times for unzipping of each single strand of the construct, using the SMF signal; note, there is no open frames because capture rate is not relevant here)

8. The next step is fitting the kinetics and making figures. To do this, I make a new folder at the level of the experiment, and copy the analysis scripts into it:

NOTE: you can see that the psf_analysis is also in this figures folder, and that's the script that I used to find the correct parameters for the PSF fitting - this should be done before running the pore_localization.py script that I mentioned above. It works by manually inputting closed and open frames of pores that are roughly in the middle of the field of view, which are then fitted in order to extract the required params

.
├── Figures
│   ├── oscr_kinetics
│   │   ├── KineticsFittingCapture_oSCR.py
│   │   └── KineticsFittingUnzipping_oSCR.py
│   ├── simul_kinetics
│   │   └── ...
│   └── psf_analysis
│       └── PSF_analysis.py
└── 32bpX_32bpY_concentration
    ├── Date1
    │   └── ...
    ├── Date2
    │   └── ...
    ├── Date3
    │   └── ...
    ├── oscr_kinetics_analysis
    │   ├── oSCR_kinetics_data_extraction.py
    │   ├── closed_frames.csv (these are dwell times/frames - unzipping rate)
    │   └── open_frames.csv (these are open times/frames - capture rate)
    └── simulmeas_kinetics_analysis
        ├── simulmeas_kinetics_data_extraction.py
        └── simulmeas_closed_frames.csv (these are dwell times for unzipping of each single strand of the construct, using the SMF signal; note, there is no open frames because capture rate is not relevant here)

Running the Kinetics....py scripts will retrieve the csvs we made in step 7 (mind the relative paths), perform some fitting and spit out the results. These scripts are currently still ran by hand and they have a couple of kinetic models (one-step, 2-step sequential, 2 one-step competing process). This can be amended and changed too of course
