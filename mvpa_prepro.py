# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 11:29:24 2020

@author: Aaron Ang
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import mne

## We set the log-level to 'warning' so the output is less verbose
mne.set_log_level('warning')

## If you need help just ask the machine
# get_ipython().run_line_magic('pinfo', 'mne.pick_types')

## Specify path to data
data_path = os.path.expanduser("D:\\Ubuntu_Programs\\meeg\\ds000117-practical\\")
data_path = os.path.expanduser("C:\\Users\\egora\\Downloads\\meg\\")

## Important to rename the file to follow this convention:
# sub-NN_ses_meg_resting-state_run-01_proc-sss_meg.fif  # resting state, eyes closed
# sub-NN_ses-meg_experimental_run-01_proc-sss_meg.fif  # experimental run

## Specify path to MEG raw data file
# raw_fname = os.path.join(data_path,
#     'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif')
# raw_fname = os.path.join(data_path,
#     'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01_ses-meg_resting-state_run-01_proc-sss_meg.fif')
raw_fname = os.path.join(data_path,
                         'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01_ses-meg_experimental_run-01_proc-sss_raw.fif')

## Read raw data file and assign it as a variable
raw = mne.io.read_raw_fif(raw_fname, preload=False)

## ============================================================================
## Section 1:
## To initiate your desired function, indicate '1' in the respective variables.
##
## Note that %matplotlib qt (or inline) won't work on Spyder.
## In the case of Spyder, you'll need to set Tools -> references -> IPython Console -> Graphics -> Backend: Automatic.
## to get plots to be generated in a sepaate window.
##
## It is also important to set Tools -> Preferences -> Run -> General Settings -> Remove all variables before execution
## to make sure that all stored memory has been purged from previous runs to prevent memory overload.
##
## Comments with *** indicates that these variables should always be set to 1. Otherwise, data/process will turn
## wonky.
## ============================================================================

operations_to_apply = dict(

    ### RAWS TO EPOCHS AND EVOKED ###
    ### -------------------------------------
    ## Get to know your data
    RawInfo=1,  # Gets all the specs of your data
    IdentifySensors=1,  # Plot all sensors used in the scanner
    LoadData=1,  # *** This should always be set to 1 to run anything else below.
    PlotRaw=1,  # Plot raw data. You can remove first round of bad data here.

    ## Data Manipulation
    FreqFilter=0,  # *** Need to set filter range in Section 2 - FreqFilter
    CorrectEvents=0,  # *** MUST ALWAYS BE SET to 1! To correct for stim delay

    ## Plot Data
    PlotManipulatedData=0,  # For the chosen sensor defined in Section 2 - SensorType

    ## Events and artifact Epochs
    PlotEvents1=0,  # Events selected based on chosen SensorType defined in section 2.
    PlotEvents2=0,  # Plot events over activity across all sensors
    PlotEoGEpochs_Manual=0,
    # You need to set all Epochs parameters below in section 2, Epoch. Recommend to use Auto instead.
    PlotEoGEpochs_Auto=0,  # Automatic Epoch rejection based on PCA-based signal space projection (SSP)
    PlotECGEpochs_Auto=0,  # As above, but ECG.

    ## PCA
    ProcessPCA=0,  # *** If running any PCA functions below, this should always be set to 1.
    PlotPCACompo=0,  # Plot PCA noise Component for EEG, Mag and Grad based on EoG and ECG.
    ApplyPCA=0,
    # *** Apply PCA output onto epoch data. ProcessPCA MUST be set to 1 for this to work. Should also be set to 1 as it overwrites old epoch data with the new projection.
    ExplorePCA=0,  # An interactive plot that allows you to explore PCA components.
    ProjectedPCA=0,  # Produce final plot that shows data after PCA been projected.

    ## Epochs
    PlotERPs=0,  # Need to identify sensor in Section 2 - ERPSensor for ERP plotting.
    PlotEpochs=0,
    # Plots every single epoch for all sensors. Here is where you can manually remove (noisy) epochs by left clicking.
    # Be careful. Once you select epochs to be removed, they will be registered into the drop log once you close the plot. The only way to undo this is to re-run the entire script once more.
    SaveEpochs=0,  # Save Epoch file as an output

    ## Evoked (ERP)
    PlotERP=0,  # You need to specify time range in Section 2 - Evoked.
    ApplyProj=0,  # *** Apply PCA projections on ERP.
    PlotProj=0,
    PlotTopo=0,
    PlotOnSpecificTime=0,  # Need to specify that time in Section 2 - ERPTime
    PlotOnConditionAndSensor=0,  # Need to specify that time in Section 2 - condition, meg and Plot min and max range
    SaveEvoked=0,  # Save Evoked file as an output
    ReadEvoked=0,
    # *** Read an Evoked file for contrast comparisons. Need to specify conditions you wish to read at Section 2 - ReadConditions.

    ## Contrasts Comparisons
    Contrast=0,
    # Shows contrast activity over time between 2 conditions. Need to define various variables in Section 2 - Contrast, part 1
    Topo=0,
    # Shows topographic activity morphing over time on a sensor group defined by the 'meg' variable in Section 2 - Evoked. Also need to specifiy settings in Section 2 - Contrasts, part 2
    SaveTopo=0,  # Saves Topo plot
    AllTopo=0,
    # Plots all sensors activity of all conditions in a single topographic plot. Need to specify time range in Section 2 - Contrasts, part 3
    CompareEvokeds=0
    # Compare evoked activity across all conditions on a specific sensor. Sensor needs to be defnied in Section 2 - Contrast, part 4

)


def loadData():
    raw.load_data()  # it is required to load data in memory
    raw.resample(300)


## ============================================================================
## Section 2:
## Here, you get a sandbox for editing the data in various ways that you require
## ============================================================================

## Define Experiment Conditions
conditions = ['left', 'right']

## FreqFilter: Low and high pass filter
LowF = 0
HighF = 80

## TemporalCrop: Time range
start = 0
stop = int(50 * raw.info['sfreq'])  # 15s

## Sensor Type - Define Sensor group or specific sensor for the events extraction
SensorType = 'STI101'  # The specific sensor label (ie. 'STI101')

## Pick Sensor to produce ERP output
ERPSensor = 'EEG065'


## Amend sensor types/names
def applyChanges():
    #     raw.set_channel_types({'EEG061': 'eog',
    #                            'EEG062': 'eog',
    #                            'EEG063': 'ecg',
    #                            'EEG064': 'misc'})  # EEG064 free-floating el.

    #     raw.rename_channels({'EEG061': 'EOG061',
    #                          'EEG062': 'EOG062',
    #                          'EEG063': 'ECG063'})

    ## To drop certain sensors
    #     to_drop = ['STI201', 'STI301', 'MISC201', 'MISC202', 'MISC203',
    #                'MISC204', 'MISC205', 'MISC206', 'MISC301', 'MISC302',
    #                'MISC303', 'MISC304', 'MISC305', 'MISC306', 'CHPI001',
    #                'CHPI002', 'CHPI003', 'CHPI004', 'CHPI005', 'CHPI006',
    #                'CHPI007', 'CHPI008', 'CHPI009']

    raw.drop_channels(to_drop)


## For event trigger and conditions we use a Python dictionary with keys that contain "/" for grouping sub-conditions
event_id = {
    'left': 5,
    'right': 6,
    # 'left/dur1/cont017'  # 1-frame duration and contrast of 17%
}
event_id = {
    'left/dur1/cont017': 2, 'left/dur1/cont033': 3, 'left/dur1/cont050': 4, 'left/dur1/cont100': 5,
    'left/dur2/cont017': 6, 'left/dur2/cont033': 7, 'left/dur2/cont050': 8, 'left/dur2/cont100': 9,
    'left/dur3/cont017': 10, 'left/dur3/cont033': 11, 'left/dur3/cont050': 12, 'left/dur3/cont100': 13,
    'left/dur4/cont017': 14, 'left/dur4/cont033': 15, 'left/dur4/cont050': 16, 'left/dur4/cont100': 17,
    'right/dur1/cont017': 22, 'right/dur1/cont033': 23, 'right/dur1/cont050': 24, 'right/dur1/cont100': 25,
    'right/dur2/cont017': 26, 'right/dur2/cont033': 27, 'right/dur2/cont050': 28, 'right/dur2/cont100': 29,
    'right/dur3/cont017': 30, 'right/dur3/cont033': 31, 'right/dur3/cont050': 32, 'right/dur3/cont100': 33,
    'right/dur4/cont017': 34, 'right/dur4/cont033': 35, 'right/dur4/cont050': 36, 'right/dur4/cont100': 37,
}
## ------------------------------ Epochs --------------------------------------
## Define epochs parameters:
tmin = -0.3  # start of each epoch (500 ms before the trigger)
tmax = 0.6  # end of each epoch (600 ms after the trigger)

## Define the baseline period:
baseline = (-0.2, 0)  # means from 200ms before to stim onset (t = 0)

## Define peak-to-peak (amplitude range) rejection parameters for gradiometers, magnetometers and EOG:
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)  # this can be highly data dependent

## Pick channels by type and names
picks = mne.pick_types(raw.info, meg=True, eeg=True, eog=True,
                       stim=False, exclude='bads')

## ------------------------------ Evoked --------------------------------------
# This is before the exclusion of the first PCA component from all three sensor types.
# Define the different time you wish to generate ERP plot.
times = [0.0, 0.1, 0.18]

ERPTime = 0.17

condition = 'left'
meg = 'grad'
PlotMinCrop = -0.2
PlotMinCrop = 0.5

ReadConditions = "left/dur4/cont100"

## ----------------------------- Contrasts ------------------------------------
## Part 1
Contrast1 = 'left'
Contrast2 = 'right'

MinTime = -0.1  # time range in seconds, for contrast plot
MaxTime = 0.3

## Part 2
TopoMinTime = 0.05  # time range in seconds, for topographic plot
TopoMaxTime = 0.15
NoOfTopos = 5  # Number of topos within the specific time range above to produce

## Part 3
AllTopoMinTime = -0.1
AllTopoMaxTime = 0.4

## Part 4
CompareEvoMinTime = -0.1
CompareEvoMaxTime = 0.4
SensorID = 'EEG065'

## ============================================================================
## Operations commands
## ============================================================================

if operations_to_apply['RawInfo']:
    print('Raw Data Info:')
    print(raw.info)

if operations_to_apply['IdentifySensors']:
    raw.plot_sensors(kind='topomap', ch_type='grad');
    raw.plot_sensors(kind='topomap', ch_type='mag');
    raw.plot_sensors(kind='topomap', ch_type='eeg');

if operations_to_apply['LoadData']:
    loadData()
    applyChanges()

if operations_to_apply['PlotRaw']:
    raw.plot();

if operations_to_apply['FreqFilter']:
    raw.filter(LowF, HighF)

if operations_to_apply['CorrectEvents']:
    ## Extracting events
    events = mne.find_events(raw, stim_channel=str(SensorType), verbose=True)

    ## Correcting for time offset of 34.5ms in the stimulus presentation. We need to correct events accordingly.
    delay = int(round(0.0345 * raw.info[
        'sfreq']))  # the mismatch value is going to be different depending on the light sensor channel
    events[:, 0] = events[:, 0] + delay
    events = events[events[:, 2] < 20]  # take only events with code less than 20

if operations_to_apply['PlotManipulatedData']:
    data = raw.get_data(SensorType, start=start, stop=stop)
    plt.plot(raw.times[start:stop], data.T)
    print('Data Shape:', data.shape)
    print('Time Shape:', raw.times[start:stop].shape)
    print('Data Max:', np.max(data))

if operations_to_apply['PlotEvents1']:
    fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'],
                              event_id=event_id)

if operations_to_apply['PlotEvents2']:
    raw.plot(event_id=event_id, events=events);

if operations_to_apply['PlotEoGEpochs_Manual']:
    ## Define Epochs based on the 4 parameters set above
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                        picks=picks, baseline=baseline,
                        reject=reject)
    epochs.drop_bad()  # remove bad epochs based on reject
    epochs.load_data()  # load data in memory
    epochs.plot_drop_log()
    for drop_log in epochs.drop_log[:20]:
        print(drop_log)

    epochs.events.shape
    events[epochs.selection] == epochs.events  # iow, epochs.events contains the kept epochs (post-rejection)

if operations_to_apply['PlotEoGEpochs_Auto']:
    # We can use a convenience function
    eog_epochs = mne.preprocessing.create_eog_epochs(raw.copy().filter(1, None))
    eog_epochs.average().plot_joint()

    # raw.plot(events=eog_epochs.events);

if operations_to_apply['PlotECGEpochs_Auto']:
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw.copy().filter(1, None))
    ecg_epochs.average().plot_joint()

if operations_to_apply['ProcessPCA']:
    layouts = [mne.find_layout(raw.info, ch_type=ch) for ch in ("eeg", "mag", "grad")]

    projs_eog, _ = mne.preprocessing.compute_proj_eog(
        raw, n_mag=3, n_grad=3, n_eeg=3, average=True)

    projs_ecg, _ = mne.preprocessing.compute_proj_ecg(
        raw, n_mag=3, n_grad=3, n_eeg=3, average=True)

if operations_to_apply['PlotPCACompo']:
    mne.viz.plot_projs_topomap(projs_eog, layout=layouts);
    mne.viz.plot_projs_topomap(projs_ecg, layout=layouts);

if operations_to_apply['ApplyPCA']:
    reject2 = dict(mag=reject['mag'], grad=reject['grad'])

    epochs_clean = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                              picks=picks, baseline=baseline,
                              preload=False,
                              reject=reject2)

    epochs_clean.add_proj(projs_eog + projs_ecg)
    epochs = epochs_clean

if operations_to_apply['ExplorePCA']:
    epochs_clean.copy().average().plot(proj='interactive', spatial_colors=True)

if operations_to_apply['ProjectedPCA']:
    epochs_clean.average().plot(proj=True, spatial_colors=True)

if operations_to_apply['PlotERPs']:
    raw.plot_psd(fmax=40);
    epochs.plot_image(picks=str(ERPSensor), sigma=1.);

if operations_to_apply['PlotEpochs']:
    epochs.plot()
    print('Epochs Drop Log:')
    print(epochs.drop_log)
    epochs.event_id

if operations_to_apply['SaveEpochs']:
    print('Revised Epochs Drop Log:')
    print(epochs.drop_log)
    epochs_fname = raw_fname.replace('_meg.fif', '-epo.fif')
    epochs.save(epochs_fname, overwrite=True)

if operations_to_apply['PlotERP']:
    evoked = epochs.average()
    evoked.plot_topomap(ch_type='mag', times=times, proj=True);
    evoked.plot_topomap(ch_type='grad', times=times, proj=True);
    evoked.plot_topomap(ch_type='eeg', times=times, proj=True);

if operations_to_apply['ApplyProj']:
    evoked = epochs.average()
    evoked.del_proj()  # delete previous proj
    # take first for each sensor type
    evoked.add_proj(projs_eog[::3] + projs_ecg[::3])  # selecting every third PCA component, starting with the first one
    # evoked.add_proj(list(projs_eog[i] for i in [0, 1, 3, 4, 6, 7]) + list(projs_ecg[i] for i in [0, 1, 3, 4, 6, 7])) # allows a custom selection of PCA components for exclusion
    evoked.apply_proj()  # apply

if operations_to_apply['PlotProj']:
    evoked.plot(spatial_colors=True, proj=True)

if operations_to_apply['PlotTopo']:
    evoked.plot_topomap(times=np.linspace(0.05, 0.45, 8),
                        ch_type='mag', proj='True');  ## or false, or interactive

    evoked.plot_topomap(times=np.linspace(0.05, 0.45, 8),
                        ch_type='grad', proj='True');

    evoked.plot_topomap(times=np.linspace(0.05, 0.45, 8),
                        ch_type='eeg', proj='True');

if operations_to_apply['PlotOnSpecificTime']:
    evoked.plot_joint(times=[ERPTime])

if operations_to_apply['PlotOnConditionAndSensor']:
    epochs[str(condition)].average().pick_types(meg=str(meg)).crop(PlotMinCrop, PlotMaxCrop).plot(spatial_colors=True);

if operations_to_apply['SaveEvoked']:
    evoked_fname = raw_fname.replace('_meg.fif', '-ave.fif')
    # evoked.save(evoked_fname)

    # or to write multiple conditions in 1 file
    evokeds_list = [epochs[k].average() for k in event_id]  # get evokeds
    mne.write_evokeds(evoked_fname, evokeds_list)

if operations_to_apply['ReadEvoked']:
    evoked_fname = raw_fname.replace('_meg.fif', '-ave.fif')
    evoked1 = mne.read_evokeds(evoked_fname, condition=str(ReadConditions),
                               baseline=(None, 0), proj=True)

if operations_to_apply['Contrast']:
    evoked_cond1 = epochs[str(Contrast1)].average()
    evoked_cond2 = epochs[str(Contrast2)].average()

    contrast = mne.combine_evoked([evoked_cond1, evoked_cond2], [0.5, -0.5])  # float list here indicates range of power

    # Note that this combines evokeds taking into account the number of averaged epochs (to scale the noise variance)
    print(evoked1.nave)  # average of 12 epochs
    print(contrast.nave)  # average of 116 epochs

    print(contrast)

    fig = contrast.copy().pick('grad').crop(MinTime, MaxTime).plot_joint()
    fig = contrast.copy().pick('mag').crop(MinTime, MaxTime).plot_joint()
    fig = contrast.copy().pick('eeg').crop(MinTime, MaxTime).plot_joint()

if operations_to_apply['Topo']:
    evoked_cond1 = epochs[str(Contrast1)].average()
    evoked_cond2 = epochs[str(Contrast2)].average()

    contrast = mne.combine_evoked([evoked_cond1, evoked_cond2], [0.5, -0.5])

    contrast.plot_topomap(times=np.linspace(TopoMinTime, TopoMaxTime, NoOfTopos), ch_type=str(meg))

if operations_to_apply['SaveTopo']:
    plt.savefig('topoMap.pdf')

if operations_to_apply['AllTopo']:
    evoked_cond1 = epochs[conditions[0]].average().crop(AllTopoMinTime, AllTopoMaxTime)
    evoked_cond2 = epochs[conditions[1]].average().crop(AllTopoMinTime, AllTopoMaxTime)
    evoked_cond3 = epochs[conditions[2]].average().crop(AllTopoMinTime, AllTopoMaxTime)

    mne.viz.plot_evoked_topo([evoked_cond1, evoked_cond2, evoked_cond3])

if operations_to_apply['CompareEvokeds']:
    evokeds = {k: epochs[k].average().crop(CompareEvoMinTime, CompareEvoMaxTime)
               for k in conditions}
    mne.viz.plot_compare_evokeds(evokeds, picks=str(SensorID));

## ---------------------------------- EXTRA MISCS -----------------------------
# ## ADVANCED: Customize your plots
# 
# Want to have every text in blue?
'''
fig = evoked1.plot(show=False)  # butterfly plots
fig.subplots_adjust(hspace=1.0)
for text in fig.findobj(mpl.text.Text):
    text.set_fontsize(18)
    text.set_color('blue')
for ax in fig.get_axes():
    ax.axvline(0., color='red', linestyle='--')
plt.tight_layout()
fig.savefig('plot_erf.pdf');
'''
## Identify sensor number of a specific sensor
# indices = [i for i, s in enumerate(raw.info['ch_names']) if 'EOG061' in s]
# indices
