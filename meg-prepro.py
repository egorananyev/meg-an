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
#get_ipython().run_line_magic('pinfo', 'mne.pick_types')

## Specify path to data
data_path = os.path.expanduser("D:\\Ubuntu_Programs\\meeg\\ds000117-practical\\")
data_path = os.path.expanduser("C:\\Users\\egora\\Downloads\\meg\\")

## Important to rename the file to follow this convention:
# sub-NN_ses_meg_resting-state_run-01_proc-sss_meg.fif  # resting state, eyes closed
# sub-NN_ses-meg_experimental_run-01_proc-sss_meg.fif  # experimental run

## Specify path to MEG raw data file
cur_subj = 1
cur_subj_str = str(cur_subj).zfill(2)
cur_run = 1
cur_run_str = str(cur_run).zfill(2)
# raw_fname = os.path.join(data_path,
#     'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif')
# raw_fname = os.path.join(data_path,
#     'derivatives\\meg_derivatives\\sub-01\\ses-meg\\meg\\sub-01_ses-meg_resting-state_run-01_proc-sss_meg.fif')

raw_fname = os.path.join(data_path,
                         'derivatives\\meg_derivatives\\' + 'sub-' + cur_subj_str + '\\ses-meg\\meg\\' + 'sub-' + \
                         cur_subj_str + '_ses-meg_experimental_run-' + cur_run_str + '_proc-sss_raw.fif')

## Read raw data file and assign it as a variable
raw = mne.io.read_raw_fif(raw_fname, preload=False)

print('Raw Data Info:')
print(raw.info)

raw.plot_sensors(kind='topomap', ch_type='grad')
raw.plot_sensors(kind='topomap', ch_type='mag')

raw.plot()  # inspect the raw channel info to establish which channels to eliminate

events = mne.find_events(raw, stim_channel='STI101', verbose=True)
events[:, 1:3] = events[:, 1:3] - 2048

# mne.pick_channels(ch_names='MISC001')
start = int(245 * raw.info['sfreq'])
stop = int(250 * raw.info['sfreq'])
light_sensor = raw.get_data(picks='MISC001', start=start, stop=stop)
light_thresh = 0.016
plt.plot(raw.times[start:stop], light_sensor.T)
light_sensor_transformed = light_sensor
light_sensor_transformed[light_sensor < light_thresh] = 0
light_sensor_transformed[light_sensor >= light_thresh] = 1
plt.plot(raw.times[start:stop], light_sensor_transformed.T)
trigger_channel = raw.get_data(picks='STI101', start=start, stop=stop) - 2048
trigger_channel[trigger_channel == 1] = 0
plt.plot(raw.times[start:stop], trigger_channel.T)
# light_sensor = mne.find_events(raw, stim_channel='MISC001', verbose=True)
# light_sensor[:, 1:3] = light_sensor[:, 1:3] - 2048

# Finds the lowest index (first instance) of trigger channel greater than 0 (1s already asigned to 0):
first_trigger = np.min(np.where(trigger_channel[0, :] > 0))
first_light = np.min(np.where(light_sensor_transformed[0, :] == 1))
sample_mismatch = first_light - first_trigger

# Checking the above value:
trigger_channel_shifted = np.append(np.repeat(0, sample_mismatch),
                                    trigger_channel[0, 0:len(trigger_channel[0])-sample_mismatch])
plt.plot(raw.times[start:stop], trigger_channel_shifted.T)

## Drop unneeded channels:
to_drop = ['STI001', 'STI002', 'STI003', 'STI004', 'STI005', 'STI006', 'SYS201']
raw.load_data()  # it is required to load data in memory
raw.resample(300)
raw.drop_channels(to_drop)

raw.plot()  # inspect the raw channel info to examine post-drop data

## Filtering
LowF = 0
HighF = 80
raw.filter(LowF, HighF)

## Correcting for time offset of 34.5ms in the stimulus presentation. We need to correct events accordingly.
events[:, 0] = events[:, 0] + sample_mismatch

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
events_ori = mne.merge_events(events, ids=np.arange(2, 18), new_id=2)
events_ori = mne.merge_events(events_ori, ids=np.arange(22, 38), new_id=3)
event_id_ori = {'left': 2, 'right': 3}
fig = mne.viz.plot_events(events_ori, sfreq=raw.info['sfreq'], event_id=event_id_ori)

raw.plot(event_id=event_id_ori, events=events_ori)

## ------------------------------ Epochs --------------------------------------
## Define epochs parameters:
tmin = -0.3  # start of each epoch (500 ms before the trigger)
tmax = 0.6  # end of each epoch (600 ms after the trigger)

## Define the baseline period:
baseline = (-0.2, 0)  # means from 200ms before to stim onset (t = 0)

## Define peak-to-peak (amplitude range) rejection parameters for gradiometers, magnetometers and EOG:
reject = dict(grad=4000e-13, mag=4e-12, eog=150e-6)  # this can be highly data dependent

## Pick channels by type and names
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True, stim=False, exclude='bads')

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

# We can use a convenience function
eog_epochs = mne.preprocessing.create_eog_epochs(raw.copy().filter(1, None))
eog_epochs.average().plot_joint()

ecg_epochs = mne.preprocessing.create_ecg_epochs(raw.copy().filter(1, None))
ecg_epochs.average().plot_joint()

layouts = [mne.find_layout(raw.info, ch_type=ch) for ch in ("mag", "grad")]
projs_eog, _ = mne.preprocessing.compute_proj_eog(raw, n_mag=3, n_grad=3, average=True)
projs_ecg, _ = mne.preprocessing.compute_proj_ecg(raw, n_mag=3, n_grad=3, average=True)

mne.viz.plot_projs_topomap(projs_eog, layout=layouts)
mne.viz.plot_projs_topomap(projs_ecg, layout=layouts)

reject2 = dict(mag=reject['mag'], grad=reject['grad'])
epochs_clean = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                          picks=picks, baseline=baseline,
                          preload=False, reject=reject2)
epochs_clean.add_proj(projs_eog + projs_ecg)
epochs = epochs_clean

ERPSensor = 'MEG2311'
raw.plot_psd(fmax=40);
epochs.plot_image(picks=str(ERPSensor), sigma=1.);

epochs.plot()
print('Epochs Drop Log:')
print(epochs.drop_log)
epochs.event_id

print('Revised Epochs Drop Log:')
print(epochs.drop_log)
epochs_fname = raw_fname.replace('_raw.fif', '_epo.fif')
epochs.save(epochs_fname, overwrite=True)

evoked = epochs.average()
times = [0.0, 0.1, 0.18]
evoked.plot_topomap(ch_type='mag', times=times, proj=True);
evoked.plot_topomap(ch_type='grad', times=times, proj=True);

evoked.del_proj()  # delete previous proj
# take first for each sensor type
evoked.add_proj(projs_eog[::1] + projs_ecg[::2])  # selecting every third PCA component, starting with the first one
# evoked.add_proj(list(projs_eog[i] for i in [0, 1, 3, 4, 6, 7]) + list(projs_ecg[i] for i in [0, 1, 3, 4, 6, 7])) # allows a custom selection of PCA components for exclusion
evoked.apply_proj()  # apply

evoked.plot(spatial_colors=True, proj=True)

evoked.plot_topomap(times=np.linspace(0.05, 0.45, 8),
                    ch_type='mag', proj='True');  ## or false, or interactive
evoked.plot_topomap(times=np.linspace(0.05, 0.45, 8),
                    ch_type='grad', proj='True');

Contrast1 = 'cont017'
Contrast2 = 'cont100'
evoked_cond1 = epochs[str(Contrast1)].average()
evoked_cond2 = epochs[str(Contrast2)].average()

contrast = mne.combine_evoked([evoked_cond1, evoked_cond2], [0.5, -0.5])  # float list here indicates range of power

# Note that this combines evokeds taking into account the number of averaged epochs (to scale the noise variance)
print(evoked.nave)  # average of 12 epochs
print(contrast.nave)  # average of 116 epochs

print(contrast)


MinTime = -0.1  # time range in seconds, for contrast plot
MaxTime = 0.3
fig = contrast.copy().pick('grad').crop(MinTime, MaxTime).plot_joint()
fig = contrast.copy().pick('mag').crop(MinTime, MaxTime).plot_joint()

TopoMinTime = -0.1  # time range in seconds, for topographic plot
TopoMaxTime = 0.3
NoOfTopos = 5  # Number of topos within the specific time range above to produce
evoked_cond1 = epochs[str(Contrast1)].average()
evoked_cond2 = epochs[str(Contrast2)].average()
contrast = mne.combine_evoked([evoked_cond1, evoked_cond2], [0.5, -0.5])
contrast.plot_topomap(times=np.linspace(TopoMinTime, TopoMaxTime, NoOfTopos), ch_type=str('grad'))

conditions = ['left', 'right']
evoked_cond1 = epochs[conditions[0]].average().crop(TopoMinTime, TopoMaxTime)
evoked_cond2 = epochs[conditions[1]].average().crop(TopoMinTime, TopoMaxTime)
mne.viz.plot_evoked_topo([evoked_cond1, evoked_cond2])
