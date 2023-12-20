# %%
import mne
from mne.preprocessing import ICA
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mne.time_frequency import tfr_morlet
import scipy.signal
from mne.preprocessing import create_eog_epochs
from mne.preprocessing import create_ecg_epochs
import matplotlib
import matplotlib.pyplot as plt
import os.path
import glob
from mne.io import concatenate_raws, read_raw_fif
import matplotlib
from mne.evoked import combine_evoked


# %%
#baseline is defined
baseline = -0.5,-0.1
#the upper and lower limit of the time windown are defined
tmin = float(input('Give the lower limit of the time window'))
tmax = float(input("Give the upper limit of the time window"))
#minimum and maximum frequency are defined
fmin=float(input('Give the lower limit of frequency'))
fmax=float(input('Give the upper limit of the frequency'))  
#0.1, 40
main_path = input("Give your main input path")
#  C:/Users/titus/Desktop/Aalto/Kandi/

#lists are created for the events and the event files
events_list = []
event_file_list=[]
directory_events = input("Give the directory of your event files")
# C:/Users/titus/Desktop/Aalto/Kandi/ica
#events get sorted and read
for file_name in sorted(os.listdir(directory_events)):
    
    if "_latency_events.fif" in file_name:
        event_file = os.path.join(directory_events, file_name)
        print(f"Processing file: {event_file}")
        event_file_list.append(event_file)
        events = mne.read_events(event_file)
        events_list.append(events)

print(event_file_list)




# %%
#a list created for the ica files
ica_file_list = []

directory_ica = input("Give the directory of your ICA files")
#a raw file is defined, filtered and added to the list
for file_name in sorted(os.listdir(directory_ica)):
    if "_ica.fif" in file_name:
        ica_file = os.path.join(directory_ica, file_name)
        raw_org = mne.io.read_raw_fif(ica_file, preload=True)
        filtered = raw_org.copy().filter(fmin, fmax)
        
        ica_file_list.append((int(file_name.split('_')[0][1:]), filtered))

# C:/Users/titus/Desktop/Aalto/Kandi/ica/
print(ica_file_list)


# %% [markdown]
# 

# %%
#different event dictionarys are defined
event_dict_s1_attention = {
  "att_auditory/left": 4,
  "att_auditory/right": 5,
  "una_auditory/left": 400,
  "una_auditory/right": 500,
  }
event_dict_s2_attention = {
  "att_auditory/left": 64,
  "att_auditory/right": 80,
  "una_auditory/left": 6400,
  "una_auditory/right": 8000,}
#defining different event types
event_types = ['att_auditory/left', 'att_auditory/right', 'una_auditory/left', 'una_auditory/right']

# %%
#the numbers of the subjects are printed in case some are missing
x=0
subject_number_list=[]
for file in ica_file_list:
    number=ica_file_list[x][0]
    
    subject_number_list.append(number)
    
    x+=1
print('The numbers of the subjects are')
print(subject_number_list)

# %%
#lists are created for each event type and for epochs
results_path = input('Give the location where you want to save the results')
# C:/Users/titus/Desktop/Aalto/Kandi/results
list_att_l = []
list_att_r = []
list_una_l = []
list_una_r = []   
matplotlib.use('Qt5Agg')
%pylab qt5
n=0
epochs_list = []
#epochs are created, plotted and saved
for n, (subject_number_tuple, ica_file) in enumerate(zip(subject_number_list, ica_file_list)):
    subject_number = subject_number_list[n]
    dict_choice = input(f'Give the event dictionary you want to use with subject {str(subject_number)} (S1 or S2): ')

    if dict_choice == 'S1':
        chosen_dict = event_dict_s1_attention
    elif dict_choice == 'S2':
        chosen_dict = event_dict_s2_attention
    else:
        print('Not a valid name')
        continue
    
    epochs = mne.Epochs(ica_file[1], events_list[n], event_id=chosen_dict,
                        tmin=tmin, tmax=tmax, baseline=baseline, event_repeated='merge', preload=True)

    subject_epochs_folder = os.path.join(results_path, f"Subject_{subject_number}_results")
    os.makedirs(subject_epochs_folder, exist_ok=True)
    epochs.save(os.path.join(subject_epochs_folder, f"epochs_subject_{subject_number}.fif"), overwrite=True)
    epochs_list.append(epochs)

  

for epoch in epochs_list:
    att_l = epoch['att_auditory/left'].average()
    list_att_l.append(att_l)
    att_r = epoch['att_auditory/right'].average()
    list_att_r.append(att_r)
    una_l = epoch['una_auditory/left'].average()
    list_una_l.append(una_l)
    una_r = epoch['una_auditory/right'].average()
    list_una_r.append(una_r)
plotting_option = input('Do you want to plot and save the averaged ERPs? (y/n)')
if plotting_option == 'y':
    for i, (att_l, att_r, una_l, una_r, subject_number) in enumerate(
        zip(list_att_l, list_att_r, list_una_l, list_una_r, subject_number_list), start=1):
    
        subject_results_folder = os.path.join(results_path, f"Subject_{subject_number}_results")
        os.makedirs(subject_epochs_folder, exist_ok=True)

        fig_att_l = att_l.plot_joint(title=f'Subject {subject_number} - Attentive auditory left')
        fig_att_r = att_r.plot_joint(title=f'Subject {subject_number} - Attentive auditory right')
        fig_una_l = una_l.plot_joint(title=f'Subject {subject_number} - Unattentive auditory left')
        fig_una_r = una_r.plot_joint(title=f'Subject {subject_number} - Unattentive auditory right')

        fig_att_l.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Attentive_Auditory_Left.png"))
        fig_att_r.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Attentive_Auditory_Right.png"))
        fig_una_l.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Unattentive_Auditory_Left.png"))
        fig_una_r.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Unattentive_Auditory_Right.png"))

        fig_att_l.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Attentive_Auditory_Left.svg"))
        fig_att_r.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Attentive_Auditory_Right.svg"))
        fig_una_l.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Unattentive_Auditory_Left.svg"))
        fig_una_r.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Unattentive_Auditory_Right.svg"))
        subject_number+=1


        

else:
    print('No plots printed or saved')


# %%
#events are plotted for each subject
n=0

for event in events_list:
  subject_number = subject_number_list[n]
  dict = input(f'Give the event dictionary you want to use with subject {str(subject_number)} (S1 or S2): ')
  if dict == 'S1':
    chosen_dict= event_dict_s1_attention
  elif dict == 'S2': 
    chosen_dict = event_dict_s2_attention
  else:
    print('Not a valid name')
  subject_number=subject_number_list[n]  
  subject_results_folder = os.path.join(results_path, f"Subject_{subject_number}_results")
  fig = mne.viz.plot_events(events_list[n], sfreq=ica_file_list[n][1].info["sfreq"], first_samp=ica_file_list[n][1].first_samp, event_id=chosen_dict, on_missing='ignore')   
  fig.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_events.png"))
  fig.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_events.svg"))
  n+=1
  
  
  
  print('Plots saved')
   



# %%
#all channels are defined
channels_left = ['AF3','F7', 'F3','FC5','FC1','C3','CP1']
channels_right = ['AF4','F8','F4', 'FC6','FC2','C4','CP2']

# %%
#plotting EEG layout
ica_file_list[0][1].plot_sensors(show_names=True)
fig = ica_file_list[0][1].plot_sensors("3d",show_names=True)


# %%
#calculating grand average for each event type, side, state of attentiveness and a total grand average
evoked_list = []

for epoch in epochs_list:   
  for resp in event_types:
    evoked = epoch[resp].average()
    evoked_list.append(evoked)
    


num_subjects = len(evoked_list) // len(event_types)
grand_average_att = []
grand_average_una = []
for i, event_type in enumerate(event_types):
    grand_averages = []
    for j in range(num_subjects):
        index = j * len(event_types) + i
        evoked = evoked_list[index]
        if f'{event_type}' in evoked.comment:
            grand_averages.append(evoked)
    if grand_averages:
        grand_average = mne.grand_average(grand_averages)
        if 'att' in event_type:
            grand_average_att.append(grand_average)
        else:
            grand_average_una.append(grand_average)


ga_list=[]
for grand_average, event_type in zip(grand_average_att, event_types[:2]):
   ga_list.append(grand_average)
for grand_average, event_type in zip(grand_average_una, event_types[2:]):
   ga_list.append(grand_average)
plotting_ga = input('Do you want to plot the grand averages? (y/n)')
if plotting_ga == 'y':
   for grand_average, event_type in zip(grand_average_att, event_types[:2]):
      grand_average.plot_joint(title=f'Grand average {event_type} ')
   for grand_average, event_type in zip(grand_average_una, event_types[2:]):
      grand_average.plot_joint(title=f'Grand average {event_type} ')
   
else:
   print('Grand averages calculated')
grand_average_total = [grand_average_att, grand_average_una]


# %%
#time windows for N1 are defined
tmin_n1 = 0.08
tmax_n1 = 0.120
#lists for results are created
list_ch_att_l_n1=[]
list_ch_att_r_n1=[]
list_ch_una_l_n1=[]
list_ch_una_r_n1=[]
list_lat_att_l_n1=[]
list_amp_att_l_n1=[]
list_lat_att_r_n1=[]
list_amp_att_r_n1=[]
list_lat_una_l_n1=[]
list_amp_una_l_n1=[]
list_lat_una_r_n1=[]
list_amp_una_r_n1=[]
list_average_amp_n1=[]
list_average_lat_n1=[]
a=0
b=0
c=0
d=0
#peak amplitude, latency and the channel are defined
print('Attentive auditory left N1 values')
for att_l in list_att_l:
    try:
        ch_al_n1, lat_al_n1, amp_al_n1 = list_att_l[a].get_peak(ch_type="eeg", tmin=tmin_n1, tmax=tmax_n1, mode="neg", return_amplitude=True)
        a+=1
        amp_al_n1=amp_al_n1*1000000
        list_lat_att_l_n1.append(lat_al_n1)
        list_amp_att_l_n1.append(amp_al_n1)
        list_ch_att_l_n1.append(ch_al_n1)
        print('Channel :', ch_al_n1)
        print('Latency :',lat_al_n1)
        print('Amplitude :',amp_al_n1)
    except ValueError: 
        print('No negative peak found within the specified time window.')
        #value 0 is added beacuse otherwise the length of list would change and cause issues
        list_amp_att_l_n1.append(0)
        list_lat_att_l_n1.append(0)
        list_ch_att_l_n1.append(0)    
print()    
print('Attentive auditory right N1 values')
for att_r in list_att_r:
    try:
        ch_ar_n1, lat_ar_n1, amp_ar_n1 = list_att_r[b].get_peak(ch_type="eeg", tmin=tmin_n1, tmax=tmax_n1, mode="neg", return_amplitude=True)
        b+=1
        amp_ar_n1=amp_ar_n1*1000000
        list_ch_att_r_n1.append(ch_ar_n1)
        list_lat_att_r_n1.append(lat_ar_n1)
        list_amp_att_r_n1.append(amp_ar_n1)
        print('Channel :', ch_ar_n1)
        print('Latency :',lat_ar_n1)
        print('Amplitude :',amp_ar_n1)
    except ValueError:
        print('No negative peak found within the specified time window.')   
        list_lat_att_r_n1.append(0)     
        list_amp_att_r_n1.append(0)
        list_ch_att_r_n1.append(0)     
print()
print('Unattentive auditory left N1 values')
for una_l in list_una_l:
    try:
        ch_ul_n1, lat_ul_n1, amp_ul_n1 = list_una_l[c].get_peak(ch_type="eeg", tmin=tmin_n1, tmax=tmax_n1, mode="neg", return_amplitude=True)
        c+=1
        amp_ul_n1=amp_ul_n1*1000000
        list_ch_una_l_n1.append(ch_ul_n1)
        list_lat_una_l_n1.append(lat_ul_n1)
        list_amp_una_l_n1.append(amp_ul_n1)
        print('Channel :', ch_ul_n1)
        print('Latency :',lat_ul_n1)
        print('Amplitude :',amp_ul_n1)
    except ValueError:
        print('No negative peak found within the specified time window.')
        list_amp_una_l_n1.append(0)
        list_lat_una_l_n1.append(0)        
        list_ch_una_l_n1.append(0)
print()
print('Unattentive auditory right N1 values')
for una_r in list_una_r:
    try:
        ch_ur_n1, lat_ur_n1, amp_ur_n1 = list_una_r[d].get_peak(ch_type="eeg", tmin=tmin_n1, tmax=tmax_n1, mode="neg", return_amplitude=True)
        d+=1
        amp_ur_n1=amp_ur_n1*1000000
        list_ch_una_r_n1.append(ch_ur_n1)
        list_lat_una_r_n1.append(lat_ur_n1)
        list_amp_una_r_n1.append(amp_ur_n1)
        print('Channel :', ch_ur_n1)
        print('Latency :',lat_ur_n1)
        print('Amplitude :',amp_ur_n1)
    except ValueError:
        print('No negative peak found within the specified time window.')
        list_amp_una_r_n1.append(0)
        list_lat_una_r_n1.append(0)
        list_ch_una_r_n1.append(0)    
print()
a = 0
for n in range(len(epochs_list)):
    average_amp_n1_subject = (list_amp_att_l_n1[n] + list_amp_att_r_n1[n] + list_amp_una_l_n1[n] + list_amp_una_r_n1[n]) / 4
    list_average_amp_n1.append(average_amp_n1_subject)
    average_lat_n1_subject = (list_lat_att_l_n1[n] + list_lat_att_r_n1[n] + list_lat_una_l_n1[n] + list_lat_una_r_n1[n]) / 4
    list_average_lat_n1.append(average_lat_n1_subject)
    print(f"Average amplitude for subject S{subject_number_list[a]} is {average_amp_n1_subject}")
    print(f"Average latency for subject S{subject_number_list[a]} is {average_lat_n1_subject}")
    print()
    a += 1
    



#averages are calculated
average_amp_att_l_n1 = sum(list_amp_att_l_n1) / len(list_amp_att_l_n1)
average_amp_att_r_n1 = sum(list_amp_att_r_n1) / len(list_amp_att_r_n1)
average_amp_una_l_n1 = sum(list_amp_una_l_n1) / len(list_amp_una_l_n1)
average_amp_una_r_n1 = sum(list_amp_una_r_n1) / len(list_amp_una_r_n1)

average_amp_n1 = (average_amp_att_l_n1+average_amp_att_r_n1+average_amp_una_l_n1+average_amp_una_r_n1)/4
print("Total average amplitude is",average_amp_n1)

average_lat_att_l_n1 = sum(list_lat_att_l_n1) / len(list_lat_att_l_n1)
average_lat_att_r_n1 = sum(list_lat_att_r_n1) / len(list_lat_att_r_n1)
average_lat_una_l_n1 = sum(list_lat_una_l_n1) / len(list_lat_una_l_n1)
average_lat_una_r_n1 = sum(list_lat_una_r_n1) / len(list_lat_una_r_n1)

average_lat_n1 = (average_lat_att_l_n1+average_lat_att_r_n1+average_lat_una_l_n1+average_lat_una_r_n1)/4
print("Total average latency is",average_lat_n1)


# %%
#time window for P2
tmin_p2 = 0.150
tmax_p2=0.275
#lists for results
list_ch_att_l_p2=[]
list_ch_att_r_p2=[]
list_ch_una_l_p2=[]
list_ch_una_r_p2=[]
list_lat_att_l_p2=[]
list_amp_att_l_p2=[]
list_lat_att_r_p2=[]
list_amp_att_r_p2=[]
list_lat_una_l_p2=[]
list_amp_una_l_p2=[]
list_lat_una_r_p2=[]
list_amp_una_r_p2=[]
list_average_amp_p2=[]
list_average_lat_p2=[]
e=0
f=0
g=0
h=0
#same calculations than with N1
print('Attentive auditory left P2 values')
for att_l in list_att_l:
    try:
        ch_al_p2, lat_al_p2, amp_al_p2 = list_att_l[e].get_peak(ch_type="eeg", tmin=tmin_p2, tmax=tmax_p2, mode="pos", return_amplitude=True)
        e+=1
        amp_al_p2=amp_al_p2*1000000
        list_ch_att_l_p2.append(ch_al_p2)
        list_lat_att_l_p2.append(lat_al_p2)
        list_amp_att_l_p2.append(amp_al_p2)
        print('Channel :', ch_al_p2)
        print('Latency :',lat_al_p2)
        print('Amplitude :',amp_al_p2)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_amp_att_l_p2.append(0)
        list_lat_att_l_p2.append(0)
        list_ch_att_l_p2.append(0)
        
print()    
print('Attentive auditory right P2 values')
for att_r in list_att_r:
    try:
        ch_ar_p2, lat_ar_p2, amp_ar_p2 = list_att_r[f].get_peak(ch_type="eeg", tmin=tmin_p2, tmax=tmax_p2, mode="pos", return_amplitude=True)
        f+=1
        amp_ar_p2=amp_ar_p2*1000000
        list_ch_att_r_p2.append(ch_ar_p2)
        list_lat_att_r_p2.append(lat_ar_p2)
        list_amp_att_r_p2.append(amp_ar_p2)
        print('Channel :', ch_ar_p2)
        print('Latency :',lat_ar_p2)
        print('Amplitude :',amp_ar_p2)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_amp_att_r_p2.append(0)
        list_lat_att_r_p2.append(0)  
        list_ch_att_r_p2.append(0)  
print()
print('Unattentive auditory left P2 values')
for una_l in list_una_l:
    try:
        ch_ul_p2, lat_ul_p2, amp_ul_p2 = list_una_l[g].get_peak(ch_type="eeg", tmin=tmin_p2, tmax=tmax_p2, mode="pos", return_amplitude=True)
        g+=1
        amp_ul_p2=amp_ul_p2*1000000
        list_ch_una_l_p2.append(ch_ul_p2)
        list_lat_una_l_p2.append(lat_ul_p2)
        list_amp_una_l_p2.append(amp_ul_p2)
        print('Channel :', ch_ul_p2)
        print('Latency :',lat_ul_p2)
        print('Amplitude :',amp_ul_p2)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_amp_una_l_p2.append(0)
        list_lat_una_l_p2.append(0)
        list_ch_una_l_p2.append(0)
print()
print('Unattentive auditory right P2 values')
for una_r in list_una_r:
    try:
        ch_ur_p2, lat_ur_p2, amp_ur_p2 = list_una_r[h].get_peak(ch_type="eeg", tmin=tmin_p2, tmax=tmax_p2, mode="pos", return_amplitude=True)
        h+=1
        amp_ur_p2=amp_ur_p2*1000000
        list_ch_una_r_p2.append(ch_ul_p2)
        list_lat_una_r_p2.append(lat_ur_p2)
        list_amp_una_r_p2.append(amp_ur_p2)
        print('Channel :', ch_ur_p2)
        print('Latency :',lat_ur_p2)
        print('Amplitude :',amp_ur_p2)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_amp_una_r_p2.append(0)
        list_lat_una_r_p2.append(0)
        list_ch_una_r_p2.append(0)
print()
a = 0
for n in range(len(epochs_list)):
    average_amp_p2_subject = (list_amp_att_l_p2[n] + list_amp_att_r_p2[n] + list_amp_una_l_p2[n] + list_amp_una_r_p2[n]) / 4
    list_average_amp_p2.append(average_amp_p2_subject)
    average_lat_p2_subject = (list_lat_att_l_p2[n] + list_lat_att_r_p2[n] + list_lat_una_l_p2[n] + list_lat_una_r_p2[n]) / 4
    list_average_lat_p2.append(average_lat_p2_subject)
    print(f"Average amplitude for subject S{subject_number_list[a]} is {average_amp_p2_subject}")
    print(f"Average latency for subject S{subject_number_list[a]} is {average_lat_p2_subject}")
    print()
    a += 1



average_amp_att_l_p2 = sum(list_amp_att_l_p2) / len(list_amp_att_l_p2)
average_amp_att_r_p2 = sum(list_amp_att_r_p2) / len(list_amp_att_r_p2)
average_amp_una_l_p2 = sum(list_amp_una_l_p2) / len(list_amp_una_l_p2)
average_amp_una_r_p2 = sum(list_amp_una_r_p2) / len(list_amp_una_r_p2)

average_amp_p2 = (average_amp_att_l_p2+average_amp_att_r_p2+average_amp_una_l_p2+average_amp_una_r_p2)/4
print("Average amplitude is",average_amp_p2)

average_lat_att_l_p2 = sum(list_lat_att_l_p2) / len(list_lat_att_l_p2)
average_lat_att_r_p2 = sum(list_lat_att_r_p2) / len(list_lat_att_r_p2)
average_lat_una_l_p2 = sum(list_lat_una_l_p2) / len(list_lat_una_l_p2)
average_lat_una_r_p2 = sum(list_lat_una_r_p2) / len(list_lat_una_r_p2)

average_lat_p2 = (average_lat_att_l_p2+average_lat_att_r_p2+average_lat_una_l_p2+average_lat_una_r_p2)/4
print("Average latency is",average_lat_p2)

# %%
#P3 time windows
tmin_p3 = 0.300
tmax_p3=0.440
list_ch_att_l_p3=[]
list_ch_att_r_p3=[]
list_ch_una_l_p3=[]
list_ch_una_r_p3=[]
list_lat_att_l_p3=[]
list_amp_att_l_p3=[]
list_lat_att_r_p3=[]
list_amp_att_r_p3=[]
list_lat_una_l_p3=[]
list_amp_una_l_p3=[]
list_lat_una_r_p3=[]
list_amp_una_r_p3=[]
list_average_amp_p3=[]
list_average_lat_p3=[]
o=0
p=0
q=0
r=0
#same calculations again
print('Attentive auditory left P3 values')
for att_l in list_att_l:
    try:
        ch_al_p3, lat_al_p3, amp_al_p3 = list_att_l[o].get_peak(ch_type="eeg", tmin=tmin_p3, tmax=tmax_p3, mode="pos", return_amplitude=True)
        o+=1
        amp_al_p3=amp_al_p3*1000000
        list_ch_att_l_p3.append(ch_al_p3)
        list_lat_att_l_p3.append(lat_al_p3)
        list_amp_att_l_p3.append(amp_al_p3)
        print('Channel :', ch_al_p3)
        print('Latency :',lat_al_p3)
        print('Amplitude :',amp_al_p3)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_amp_att_l_p3.append(0)
        list_lat_att_l_p3.append(0)
        list_ch_att_l_p3.append(0)
        
print()    
print('Attentive auditory right P3 values')
for att_r in list_att_r:
    try:
        ch_ar_p3, lat_ar_p3, amp_ar_p3 = list_att_r[p].get_peak(ch_type="eeg", tmin=tmin_p3, tmax=tmax_p3, mode="pos", return_amplitude=True)
        p+=1
        amp_ar_p3=amp_ar_p3*1000000
        list_ch_att_r_p3.append(ch_ar_p3)
        list_lat_att_r_p3.append(lat_ar_p3)
        list_amp_att_r_p3.append(amp_ar_p3)
        print('Channel :', ch_ar_p3)
        print('Latency :',lat_ar_p3)
        print('Amplitude :',amp_ar_p3)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_amp_att_r_p3.append(0)
        list_lat_att_r_p3.append(0)
        list_ch_att_r_p3.append(0)
print()
print('Unattentive auditory left P3 values')
for una_l in list_una_l:
    try:
        ch_ul_p3, lat_ul_p3, amp_ul_p3 = list_una_l[q].get_peak(ch_type="eeg", tmin=tmin_p3, tmax=tmax_p3, mode="pos", return_amplitude=True)
        q+=1
        amp_ul_p3=amp_ul_p3*1000000
        list_ch_una_l_p3.append(ch_ul_p3)
        list_lat_una_l_p3.append(lat_ul_p3)
        list_amp_una_l_p3.append(amp_ul_p3)
        print('Channel :', ch_ul_p3)
        print('Latency :',lat_ul_p3)
        print('Amplitude :',amp_ul_p3)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_amp_una_l_p3.append(0)
        list_lat_una_l_p3.append(0)    
        list_ch_una_l_p3.append(0)
print()
print('Unattentive auditory right P3 values')
for una_r in list_una_r:
    try:
        ch_ur_p3, lat_ur_p3, amp_ur_p3 = list_una_r[r].get_peak(ch_type="eeg", tmin=tmin_p3, tmax=tmax_p3, mode="pos", return_amplitude=True)
        r+=1
        amp_ur_p3=amp_ur_p3*1000000
        list_ch_una_r_p3.append(ch_ur_p3)
        list_lat_una_r_p3.append(lat_ur_p3)
        list_amp_una_r_p3.append(amp_ur_p3)
        print('Channel :', ch_ur_p3)
        print('Latency :',lat_ur_p3)
        print('Amplitude :',amp_ur_p3)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_amp_una_r_p3.append(0)
        list_lat_una_r_p3.append(0) 
        list_ch_una_r_p3.append(0)


print()
a = 0
for n in range(len(epochs_list)):
    average_amp_p3_subject = (list_amp_att_l_p3[n] + list_amp_att_r_p3[n] + list_amp_una_l_p3[n] + list_amp_una_r_p3[n]) / 4
    list_average_amp_p3.append(average_amp_p3_subject)
    average_lat_p3_subject = (list_lat_att_l_p3[n] + list_lat_att_r_p3[n] + list_lat_una_l_p3[n] + list_lat_una_r_p3[n]) / 4
    list_average_lat_p3.append(average_lat_p3_subject)
    print(f"Average amplitude for subject S{subject_number_list[a]} is {average_amp_p3_subject}")
    print(f"Average latency for subject S{subject_number_list[a]} is {average_lat_p3_subject}")
    print()
    a += 1


average_amp_att_l_p3 = sum(list_amp_att_l_p3) / len(list_amp_att_l_p3)
average_amp_att_r_p3 = sum(list_amp_att_r_p3) / len(list_amp_att_r_p3)
average_amp_una_l_p3 = sum(list_amp_una_l_p3) / len(list_amp_una_l_p3)
average_amp_una_r_p3 = sum(list_amp_una_r_p3) / len(list_amp_una_r_p3)

average_amp_p3 = (average_amp_att_l_p3+average_amp_att_r_p3+average_amp_una_l_p3+average_amp_una_r_p3)/4
print("Average amplitude is",average_amp_p3)

average_lat_att_l_p3 = sum(list_lat_att_l_p3) / len(list_lat_att_l_p3)
average_lat_att_r_p3 = sum(list_lat_att_r_p3) / len(list_lat_att_r_p3)
average_lat_una_l_p3 = sum(list_lat_una_l_p3) / len(list_lat_una_l_p3)
average_lat_una_r_p3 = sum(list_lat_una_r_p3) / len(list_lat_una_r_p3)

average_lat_p3 = (average_lat_att_l_p3+average_lat_att_r_p3+average_lat_una_l_p3+average_lat_una_r_p3)/4
print("Average latency is",average_lat_p3)

# %%
#evoked object comments are modified for being accurate in the plot legends
original_comment_1 = ga_list[0].comment
print("Original Comment:", original_comment_1)
new_comment_1 = "att_left"
ga_list[0].comment = new_comment_1
modified_comment_1 = ga_list[0].comment
print("Modified Comment:", modified_comment_1)

original_comment_2 = ga_list[1].comment
print("Original Comment:", original_comment_2)
new_comment_2 = "att_right"
ga_list[1].comment = new_comment_2
modified_comment_2 = ga_list[1].comment
print("Modified Comment:", modified_comment_2)

original_comment_3 = ga_list[2].comment
print("Original Comment:", original_comment_3)
new_comment_3 = "una_left"
ga_list[2].comment = new_comment_3
modified_comment_3 = ga_list[2].comment
print("Modified Comment:", modified_comment_3)


original_comment_4 = ga_list[3].comment
print("Original Comment:", original_comment_4)
new_comment_4 = "una_right"
ga_list[3].comment = new_comment_4
modified_comment_4 = ga_list[3].comment
print("Modified Comment:", modified_comment_4)

# %%

#comparison plots una vs att are created and saved

saving_path = os.path.join(results_path, 'mutual_plots')
os.makedirs(saving_path, exist_ok=True)

fig_r = mne.viz.plot_compare_evokeds([ga_list[0], ga_list[2]], combine='mean', picks=channels_left,legend=True,
                                     title='Grand average comparison plot left att vs una')
fig_r[0].savefig(os.path.join(saving_path, "comparison_plot_r.png"), format="png")
fig_r[0].savefig(os.path.join(saving_path, "comparison_plot_r.svg"), format="svg")

fig_l = mne.viz.plot_compare_evokeds([ga_list[1], ga_list[3]], combine='mean', picks=channels_right,legend=True,
                                      title='Grand average comparison plot right att vs una ')
fig_l[0].savefig(os.path.join(saving_path, "comparison_plot_l.png"), format="png")
fig_l[0].savefig(os.path.join(saving_path, "comparison_plot_l.svg"), format="svg")

print('Figures saved to', saving_path)



# %%
#comparison plots with EEG layout and confidence intervals are created and saved
print('Plotting left side grand average comparison plot. Blue represents attentive state and orange unanttentive')
figures_l = mne.viz.plot_compare_evokeds([ga_list[0], ga_list[2]], title='Left comparison plot', axes='topo')
print('Plotting right side grand average comparison plot. Blue represents attentive state and orange unanttentive')
figures_r = mne.viz.plot_compare_evokeds([ga_list[1], ga_list[3]], title='Right side comparison plot', axes="topo")
print('Plotting total grand average comparison plot. Blue represents attentive state and orange unanttentive')
figures_total = mne.viz.plot_compare_evokeds(grand_average_total, title='Grand average with confidence intervals', axes="topo")


saving_path = os.path.join(results_path, 'mutual_plots')

for i, fig in enumerate(figures_l):
    fig.savefig(os.path.join(saving_path, f"left_side_comparison_layout.png"))
    fig.savefig(os.path.join(saving_path, f"left_side_comparison_layout.svg"))


for i, fig in enumerate(figures_r):
    fig.savefig(os.path.join(saving_path, f"right_side_comparison_layout.png"))
    fig.savefig(os.path.join(saving_path, f"right_side_comparison_layout.svg"))


for i, fig in enumerate(figures_total):
    fig.savefig(os.path.join(saving_path, f"total_grand_average_layout.png"))
    fig.savefig(os.path.join(saving_path, f"total_grand_average_layout.svg"))

print('Figures saved to',saving_path)



# %%
#user picks the channels to used
picked_channels_list_r = []
picked_channels_list_l = []
picked_channels_r = input("Give the right side channels you want to use separated by comma: ")
picked_channels_l = input("Give the left side channels you want to use separated by comma: ")

picked_channels_list_r = [channel.strip() for channel in picked_channels_r.split(',')]
picked_channels_list_l = [channel.strip() for channel in picked_channels_l.split(',')]
#r F4,F8,FC6
#l F7,F3,FC5
print('The channels you have selected are')
print('Right side channels:')
print(picked_channels_list_l)
print('Left side channels')
print(picked_channels_list_r)



# %%
#epochs are averaged again with the new channel picks, plotted and saved
list_att_l_3ch=[]
list_att_r_3ch=[]
list_una_l_3ch=[]
list_una_r_3ch=[]
n=0
for epoch in epochs_list:
    att_l_3_ch = epoch['att_auditory/left'].average(picks = picked_channels_list_l)
    list_att_l_3ch.append(att_l_3_ch)

    att_r_3ch = epoch['att_auditory/right'].average(picks = picked_channels_list_r)
    list_att_r_3ch.append(att_r_3ch)

    una_l_3_ch = epoch['una_auditory/left'].average(picks = picked_channels_list_l)
    list_una_l_3ch.append(una_l_3_ch)
    
    una_r_3_ch = epoch['una_auditory/right'].average(picks = picked_channels_list_r)
    list_una_r_3ch.append(una_r_3_ch)
    
    
n=0    
plotting_option = input('Do you want to plot and save the averaged ERPs? (y/n)')
if plotting_option == 'y':
    for i, (att_l_3ch, att_r_3ch, una_l_3ch, una_r_3ch, subject_number) in enumerate(
        zip(list_att_l_3ch, list_att_r_3ch, list_una_l_3ch, list_una_r_3ch, subject_number_list), start=1):
        subject_number=subject_number_list[n]
        subject_results_folder = os.path.join(results_path, f"Subject_{subject_number}_results")
        os.makedirs(subject_epochs_folder, exist_ok=True)

        fig_att_l_3ch = att_l_3ch.plot_joint(title=f'Attentive auditory left picked channels Subject {subject_number_list[n]}', picks=picked_channels_list_l)
        fig_att_r_3ch = att_r_3ch.plot_joint(title=f'Attentive auditory right picked channels Subject {subject_number_list[n]}', picks=picked_channels_list_r)
        fig_una_l_3ch = una_l_3ch.plot_joint(title=f'Unattentive auditory left picked channels Subject {subject_number_list[n]}', picks=picked_channels_list_l)
        fig_una_r_3ch = una_r_3ch.plot_joint(title=f'Unattentive auditory right picked channels Subject {subject_number_list[n]}', picks=picked_channels_list_r)

        fig_att_l_3ch.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Attentive_Auditory_Left_3ch.png"))
        fig_att_r_3ch.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Attentive_Auditory_Right_3ch.png"))
        fig_una_l_3ch.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Unattentive_Auditory_Left_3ch.png"))
        fig_una_r_3ch.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Unattentive_Auditory_Right_3ch.png"))

        fig_att_l_3ch.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Attentive_Auditory_Left_3ch.svg"))
        fig_att_r_3ch.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Attentive_Auditory_Right_3ch.svg"))
        fig_una_l_3ch.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Unattentive_Auditory_Left_3ch.svg"))
        fig_una_r_3ch.savefig(os.path.join(subject_results_folder, f"Subject_{subject_number}_Unattentive_Auditory_Right_3ch.svg"))
        n+=1


        

else:
    print('No plots printed or saved')


    n+=1

# %%
#more averaging is done with the new picks

list_att_l_3ch = []
list_att_r_3ch = []
list_una_l_3ch = []
list_una_r_3ch = [] 
evoked_list_l = []
evoked_list_r = []

for epoch in epochs_list:
  att_l_3ch = epoch['att_auditory/left'].average(picks=picked_channels_list_l)
  list_att_l_3ch.append(att_l_3ch)
  att_r_3ch = epoch['att_auditory/right'].average(picks=picked_channels_list_r)
  list_att_r_3ch.append(att_r_3ch)
  una_l_3ch = epoch['una_auditory/left'].average(picks=picked_channels_list_l)
  list_una_l_3ch.append(una_l_3ch)
  una_r_3ch = epoch['una_auditory/right'].average(picks=picked_channels_list_r)
  list_una_r_3ch.append(una_r_3ch)   
  for resp in event_types:
    evoked_l = epoch[resp].average(picks=picked_channels_list_l)
    evoked_r = epoch[resp].average(picks=picked_channels_list_r)
    evoked_list_l.append(evoked_l)
    evoked_list_r.append(evoked_r)
    
    


# %%
#same N1 calculations are done with the new channel picks
list_ch_att_l_n1_3ch = []
list_ch_att_r_n1_3ch = []
list_ch_una_l_n1_3ch = []
list_ch_una_r_n1_3ch = []
list_lat_att_l_n1_3ch = []
list_amp_att_l_n1_3ch = []
list_lat_att_r_n1_3ch = []
list_amp_att_r_n1_3ch = []
list_lat_una_l_n1_3ch = []
list_amp_una_l_n1_3ch = []
list_lat_una_r_n1_3ch = []
list_amp_una_r_n1_3ch = []
list_average_amp_n1_3ch = []
list_average_lat_n1_3ch = []
o = 0
p = 0
q = 0
r = 0
print('Attentive auditory left N1 values')
for att_l_3ch in list_att_l_3ch:
    try:
        ch_al_n1_3ch, lat_al_n1_3ch, amp_al_n1_3ch = list_att_l_3ch[o].get_peak(ch_type="eeg", tmin=tmin_n1, tmax=tmax_n1, mode="neg", return_amplitude=True)
        o += 1
        amp_al_n1_3ch=amp_al_n1_3ch*1000000
        list_ch_att_l_n1_3ch.append(ch_al_n1_3ch)
        list_lat_att_l_n1_3ch.append(lat_al_n1_3ch)
        list_amp_att_l_n1_3ch.append(amp_al_n1_3ch)
        print('Channel :', ch_al_n1_3ch)
        print('Latency :', lat_al_n1_3ch)
        print('Amplitude :', amp_al_n1_3ch)
    except ValueError:
        print('No negative peak found within the specified time window.')
        list_lat_att_l_n1_3ch.append(0)
        list_amp_att_l_n1_3ch.append(0)
        list_ch_att_l_n1_3ch.append(0)
print()
print('Attentive auditory right N1 values')
for att_r_3ch in list_att_r_3ch:
    try:
        ch_ar_n1_3ch, lat_ar_n1_3ch, amp_ar_n1_3ch = list_att_r_3ch[p].get_peak(ch_type="eeg", tmin=tmin_n1, tmax=tmax_n1, mode="neg", return_amplitude=True)
        p += 1
        amp_ar_n1_3ch=amp_ar_n1_3ch*1000000
        list_ch_att_r_n1_3ch.append(ch_ar_n1_3ch)
        list_lat_att_r_n1_3ch.append(lat_ar_n1_3ch)
        list_amp_att_r_n1_3ch.append(amp_ar_n1_3ch)
        print('Channel :', ch_ar_n1_3ch)
        print('Latency :', lat_ar_n1_3ch)
        print('Amplitude :', amp_ar_n1_3ch)
    except ValueError:
        print('No negative peak found within the specified time window.')
        list_lat_att_r_n1_3ch.append(0)
        list_amp_att_r_n1_3ch.append(0)
        list_ch_att_r_n1_3ch.append(0)
print()
print('Unattentive auditory left N1 values')
for una_l_3ch in list_una_l_3ch:
    try:
        ch_ul_n1_3ch, lat_ul_n1_3ch, amp_ul_n1_3ch = list_una_l_3ch[q].get_peak(ch_type="eeg", tmin=tmin_n1, tmax=tmax_n1, mode="neg", return_amplitude=True)
        q += 1
        amp_ul_n1_3ch=amp_ul_n1_3ch*1000000
        list_ch_una_l_n1_3ch.append(ch_ul_n1_3ch)
        list_lat_una_l_n1_3ch.append(lat_ul_n1_3ch)
        list_amp_una_l_n1_3ch.append(amp_ul_n1_3ch)
        print('Channel :', ch_ul_n1_3ch)
        print('Latency :', lat_ul_n1_3ch)
        print('Amplitude :', amp_ul_n1_3ch)
    except ValueError:
        print('No negative peak found within the specified time window.')
        list_lat_una_l_n1_3ch.append(0)
        list_amp_una_l_n1_3ch.append(0)
        list_ch_una_l_n1_3ch.append(0)
print()
print('Unattentive auditory right N1 values')
for una_r_3ch in list_una_r_3ch:
    try:
        ch_ur_n1_3ch, lat_ur_n1_3ch, amp_ur_n1_3ch = list_una_r_3ch[r].get_peak(ch_type="eeg", tmin=tmin_n1, tmax=tmax_n1, mode="neg", return_amplitude=True)
        r += 1
        amp_ur_n1_3ch=amp_ur_n1_3ch*1000000
        list_ch_una_r_n1_3ch.append(ch_ur_n1_3ch)
        list_lat_una_r_n1_3ch.append(lat_ur_n1_3ch)
        list_amp_una_r_n1_3ch.append(amp_ur_n1_3ch)
        print('Channel :', ch_ur_n1_3ch)
        print('Latency :', lat_ur_n1_3ch)
        print('Amplitude :', amp_ur_n1_3ch)
    except ValueError:
        print('No negative peak found within the specified time window.')
        list_lat_una_r_n1_3ch.append(0)
        list_amp_una_r_n1_3ch.append(0)
        list_ch_una_r_n1_3ch.append(0)
print()
a = 0
for n in range(len(epochs_list)):
    average_amp_n1_subject_3ch = (list_amp_att_l_n1_3ch[n] + list_amp_att_r_n1_3ch[n] + list_amp_una_l_n1_3ch[n] + list_amp_una_r_n1_3ch[n]) / 4
    list_average_amp_n1_3ch.append(average_amp_n1_subject_3ch)
    average_lat_n1_subject_3ch = (list_lat_att_l_n1_3ch[n] + list_lat_att_r_n1_3ch[n] + list_lat_una_l_n1_3ch[n] + list_lat_una_r_n1_3ch[n]) / 4
    list_average_lat_n1_3ch.append(average_lat_n1_subject_3ch)
    print(f"Average amplitude for subject S{subject_number_list[a]} is {average_amp_n1_subject_3ch}")
    print(f"Average latency for subject S{subject_number_list[a]} is {average_lat_n1_subject_3ch}")
    print()
    a += 1
average_amp_att_l_n1_3ch = sum(list_amp_att_l_n1_3ch) / len(list_amp_att_l_n1_3ch)
average_amp_att_r_n1_3ch = sum(list_amp_att_r_n1_3ch) / len(list_amp_att_r_n1_3ch)
average_amp_una_l_n1_3ch = sum(list_amp_una_l_n1_3ch) / len(list_amp_una_l_n1_3ch)
average_amp_una_r_n1_3ch = sum(list_amp_una_r_n1_3ch) / len(list_amp_una_r_n1_3ch)
average_amp_n1_3ch = (average_amp_att_l_n1_3ch + average_amp_att_r_n1_3ch + average_amp_una_l_n1_3ch + average_amp_una_r_n1_3ch) / 4
print("Total average amplitude is", average_amp_n1_3ch)
average_lat_att_l_n1_3ch = sum(list_lat_att_l_n1_3ch) / len(list_lat_att_l_n1_3ch)
average_lat_att_r_n1_3ch = sum(list_lat_att_r_n1_3ch) / len(list_lat_att_r_n1_3ch)
average_lat_una_l_n1_3ch = sum(list_lat_una_l_n1_3ch) / len(list_lat_una_l_n1_3ch)
average_lat_una_r_n1_3ch = sum(list_lat_una_r_n1_3ch) / len(list_lat_una_r_n1_3ch)
average_lat_n1_3ch = (average_lat_att_l_n1_3ch + average_lat_att_r_n1_3ch + average_lat_una_l_n1_3ch + average_lat_una_r_n1_3ch) / 4
print("Total average latency is", average_lat_n1_3ch)

# %%
#similar P2 calculations, only difference is the channels used
list_ch_att_l_p2_3ch=[]
list_ch_att_r_p2_3ch=[]
list_ch_una_l_p2_3ch=[]
list_ch_una_r_p2_3ch=[]
list_lat_att_l_p2_3ch = []
list_amp_att_l_p2_3ch = []
list_lat_att_r_p2_3ch = []
list_amp_att_r_p2_3ch = []
list_lat_una_l_p2_3ch = []
list_amp_una_l_p2_3ch = []
list_lat_una_r_p2_3ch = []
list_amp_una_r_p2_3ch = []
list_average_amp_p2_3ch = []
list_average_lat_p2_3ch = []

o = 0
p = 0
q = 0
r = 0

print('Attentive auditory left P2 values')
for att_l_3ch in list_att_l_3ch:
    try:
        ch_al_p2_3ch, lat_al_p2_3ch, amp_al_p2_3ch = list_att_l_3ch[o].get_peak(ch_type="eeg", tmin=tmin_p2, tmax=tmax_p2, mode="pos", return_amplitude=True)
        o += 1
        amp_al_p2_3ch=amp_al_p2_3ch*1000000
        list_ch_att_l_p2_3ch.append(ch_al_p2_3ch)
        list_lat_att_l_p2_3ch.append(lat_al_p2_3ch)
        list_amp_att_l_p2_3ch.append(amp_al_p2_3ch)
        print('Channel :', ch_al_p2_3ch)
        print('Latency :', lat_al_p2_3ch)
        print('Amplitude :', amp_al_p2_3ch)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_lat_att_l_p2_3ch.append(0)
        list_amp_att_l_p2_3ch.append(0)
        list_ch_att_l_p2_3ch.append(0)

print()
print('Attentive auditory right P2 values')
for att_r_3ch in list_att_r_3ch:
    try:
        ch_ar_p2_3ch, lat_ar_p2_3ch, amp_ar_p2_3ch = list_att_r_3ch[p].get_peak(ch_type="eeg", tmin=tmin_p2, tmax=tmax_p2, mode="pos", return_amplitude=True)
        p += 1
        amp_ar_p2_3ch=amp_ar_p2_3ch*1000000
        list_ch_att_r_p2_3ch.append(ch_ar_p2_3ch)
        list_lat_att_r_p2_3ch.append(lat_ar_p2_3ch)
        list_amp_att_r_p2_3ch.append(amp_ar_p2_3ch)
        print('Channel :', ch_ar_p2_3ch)
        print('Latency :', lat_ar_p2_3ch)
        print('Amplitude :', amp_ar_p2_3ch)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_lat_att_r_p2_3ch.append(0)
        list_amp_att_r_p2_3ch.append(0)
        list_ch_att_r_p2_3ch.append(0)

print()
print('Unattentive auditory left P2 values')
for una_l_3ch in list_una_l_3ch:
    try:
        ch_ul_p2_3ch, lat_ul_p2_3ch, amp_ul_p2_3ch = list_una_l_3ch[q].get_peak(ch_type="eeg", tmin=tmin_p2, tmax=tmax_p2, mode="pos", return_amplitude=True)
        q += 1
        amp_ul_p2_3ch=amp_ul_p2_3ch*1000000
        list_ch_una_l_p2_3ch.append(ch_ul_p2_3ch)
        list_lat_una_l_p2_3ch.append(lat_ul_p2_3ch)
        list_amp_una_l_p2_3ch.append(amp_ul_p2_3ch)
        print('Channel :', ch_ul_p2_3ch)
        print('Latency :', lat_ul_p2_3ch)
        print('Amplitude :', amp_ul_p2_3ch)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_lat_una_l_p2_3ch.append(0)
        list_amp_una_l_p2_3ch.append(0)
        list_ch_una_l_p2_3ch.append(0)

print()
print('Unattentive auditory right P2 values')
for una_r_3ch in list_una_r_3ch:
    try:
        ch_ur_p2_3ch, lat_ur_p2_3ch, amp_ur_p2_3ch = list_una_r_3ch[r].get_peak(ch_type="eeg", tmin=tmin_p2, tmax=tmax_p2, mode="pos", return_amplitude=True)
        r += 1
        amp_ur_p2_3ch=amp_ur_p2_3ch*1000000
        list_ch_una_r_p2_3ch.append(ch_ur_p2_3ch)
        list_lat_una_r_p2_3ch.append(lat_ur_p2_3ch)
        list_amp_una_r_p2_3ch.append(amp_ur_p2_3ch)
        print('Channel :', ch_ur_p2_3ch)
        print('Latency :', lat_ur_p2_3ch)
        print('Amplitude :', amp_ur_p2_3ch)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_lat_una_r_p2_3ch.append(0)
        list_amp_una_r_p2_3ch.append(0)
        list_ch_una_r_p2_3ch.append(0)

print()
a = 0
for n in range(len(epochs_list)):
    average_amp_p2_subject_3ch = (list_amp_att_l_p2_3ch[n] + list_amp_att_r_p2_3ch[n] + list_amp_una_l_p2_3ch[n] + list_amp_una_r_p2_3ch[n]) / 4
    list_average_amp_p2_3ch.append(average_amp_p2_subject_3ch)
    average_lat_p2_subject_3ch = (list_lat_att_l_p2_3ch[n] + list_lat_att_r_p2_3ch[n] + list_lat_una_l_p2_3ch[n] + list_lat_una_r_p2_3ch[n]) / 4
    list_average_lat_p2_3ch.append(average_lat_p2_subject_3ch)
    print(f"Average amplitude for subject S{subject_number_list[a]} is {average_amp_p2_subject_3ch}")
    print(f"Average latency for subject S{subject_number_list[a]} is {average_lat_p2_subject_3ch}")
    print()
    a += 1
average_amp_att_l_p2_3ch = sum(list_amp_att_l_p2_3ch) / len(list_amp_att_l_p2_3ch)
average_amp_att_r_p2_3ch = sum(list_amp_att_r_p2_3ch) / len(list_amp_att_r_p2_3ch)
average_amp_una_l_p2_3ch = sum(list_amp_una_l_p2_3ch) / len(list_amp_una_l_p2_3ch)
average_amp_una_r_p2_3ch = sum(list_amp_una_r_p2_3ch) / len(list_amp_una_r_p2_3ch)
average_amp_p2_3ch = (average_amp_att_l_p2_3ch + average_amp_att_r_p2_3ch + average_amp_una_l_p2_3ch + average_amp_una_r_p2_3ch) / 4
print("Total average amplitude is", average_amp_p2_3ch)
average_lat_att_l_p2_3ch = sum(list_lat_att_l_p2_3ch) / len(list_lat_att_l_p2_3ch)
average_lat_att_r_p2_3ch = sum(list_lat_att_r_p2_3ch) / len(list_lat_att_r_p2_3ch)
average_lat_una_l_p2_3ch = sum(list_lat_una_l_p2_3ch) / len(list_lat_una_l_p2_3ch)
average_lat_una_r_p2_3ch = sum(list_lat_una_r_p2_3ch) / len(list_lat_una_r_p2_3ch)
average_lat_p2_3ch = (average_lat_att_l_p2_3ch + average_lat_att_r_p2_3ch + average_lat_una_l_p2_3ch + average_lat_una_r_p2_3ch) / 4
print("Total average latency is", average_lat_p2_3ch)

# %%
#P3 calculations with the new channels
tmin_p3 = 0.300
tmax_p3 = 0.440
list_ch_att_l_p3_3ch=[]
list_ch_att_r_p3_3ch=[]
list_ch_una_l_p3_3ch=[]
list_ch_una_r_p3_3ch=[]
list_lat_att_l_p3_3ch = []
list_amp_att_l_p3_3ch = []
list_lat_att_r_p3_3ch = []
list_amp_att_r_p3_3ch = []
list_lat_una_l_p3_3ch = []
list_amp_una_l_p3_3ch = []
list_lat_una_r_p3_3ch = []
list_amp_una_r_p3_3ch = []
list_average_amp_p3_3ch = []
list_average_lat_p3_3ch = []

o = 0
p = 0
q = 0
r = 0

print('Attentive auditory left P3 values')
for att_l_3ch in list_att_l_3ch:
    try:
        ch_al_p3_3ch, lat_al_p3_3ch, amp_al_p3_3ch = list_att_l_3ch[o].get_peak(ch_type="eeg", tmin=tmin_p3, tmax=tmax_p3, mode="pos", return_amplitude=True)
        o += 1
        amp_al_p3_3ch=amp_al_p3_3ch*1000000
        list_ch_att_l_p3_3ch.append(ch_al_p3_3ch)
        list_lat_att_l_p3_3ch.append(lat_al_p3_3ch)
        list_amp_att_l_p3_3ch.append(amp_al_p3_3ch)
        print('Channel:', ch_al_p3_3ch)
        print('Latency:', lat_al_p3_3ch)
        print('Amplitude:', amp_al_p3_3ch)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_lat_att_l_p3_3ch.append(0)
        list_amp_att_l_p3_3ch.append(0)
        list_ch_att_l_p3_3ch.append(0)

print()
print('Attentive auditory right P3 values')
for att_r_3ch in list_att_r_3ch:
    try:
        ch_ar_p3_3ch, lat_ar_p3_3ch, amp_ar_p3_3ch = list_att_r_3ch[p].get_peak(ch_type="eeg", tmin=tmin_p3, tmax=tmax_p3, mode="pos", return_amplitude=True)
        p += 1
        amp_ar_p3_3ch=amp_ar_p3_3ch*1000000
        list_ch_att_r_p3_3ch.append(ch_ar_p3_3ch)
        list_lat_att_r_p3_3ch.append(lat_ar_p3_3ch)
        list_amp_att_r_p3_3ch.append(amp_ar_p3_3ch)
        print('Channel:', ch_ar_p3_3ch)
        print('Latency:', lat_ar_p3_3ch)
        print('Amplitude:', amp_ar_p3_3ch)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_lat_att_r_p3_3ch.append(0)
        list_amp_att_r_p3_3ch.append(0)
        list_ch_att_r_p3_3ch.append(0)

print()
print('Unattentive auditory left P3 values')
for una_l_3ch in list_una_l_3ch:
    try:
        ch_ul_p3_3ch, lat_ul_p3_3ch, amp_ul_p3_3ch = list_una_l_3ch[q].get_peak(ch_type="eeg", tmin=tmin_p3, tmax=tmax_p3, mode="pos", return_amplitude=True)
        q += 1
        amp_ul_p3_3ch=amp_ul_p3_3ch*1000000
        list_ch_una_l_p3_3ch.append(ch_ul_p3_3ch)
        list_lat_una_l_p3_3ch.append(lat_ul_p3_3ch)
        list_amp_una_l_p3_3ch.append(amp_ul_p3_3ch)
        print('Channel:', ch_ul_p3_3ch)
        print('Latency:', lat_ul_p3_3ch)
        print('Amplitude:', amp_ul_p3_3ch)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_lat_una_l_p3_3ch.append(0)
        list_amp_una_l_p3_3ch.append(0)
        list_ch_una_l_p3_3ch.append(0)

print()
print('Unattentive auditory right P3 values')
for una_r_3ch in list_una_r_3ch:
    try:
        ch_ur_p3_3ch, lat_ur_p3_3ch, amp_ur_p3_3ch = list_una_r_3ch[r].get_peak(ch_type="eeg", tmin=tmin_p3, tmax=tmax_p3, mode="pos", return_amplitude=True)
        r += 1
        amp_ur_p3_3ch=amp_ur_p3_3ch*1000000
        list_ch_una_r_p3_3ch.append(ch_ur_p3_3ch)
        list_lat_una_r_p3_3ch.append(lat_ur_p3_3ch)
        list_amp_una_r_p3_3ch.append(amp_ur_p3_3ch)
        print('Channel:', ch_ur_p3_3ch)
        print('Latency:', lat_ur_p3_3ch)
        print('Amplitude:', amp_ur_p3_3ch)
    except ValueError:
        print('No positive peak found within the specified time window.')
        list_lat_una_r_p3_3ch.append(0)
        list_amp_una_r_p3_3ch.append(0)
        list_ch_una_r_p3_3ch.append(0)
print()
a = 0
for n in range(len(epochs_list)):
    average_amp_p3_subject_3ch = (list_amp_att_l_p3_3ch[n] + list_amp_att_r_p3_3ch[n] + list_amp_una_l_p3_3ch[n] + list_amp_una_r_p3_3ch[n]) / 4
    list_average_amp_p3_3ch.append(average_amp_p3_subject_3ch)
    average_lat_p3_subject_3ch = (list_lat_att_l_p3_3ch[n] + list_lat_att_r_p3_3ch[n] + list_lat_una_l_p3_3ch[n] + list_lat_una_r_p3_3ch[n]) / 4
    list_average_lat_p3_3ch.append(average_lat_p3_subject_3ch)
    print(f"Average amplitude for subject S{subject_number_list[a]} is {average_amp_p3_subject_3ch}")
    print(f"Average latency for subject S{subject_number_list[a]} is {average_lat_p3_subject_3ch}")
    print()
    a += 1

average_amp_att_l_p3_3ch = sum(list_amp_att_l_p3_3ch) / len(list_amp_att_l_p3_3ch)
average_amp_att_r_p3_3ch = sum(list_amp_att_r_p3_3ch) / len(list_amp_att_r_p3_3ch)
average_amp_una_l_p3_3ch = sum(list_amp_una_l_p3_3ch) / len(list_amp_una_l_p3_3ch)
average_amp_una_r_p3_3ch = sum(list_amp_una_r_p3_3ch) / len(list_amp_una_r_p3_3ch)

average_amp_p3_3ch = (average_amp_att_l_p3_3ch + average_amp_att_r_p3_3ch + average_amp_una_l_p3_3ch + average_amp_una_r_p3_3ch) / 4
print("Total average amplitude is", average_amp_p3_3ch)

average_lat_att_l_p3_3ch = sum(list_lat_att_l_p3_3ch) / len(list_lat_att_l_p3_3ch)
average_lat_att_r_p3_3ch = sum(list_lat_att_r_p3_3ch) / len(list_lat_att_r_p3_3ch)
average_lat_una_l_p3_3ch = sum(list_lat_una_l_p3_3ch) / len(list_lat_una_l_p3_3ch)
average_lat_una_r_p3_3ch = sum(list_lat_una_r_p3_3ch) / len(list_lat_una_r_p3_3ch)

average_lat_p3_3ch = (average_lat_att_l_p3_3ch + average_lat_att_r_p3_3ch + average_lat_una_l_p3_3ch + average_lat_una_r_p3_3ch) / 4


print("Total average latency is", average_lat_p3_3ch)



# %%
#results are saved to each subjects folder
for k, epoch in enumerate(epochs_list):
    subject_number = subject_number_list[k]
    subject_folder_name = f'Subject_{subject_number}_results'
    subject_folder = os.path.join(results_path, subject_folder_name)

    
    if not os.path.exists(subject_folder):
        os.makedirs(subject_folder)

    #each results is saved to a dictionary
    subject_data = {
        'lat_att_l_n1': list_lat_att_l_n1[k],
        'amp_att_l_n1': list_amp_att_l_n1[k],
        'ch_att_l_n1': list_ch_att_l_n1[k],
        'lat_att_r_n1': list_lat_att_r_n1[k],
        'amp_att_r_n1': list_amp_att_r_n1[k],
        'ch_att_r_n1': list_ch_att_r_n1[k],
        'lat_una_l_n1': list_lat_una_l_n1[k],
        'amp_una_l_n1': list_amp_una_l_n1[k],
        'ch_una_l_n1': list_ch_una_l_n1[k],
        'lat_una_r_n1': list_lat_una_r_n1[k],
        'amp_una_r_n1': list_amp_una_r_n1[k],
        'ch_una_r_n1': list_ch_una_r_n1[k],
        'avg_amp_n1'  : list_average_amp_n1[k],
        'avg_lat_n1'  : list_average_lat_n1[k],
        'avg_amp_n1_total': average_amp_n1,
        'avg_lat_n1_total': average_lat_n1,


        'lat_att_l_p2': list_lat_att_l_p2[k],
        'amp_att_l_p2': list_amp_att_l_p2[k],
        'ch_att_l_p2': list_ch_att_l_p2[k],
        'lat_att_r_p2': list_lat_att_r_p2[k],
        'amp_att_r_p2': list_amp_att_r_p2[k],
        'ch_att_r_p2': list_ch_att_r_p2[k],
        'lat_una_l_p2': list_lat_una_l_p2[k],
        'amp_una_l_p2': list_amp_una_l_p2[k],
        'ch_una_l_p2': list_ch_una_l_p2[k],
        'lat_una_r_p2': list_lat_una_r_p2[k],
        'amp_una_r_p2': list_amp_una_r_p2[k],
        'ch_una_r_p2': list_ch_una_r_p2[k],
        'avg_amp_p2'  : list_average_amp_p2[k],
        'avg_lat_p2'  : list_average_lat_p2[k],
        'avg_amp_p2_total': average_amp_p2,
        'avg_lat_p2_total': average_lat_p2,


        'lat_att_l_p3': list_lat_att_l_p3[k],
        'amp_att_l_p3': list_amp_att_l_p3[k],
        'ch_att_l_p3': list_ch_att_l_p3[k],
        'lat_att_r_p3': list_lat_att_r_p3[k],
        'amp_att_r_p3': list_amp_att_r_p3[k],
        'ch_att_r_p3': list_ch_att_r_p3[k],
        'lat_una_l_p3': list_lat_una_l_p3[k],
        'amp_una_l_p3': list_amp_una_l_p3[k],
        'ch_una_l_p3': list_ch_una_l_p3[k],
        'lat_una_r_p3': list_lat_una_r_p3[k],
        'amp_una_r_p3': list_amp_una_r_p3[k],
        'ch_una_r_p3': list_ch_una_r_p3[k],
        'avg_amp_p3'  : list_average_amp_p3[k],
        'avg_lat_p3'  : list_average_lat_p3[k],
        'avg_amp_p3_total': average_amp_p3,
        'avg_lat_p3_total': average_lat_p3,

        'lat_att_l_n1_3ch': list_lat_att_l_n1_3ch[k],
        'amp_att_l_n1_3ch': list_amp_att_l_n1_3ch[k],
        'ch_att_l_n1_3ch' : list_ch_att_l_n1_3ch[k],
        'lat_att_r_n1_3ch': list_lat_att_r_n1_3ch[k],
        'amp_att_r_n1_3ch': list_amp_att_r_n1_3ch[k],
        'ch_att_r_n1_3ch' : list_ch_att_r_n1_3ch[k],
        'lat_una_l_n1_3ch': list_lat_una_l_n1_3ch[k],
        'amp_una_l_n1_3ch': list_amp_una_l_n1_3ch[k],
        'ch_una_l_n1_3ch' : list_ch_una_l_n1_3ch[k],
        'lat_una_r_n1_3ch': list_lat_una_r_n1_3ch[k],
        'amp_una_r_n1_3ch': list_amp_una_r_n1_3ch[k],
        'ch_una_r_n1_3ch' : list_ch_una_r_n1_3ch[k],
        'avg_amp_n1_3ch'  : list_average_amp_n1_3ch[k],
        'avg_lat_n1_3ch'  : list_average_lat_n1_3ch[k],
        'avg_amp_n1_total_3ch': average_amp_n1_3ch,
        'avg_lat_n1_total_3ch': average_lat_n1_3ch,

        'lat_att_l_p2_3ch': list_lat_att_l_p2_3ch[k],
        'amp_att_l_p2_3ch': list_amp_att_l_p2_3ch[k],
        'ch_att_l_p2_3ch' : list_ch_att_l_p2_3ch[k],
        'lat_att_r_p2_3ch': list_lat_att_r_p2_3ch[k],
        'amp_att_r_p2_3ch': list_amp_att_r_p2_3ch[k],
        'ch_att_r_p2_3ch' : list_ch_att_r_p2_3ch[k],
        'lat_una_l_p2_3ch': list_lat_una_l_p2_3ch[k],
        'amp_una_l_p2_3ch': list_amp_una_l_p2_3ch[k],
        'ch_una_l_p2_3ch' : list_ch_una_l_p2_3ch[k],
        'lat_una_r_p2_3ch': list_lat_una_r_p2_3ch[k],
        'amp_una_r_p2_3ch': list_amp_una_r_p2_3ch[k],
        'ch_una_r_p2_3ch' : list_ch_una_r_p2_3ch[k],
        'avg_amp_p2_3ch'  : list_average_amp_p2_3ch[k],
        'avg_lat_p2_3ch'  : list_average_lat_p2_3ch[k],
        'avg_amp_p2_total_3ch': average_amp_p2_3ch,
        'avg_lat_p2_total_3ch': average_lat_p2_3ch,

        'lat_att_l_p3_3ch': list_lat_att_l_p3_3ch[k],
        'amp_att_l_p3_3ch': list_amp_att_l_p3_3ch[k],
        'ch_att_l_p3_3ch' : list_ch_att_l_p3_3ch[k],
        'lat_att_r_p3_3ch': list_lat_att_r_p3_3ch[k],
        'amp_att_r_p3_3ch': list_amp_att_r_p3_3ch[k],
        'ch_att_r_p3_3ch' : list_ch_att_r_p3_3ch[k],
        'lat_una_l_p3_3ch': list_lat_una_l_p3_3ch[k],
        'amp_una_l_p3_3ch': list_amp_una_l_p3_3ch[k],
        'ch_una_l_p3_3ch' : list_ch_una_l_p3_3ch[k],
        'ch_una_r_p3_3ch' : list_ch_una_r_p3_3ch[k],
        'lat_una_r_p3_3ch': list_lat_una_r_p3_3ch[k],
        'amp_una_r_p3_3ch': list_amp_una_r_p3_3ch[k],
        'avg_amp_p3_3ch'  : list_average_amp_p3_3ch[k],
        'avg_lat_p3_3ch'  : list_average_lat_p3_3ch[k],
        'avg_amp_p3_total_3ch': average_amp_p3_3ch,
        'avg_lat_p3_total_3ch': average_lat_p3_3ch,

    }
    #data frame is converted to a excel and saved
    df = pd.DataFrame(subject_data, index=[0])
    excel_file_path = os.path.join(subject_folder, subject_folder_name)
    df.to_excel(excel_file_path, index=False, engine='openpyxl')

    print(f'Results for Subject {subject_number} saved to {excel_file_path}')


# %%



