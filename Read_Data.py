



import scipy
import spkit as sp
import matplotlib.pyplot as plt

import os
import glob
import re
import numpy as np
import scipy.io
import scipy.signal
import mne
import warnings
from scipy.ndimage import uniform_filter1d
mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# --------------------------------------------------
# Utility
# --------------------------------------------------
def patient_number(path):
    fname = os.path.basename(path)
    return int(re.findall(r"\d+", fname)[0])

# --------------------------------------------------
# Bipolar construction EXACTLY like Read_Data.py
# --------------------------------------------------
def signal_array_18(raw_data):
    ch1  = raw_data[1]  - raw_data[3]   # FP2-F4
    ch2  = raw_data[3]  - raw_data[5]   # F4-C4
    ch3  = raw_data[5]  - raw_data[7]   # C4-P4
    ch4  = raw_data[7]  - raw_data[9]   # P4-O2
    ch5  = raw_data[0]  - raw_data[2]   # FP1-F3
    ch6  = raw_data[2]  - raw_data[4]   # F3-C3
    ch7  = raw_data[4]  - raw_data[6]   # C3-P3
    ch8  = raw_data[6]  - raw_data[8]   # P3-O1
    ch9  = raw_data[1]  - raw_data[11]  # FP2-F8
    ch10 = raw_data[11] - raw_data[13]  # F8-T4
    ch11 = raw_data[13] - raw_data[15]  # T4-T6
    ch12 = raw_data[15] - raw_data[9]   # T6-O2
    ch13 = raw_data[0]  - raw_data[10]  # FP1-F7
    ch14 = raw_data[10] - raw_data[12]  # F7-T3
    ch15 = raw_data[12] - raw_data[14]  # T3-T5
    ch16 = raw_data[14] - raw_data[8]   # T5-O1
    ch17 = raw_data[16] - raw_data[17]  # FZ-CZ
    ch18 = raw_data[17] - raw_data[18]  # CZ-PZ

    # ORDER IS CRITICAL (matches Read_Data.py)
    return np.array([ch9,ch10,ch11,ch12,ch4,ch3,ch2,ch1,ch17,ch18,ch13,ch14,ch15,ch16,ch8,ch7,ch6,ch5])
def signal_array_12(raw_data):
    ch1=raw_data[0]-raw_data[5] #FP1-T3
    ch2=raw_data[5]-raw_data[7] #T3-O1
    ch3=raw_data[0]-raw_data[2] #FP1-C3
    ch4=raw_data[2]-raw_data[7] #C3-O1
    ch5=raw_data[1]-raw_data[3] #FP2-C4
    ch6=raw_data[3]-raw_data[8] #C4-O2
    ch7=raw_data[1]-raw_data[6] #FP2-T4
    ch8=raw_data[6]-raw_data[8] #T4-O2
    ch9=raw_data[5]-raw_data[2] #T3-C3
    ch10=raw_data[2]-raw_data[4] #C3-Cz
    ch11=raw_data[4]-raw_data[3] #Cz-C4
    ch12=raw_data[3]-raw_data[6] #C4-T4
    
    return np.array([ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12])
# --------------------------------------------------
# Main preprocessing
# --------------------------------------------------

def detect_bad_samples(data, sfreq, 
                       flat_sec=3.0, 
                       flat_std_thresh=0.5e-6,
                       amp_uv_thresh=200e-6, # Ensure units match (Volts)
                       amp_sec=2.0):
    
    n_ch, n_samp = data.shape
    flat_win = int(flat_sec * sfreq)
    amp_win = int(amp_sec * sfreq)
    
    # Preallocate global mask
    bad = np.zeros(n_samp, dtype=bool)
    
    # Kernel for rolling mean (boxcar)
    # We create it once since it's the same for all channels
    flat_kernel = np.ones(flat_win) / flat_win
    
    for ch in range(n_ch):
        # --- 1. Flat detection (Vectorized) ---
        # Formula: Var = E[x^2] - (E[x])^2
        # 'valid' mode prevents edge artifacts but shrinks the array
        
        # Calculate rolling terms
        mean = np.convolve(data[ch], flat_kernel, mode='valid')
        mean_sq = np.convolve(data[ch]**2, flat_kernel, mode='valid')
        
        # Calculate STD (safely)
        # np.maximum(0, ...) protects against negative epsilon errors
        std = np.sqrt(np.maximum(0, mean_sq - mean**2))
        
        # Re-align with original data
        # 'valid' convolution removes (flat_win - 1) samples
        # We pad symmetrically to put the detection in the center of the window
        pad_front = flat_win // 2
        pad_back = n_samp - len(std) - pad_front
        
        # Create full-length mask
        # We extend the 'valid' std array back to original size
        # Areas where convolution wasn't possible (edges) default to False (Good)
        flat_mask = np.pad(std < flat_std_thresh, (pad_front, pad_back), constant_values=False)
        if np.any(flat_mask):
            mask = np.convolve(flat_mask, np.ones(flat_win), mode='same') > 0
            bad = bad | mask
        
        # --- 2. High amplitude detection (Vectorized) ---
        high_mask = np.abs(data[ch]) > amp_uv_thresh
        
        # Check if ALL samples in window are high
        # We calculate the fraction of high samples in the window.
        # If fraction > 0.99, we assume 100% (safer than == 1.0)
        high_fraction = uniform_filter1d(high_mask.astype(float), size=amp_win, mode='constant', cval=0)
        
        all_high = high_fraction > 0.99
        if np.any(all_high):
            mask_expanded = np.convolve(all_high, np.ones(amp_win), mode='same') > 0
            bad = bad | mask_expanded
        
    
    return bad

def preprocess_helsinki_data(edf_path, annotation_path, output_dir, patient_idx):

    print(f"\n=== Processing patient {patient_idx + 1} ===")

    # --------------------------------------------------
    # Load annotations (consensus)
    # --------------------------------------------------
    mat = scipy.io.loadmat(annotation_path)
    raw_annots = mat["annotat_new"][0, patient_idx]
    consensus = (np.sum(raw_annots, axis=0) >= 3).astype(np.int8)

    if consensus.sum() == 0:
        print("No seizures : skipping")
        return False

    # --------------------------------------------------
    # Load EEG
    # --------------------------------------------------
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    # --------------------------------------------------
    # Pick referential channels robustly
    # --------------------------------------------------
    base_chs = [
        "Fp1","Fp2","C3","C4","Cz","T3","T4","O1","O2"
    ]
    # channels_1_12=['EEG Fp1-Ref', 'EEG Fp2-Ref',  'EEG C3-Ref', 'EEG C4-Ref', 'EEG Cz-Ref', 'EEG T3-Ref','EEG T4-Ref', 'EEG O1-Ref', 'EEG O2-Ref']

    ch_map = {}
    for ch in raw.ch_names:
        clean = ch.replace("EEG ", "").replace("-Ref", "").replace("-REF", "")
        ch_map[clean] = ch

    try:
        picks = [ch_map[ch] for ch in base_chs]
    except KeyError as e:
        print(f"Missing channel {e}, skipping patient")
        return False

    raw.pick_channels(picks, ordered=True)
    data = raw.get_data()  # shape (19, N)

    # --------------------------------------------------
    # Bipolar montage (NUMERICAL, robust)
    # --------------------------------------------------
    data = signal_array_12(data)  # (18, N)

    # --------------------------------------------------
    # Filtering & resampling (paper-faithful)
    # --------------------------------------------------
    raw_bip = mne.io.RawArray(
        data,
        mne.create_info(
            ch_names=[f"ch{i}" for i in range(12)],
            sfreq=raw.info["sfreq"],
            ch_types="eeg"
        ),
        verbose=False
    )
    sfreq_d = 32
    raw_bip.filter(
        l_freq=0.5,
        h_freq=sfreq_d/2,
        method="iir",
        iir_params=dict(order=7, ftype="butter"),
        phase="zero",
        verbose=False
    )

    raw_bip.resample(sfreq_d, verbose=False)
    data = raw_bip.get_data()
    sfreq = sfreq_d

    # --------------------------------------------------
    # Align labels (1 Hz → 32 Hz)
    # --------------------------------------------------
    labels = np.repeat(consensus, sfreq)
    labels = labels[:data.shape[1]]

    bad_mask = detect_bad_samples(
    data,
    sfreq=sfreq,
    flat_sec=3.0,
    flat_std_thresh=0.5e-6,
    amp_uv_thresh=200e-6,
    amp_sec=2.0
    )

    # Remove bad samples
    data = data[:, ~bad_mask]
    labels = labels[~bad_mask]

    # --------------------------------------------------
    # Windowing + normalization
    # --------------------------------------------------
    win_len = 12 * sfreq
    hop_seiz = 1 * sfreq
    hop_non = 2 * sfreq

    X, y = [], []
    idx = 0

    while idx + win_len <= data.shape[1]:
        window = data[:, idx:idx + win_len]

        # mu = window.mean(axis=1, keepdims=True)
        # std = window.std(axis=1, keepdims=True)
        # window = (window - mu) / (std + 1e-12)

        label = int(labels[idx:idx + win_len].mean() > 0)

        X.append(window)
        y.append(label)

        idx += hop_seiz if label else hop_non

    X = np.asarray(X)
    y = np.asarray(y)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"patient_{patient_idx + 1}_processed.npz")
    np.savez(out_path, x=X, y=y)

    print(f"Saved {len(X)} windows | {y.sum()} seizure")
    return True



# Read the annotation file and find the seizure time
def find_seizure_time(file_no):
    # Annotations=scipy.io.loadmat('./Datasets/zenodo_eeg/annotations_2017.mat')
    Annotations = scipy.io.loadmat(r"D:\Datasets\zenodo_eeg\annotations_2017.mat")
    Annotations_by_consensus=Annotations['annotat_new'][0][file_no-1][0] & Annotations['annotat_new'][0][file_no-1][1] & Annotations['annotat_new'][0][file_no-1][2]
    a=np.where(Annotations_by_consensus==1)[0]
    start_time=[]
    end_time=[]
    if len(a)!=0:
        start_time.append(a[0])  
        for r in range(1,a.shape[0]):
            if a[r-1]-a[r]!=-1:
                end_time.append(a[r-1]+1)
                start_time.append(a[r]) 
        end_time.append(a[-1]+1)  
        
    return np.array(start_time),np.array(end_time)

# Signal filtering and downsampling
fs=256
fs_d=32
order=7
resampling_factor=fs_d/fs

# l_cut=30
l_cut=16
h_cut=1 

b_low,a_low=scipy.signal.cheby2(order,20,l_cut,'low',fs=fs) 
b_high,a_high=scipy.signal.cheby2(order,20,h_cut,'high',fs=fs) 

def bandpassFilter(signal1):
    lowpasss=scipy.signal.filtfilt(b_low,a_low,signal1,axis=1)
    return scipy.signal.filtfilt(b_high,a_high,lowpasss,axis=1)

def downsample(signal1):
    filter_signal=bandpassFilter(signal1)
    new_n_samples=int(signal1.shape[1]*resampling_factor)
    return scipy.signal.resample(filter_signal,new_n_samples,axis=1)

def check_consecutive_zeros(arr):
    return np.all(arr==0)

def removeArtifacts(signal1):
    beta_val=0.3
    signal1=signal1*3000000
    signal1=bandpassFilter(signal1)
    signal1=scipy.signal.resample_poly(signal1,up=128,down=256,axis=1)
    clean_array= sp.eeg.ATAR(signal1.T, wv='db4', winsize=128, beta=beta_val, thr_method='ipr', OptMode='soft',verbose=0)
    clean_array=clean_array.T/3000000
    signal1=scipy.signal.resample_poly(clean_array,up=32,down=128,axis=1)
    return signal1

# Channel names that need to be picked from the raw data
channels_1_18=['EEG Fp1-Ref', 'EEG Fp2-Ref','EEG F3-Ref', 'EEG F4-Ref', 'EEG C3-Ref', 'EEG C4-Ref','EEG P3-Ref',
            'EEG P4-Ref','EEG O1-Ref', 'EEG O2-Ref','EEG F7-Ref','EEG F8-Ref', 'EEG T3-Ref', 'EEG T4-Ref','EEG T5-Ref', 'EEG T6-Ref','EEG Fz-Ref','EEG Cz-Ref','EEG Pz-Ref' ]
channels_2_18=['EEG Fp1-REF', 'EEG Fp2-REF','EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF','EEG P3-REF',
            'EEG P4-REF','EEG O1-REF', 'EEG O2-REF','EEG F7-REF','EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF','EEG T5-REF', 'EEG T6-REF','EEG Fz-REF','EEG Cz-REF','EEG Pz-REF' ]

channels_1_12=['EEG Fp1-Ref', 'EEG Fp2-Ref',  'EEG C3-Ref', 'EEG C4-Ref', 'EEG Cz-Ref', 'EEG T3-Ref','EEG T4-Ref', 'EEG O1-Ref', 'EEG O2-Ref']
channels_2_12=['EEG Fp1-REF', 'EEG Fp2-REF',  'EEG C3-REF', 'EEG C4-REF', 'EEG Cz-REF', 'EEG T3-REF','EEG T4-REF', 'EEG O1-REF', 'EEG O2-REF']

channels={"121":channels_1_12,"122":channels_2_12,"181":channels_1_18,"182":channels_2_18}

# Read the file and return the signal array with seizure time
def read_a_file(file,n,plot_=False,number_of_channel=12,pre_processing=True):
    data = mne.io.read_raw_edf(file)
    try:
        data=data.pick_channels(channels[f"{number_of_channel}1"],ordered=True)
        raw_data=data.get_data()
    except:
        data=data.pick_channels(channels[f"{number_of_channel}2"],ordered=True)
        raw_data=data.get_data()

    if number_of_channel==12:
        signal=signal_array_12(raw_data)
    else:
        signal=signal_array_18(raw_data)

    s_Time,e_Time=find_seizure_time(n) 

    if pre_processing:
        signal=downsample(signal)

    if plot_:
        channel_names=["Fp1-T3","T3-O1","Fp1-C3","C3-O1","Fp2-C4","C4-O2","Fp2-T4","T4-O2","T3-C3","C3-Cz","Cz-C4","C4-T4"]
        seizures=np.zeros((12,len(signal[0])))
        print(s_Time,e_Time)
        for i in range(12):
            for r in range(len(s_Time)):
                seizures[i,s_Time[r]*32:e_Time[r]*32]=signal[i,s_Time[r]*32:e_Time[r]*32]

        fig,ax=plt.subplots(12,1,figsize=(20,20))
        a=np.arange(0,len(signal[0])/(32*60),1/(32*60))
        for r in range(12):
            ax[r].plot(a,signal[r])
            ax[r].plot(a,seizures[r])
            ax[r].set_title(channel_names[r])
        fig.tight_layout()
        plt.show() 

        return signal,seizures
    else:
        return signal,s_Time,e_Time
    
def make_window_seizure_mask_32hz(l_bound, u_bound, s_Time, e_Time):
    win_len = u_bound - l_bound
    mask = np.zeros(win_len, dtype=np.uint8)
    for st, et in zip(s_Time, e_Time):
        st32 = int(st * 32)
        et32 = int(et * 32)
        left = max(l_bound, st32)
        right = min(u_bound, et32)
        if right > left:
            mask[left - l_bound : right - l_bound] = 1
    return mask

# Read all the files and save the signals and seizure time in the numpy file
def read_data(folder,files,low,high):
    train_signals=[]
    train_seizure=[]
    train_masks = []
    test_signals=[]
    test_seizure=[]
    test_masks = []
    c=1
    for file in files:
        if file.endswith('.edf'):

            signal1,s_Time,e_Time=read_a_file(folder+file,int(file.split('.')[0][3:]),pre_processing=False)

            if (len(s_Time)!=0 ):
                #Downsample and filter
                signal=downsample(signal1)
                # print(signal)
                # print(signal.shape)
                # print(signal.mean(axis=1).reshape(-1,1))
                # print(signal.std(axis=1).reshape(-1,1))
                #Normalize per patient per channel
                # print(signal)
                patient_mean = signal.mean(axis=1).reshape(-1,1)
                patient_std = signal.std(axis=1).reshape(-1,1)
                signal = (signal - patient_mean) / (np.maximmum(patient_std,1e-12))
                # print(signal)
                u_bound=384
                l_bound=0
                started=False

                while u_bound<signal.shape[1]:
                    
                    partition=signal[:,l_bound:u_bound]
                    mask = make_window_seizure_mask_32hz(l_bound, u_bound, s_Time, e_Time)

                    if np.any((s_Time*32>=l_bound) & (s_Time*32<=u_bound-1)):
                        if low<=c<=high:
                            test_signals.append(partition)
                            test_seizure.append(1)
                            test_masks.append(mask)
                        else:
                            train_signals.append(partition)
                            train_seizure.append(1)
                            train_masks.append(mask)
                        
                        started=True
                        u_bound+=32
                        l_bound+=32

                    elif np.any((l_bound<(e_Time*32-1)) & ((e_Time*32-1)<=(u_bound-1))):
                        if low<=c<=high:
                            test_signals.append(partition)
                            test_seizure.append(1)
                            test_masks.append(mask)
                        else:
                            train_signals.append(partition)
                            train_seizure.append(1)
                            train_masks.append(mask)
                        started=False
                        u_bound+=32
                        l_bound+=32

                    elif (started and np.any((e_Time*32-1)>u_bound-1)):
                        if low<=c<=high:
                            test_signals.append(partition)
                            test_seizure.append(1)
                            test_masks.append(mask)
                        else:
                            train_signals.append(partition)
                            train_seizure.append(1)
                            train_masks.append(mask)
                        u_bound+=32
                        l_bound+=32
                    else:
                        if check_consecutive_zeros(signal1[:,l_bound//32*256:u_bound//32*256]):
                            pass
                        else:
                            if low<=c<=high:
                                test_signals.append(partition)
                                test_seizure.append(0)
                                test_masks.append(mask)
                            else:
                                train_signals.append(partition)
                                train_seizure.append(0)
                                train_masks.append(mask)
                        u_bound+=64
                        l_bound+=64
                c+=1

    test_signals=np.array(test_signals)
    test_seizure=np.array(test_seizure)
    test_masks=np.array(test_masks)
    train_signals=np.array(train_signals)
    train_seizure=np.array(train_seizure)
    train_masks=np.array(train_masks)
    return train_signals,train_seizure,test_signals,test_seizure, train_masks, test_masks



# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    RAW_DATA_DIR = "./Datasets/zenodo_eeg"
    # RAW_DATA_DIR = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\BraiNeoCare-main\Datasets\zenodo_eeg"
    ANNOTATION_FILE =  "./Datasets/zenodo_eeg/annotations_2017.mat"
    # ANNOTATION_FILE = r"C:\Users\Thomas\OneDrive - Universiteit Twente\UT_MASTER\Q678-Thesis\Project_InterpretableGNN\BraiNeoCare-main\Datasets\zenodo_eeg\annotations_2017.mat"
    PROCESSED_DIR = "./Datasets/Processed_data"

    edf_files = sorted(glob.glob(os.path.join(RAW_DATA_DIR, "*.edf")), key=patient_number)

    for i, edf in enumerate(edf_files):
        preprocess_helsinki_data(edf, ANNOTATION_FILE, PROCESSED_DIR, i)




# # Load the annotation file. Replace the file path with the path to the annotation file.
# Annotations=scipy.io.loadmat('Datasets/zenodo_eeg/annotations_2017.mat')

# # Defined function to calculate the signal array
# # For 12 channels the signal array is calculated as follows:
# def signal_array_12(raw_data):
#     ch1=raw_data[0]-raw_data[5] #FP1-T3
#     ch2=raw_data[5]-raw_data[7] #T3-O1
#     ch3=raw_data[0]-raw_data[2] #FP1-C3
#     ch4=raw_data[2]-raw_data[7] #C3-O1
#     ch5=raw_data[1]-raw_data[3] #FP2-C4
#     ch6=raw_data[3]-raw_data[8] #C4-O2
#     ch7=raw_data[1]-raw_data[6] #FP2-T4
#     ch8=raw_data[6]-raw_data[8] #T4-O2
#     ch9=raw_data[5]-raw_data[2] #T3-C3
#     ch10=raw_data[2]-raw_data[4] #C3-Cz
#     ch11=raw_data[4]-raw_data[3] #Cz-C4
#     ch12=raw_data[3]-raw_data[6] #C4-T4
    
#     return np.array([ch1,ch2,ch3,ch4,ch5,ch6,ch7,ch8,ch9,ch10,ch11,ch12])

# # For 18 channels the signal array is calculated as follows:
# def signal_array_18(raw_data):
#     ch1=raw_data[1]-raw_data[3] #FP2-F4
#     ch2=raw_data[3]-raw_data[5] #F4-C4
#     ch3=raw_data[5]-raw_data[7] #C4-P4
#     ch4=raw_data[7]-raw_data[9] #P4-O2
#     ch5=raw_data[0]-raw_data[2] #FP1-F3
#     ch6=raw_data[2]-raw_data[4] #F3-C3
#     ch7=raw_data[4]-raw_data[6] #C3-P3
#     ch8=raw_data[6]-raw_data[8] #P3-O1
#     ch9=raw_data[1]-raw_data[11] #FP2-F8
#     ch10=raw_data[11]-raw_data[13] #F8-T4
#     ch11=raw_data[13]-raw_data[15] #T4-T6
#     ch12=raw_data[15]-raw_data[9] #T6-O2
#     ch13=raw_data[0]-raw_data[10] #FP1-F7
#     ch14=raw_data[10]-raw_data[12] #F7-T3
#     ch15=raw_data[12]-raw_data[14] #T3-T5
#     ch16=raw_data[14]-raw_data[8] #T5-O1
#     ch17=raw_data[16]-raw_data[17] #FZ-CZ
#     ch18=raw_data[17]-raw_data[18] #CZ-PZ
    
#     return np.array([ch9,ch10,ch11,ch12,ch4,ch3,ch2,ch1,ch17,ch18,ch13,ch14,ch15,ch16,ch8,ch7,ch6,ch5])


# if __name__=="__main__":

#     folder='Datasets/zenodo_eeg/'
#     files=os.listdir(folder)
#     np.random.seed(42)
#     np.random.shuffle(files)
#     train_signals,train_seizure,test_signals,test_seizure=read_data(folder,files,1,8)
    
#     np.save('Datasets/Processed_data/testdata.npy',test_signals)
#     np.save('Datasets/Processed_data/testlabels.npy',test_seizure)
#     np.save('Datasets/Processed_data/traindata.npy',train_signals)
#     np.save('Datasets/Processed_data/trainlabels.npy',train_seizure)