import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
from scipy import signal 
#import emd
import os
import librosa 
import sys
from scipy import signal


def get_good_data(good_folder, recording_names_good):
    X_good = []
    for i in range(len(recording_names_good)):
        
        X_ex, Fs = librosa.load(good_folder + recording_names_good[i], sr=None, mono=True) # ovooo
        X_good.append(X_ex)
        
    return np.array(X_good)

def get_broken_data(broken_folder, recording_names_broken):
    X_broken = []
    for i in range(len(recording_names_broken)):
        
        X_ex, Fs = librosa.load(broken_folder + recording_names_broken[i], sr=None, mono=True) # ovooo
        X_broken.append(X_ex)
        
    return np.array(X_broken)


def get_heavy_data(heavy_folder, recording_names_heavy):
    X_heavy = []
    for i in range(len(recording_names_heavy)):
        
        X_ex, Fs = librosa.load(heavy_folder + recording_names_heavy[i], sr=None, mono=True) # ovooo
        X_heavy.append(X_ex)
        
    return np.array(X_heavy)


def load_data(good_folder, broken_folder,heavy_folder):
    
    c_lbl = 0
    recording_names_good = os.listdir(good_folder) 
    recording_names_broken = os.listdir(broken_folder) 
    recording_names_heavy = os.listdir(heavy_folder) 

    X_good = get_good_data(good_folder, recording_names_good)
    Y_good = np.zeros(len(X_good))
    
    c_lbl+=1
    X_broken = get_broken_data(broken_folder, recording_names_broken)
    Y_broken = np.zeros(len(X_broken)) + c_lbl

    c_lbl+=1
    X_heavy = get_heavy_data(heavy_folder, recording_names_heavy)
    Y_heavy = np.zeros(len(X_heavy)) + c_lbl


    X = np.concatenate((X_good, X_broken, X_heavy), axis=0)
    Y = np.concatenate((Y_good, Y_broken, Y_heavy), axis=0)


    return X, Y
 
def load_data_test(good_folder, broken_folder, heavy_folder, sel_class):
    
    c_lbl = 0
    recording_names_good = os.listdir(good_folder) 
    recording_names_broken = os.listdir(broken_folder) 
    recording_names_heavy = os.listdir(heavy_folder) 

    recording_names_good = [sublist for sublist in recording_names_good if sel_class in sublist]
    recording_names_broken = [sublist for sublist in recording_names_broken if sel_class in sublist]
    recording_names_heavy = [sublist for sublist in recording_names_heavy if sel_class in sublist]


    X_good = get_good_data(good_folder, recording_names_good)
    Y_good = np.zeros(len(X_good))
    
    c_lbl+=1
    X_broken = get_broken_data(broken_folder, recording_names_broken)
    Y_broken = np.zeros(len(X_broken)) + c_lbl

    c_lbl+=1
    X_heavy = get_heavy_data(heavy_folder, recording_names_heavy)
    Y_heavy = np.zeros(len(X_heavy)) + c_lbl


    X = np.concatenate((X_good, X_broken, X_heavy), axis=0)
    Y = np.concatenate((Y_good, Y_broken, Y_heavy), axis=0)


    return X, Y

def rehsape_2_3d(X, frame_x, frame_y):
    return  np.reshape(X, (X.shape[0], X.shape[1], frame_x, frame_y))

def multiclass_conf_matrix(y_true, y_pred):
    
    n_classes = len(np.unique(y_true))
    conf_matrix = np.zeros((n_classes, n_classes))
    
    # ***
    return conf_matrix

def just_winow_time_signal(X, frame_len, frame_overlap_len, N_fft):
 
    frame_begin = 0
    frame_end = frame_len
    all_frames = []
    while frame_end<len(X):          
            curr_frame = X[frame_begin:frame_end]
            all_frames.append(curr_frame)
            frame_begin += frame_overlap_len
            frame_end = frame_begin + frame_len
            
    return all_frames
            

def fft_analysis(X, Fs):
    
    T = 1/Fs
    N = len(X)
    yf = fft(np.array(X))
    xf = fftfreq(N,T)[:N//2]
  #  plt.plot(xf, 2.0/N*np.abs(yf[0:N//2]))
    
    return xf,yf     



def mel_log_energy(X_ex, Fs,
                         n_mels=128,
                         frames=5,
                         n_fft=1024,
                         hop_length=512,
                         power=2.0):

    
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram using librosa
    mel_spectrogram = librosa.feature.melspectrogram(y=np.array(X_ex, dtype=np.float32),
                                                     sr=Fs,
                                                     n_fft=n_fft,
                                                     hop_length=hop_length,
                                                     n_mels=n_mels,
                                                     power=power)

    # 03 convert melspectrogram to log mel energy
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t: t + vector_array_size].T

    return vector_array

def extract_melener_features(X_all, Fs, **kwargs):
    
    
    nfft_mle = kwargs['nfft_mle']
    hop_len_mle = kwargs['hop_len_mle']
    mle_n_frames = kwargs['mle_n_frames']
    

    N_signals = len(X_all)
    melener_matrix = []
    for i  in range(0, N_signals):
        X = X_all[i]
        
        mle_feat = mel_log_energy(X , Fs, hop_length=hop_len_mle, n_fft=nfft_mle, 
                                  frames=mle_n_frames)

        melener_matrix.append(mle_feat)
        
            
    return np.array(melener_matrix)


def extract_mellog_features(X_all,  Fs, **kwargs):
    
    
    win_len = kwargs['win_len_mel']
    hop_len = kwargs['hop_len_mel']
    
    N_signals = len(X_all)
    mellog_matrix = []
    for i  in range(0, N_signals):
        X = X_all[i]
        
        mel_signal = librosa.feature.melspectrogram(y=np.array(X, dtype=np.float32) , sr=Fs, hop_length=hop_len, n_fft=win_len, norm='slaney')
        spectrogram = np.abs(mel_signal)
        power_to_db = librosa.power_to_db(spectrogram, ref=np.max)

        mellog_matrix.append(power_to_db)
        
            
    return np.array(mellog_matrix)

def extract_stft_features(X_all,  Fs, **kwargs):
    
    
    win_len = kwargs['win_len']
    overlap_l = kwargs['overlap_l']
    
    N_signals = len(X_all)
    stft_matrix = []
    for i  in range(0, N_signals):
        X = X_all[i]
        
        f, t, X_stft = signal.stft(X, Fs, nperseg=win_len, noverlap=overlap_l)
        stft_matrix.append(np.abs(X_stft))
        
            
    return np.array(stft_matrix), f, t

def reshape_array(arr):
    """
    Reshape an array from (N, 132300) to (13*N, 10000) by:
    1. Dividing each row into 13 segments of 10000 elements
    2. Dropping the remaining 2300 elements
    3. Stacking all segments vertically
    
    Parameters:
    arr : numpy.ndarray
        Input array of shape (N, 132300)
        
    Returns:
    numpy.ndarray
        Reshaped array of shape (13*N, 10000)
    """
    # Get the number of rows in the original array
    n_rows = arr.shape[0]
    
    # Reshape each row into (13, 10000) and drop the last 2300 elements
    # Then reshape the entire array to (n_rows*13, 10000)
    reshaped = arr[:, :130000].reshape(n_rows * 13, 10000)
    
    return reshaped



def extract_fft_features(X_all,  Fs, **kwargs):
    
    
    X_all = reshape_array(X_all)
    N_fft = int(X_all.shape[1]) #kwargs['N_fft']
     
    N_signals = len(X_all)
    fft_matrix = []
    #brojac_signala = 0
    #Fs = int(Fs/1)
    for i  in range(0, N_signals):
        X = X_all[i]
        
        #X = signal.decimate(X, 1)
        
        
        xf, Xfft = fft_analysis(X, Fs)   
        fft_matrix.append(2.0/N_fft*np.abs(Xfft[0:N_fft//2]))        
        
    
    return np.array(fft_matrix), xf


def smooth_fft(fft_matrix):
    
    n_groups = fft_matrix.shape[0] // 3
    
    fft_matrix = fft_matrix[:3 * n_groups].reshape(n_groups, 3, -1).mean(axis=1)
    
    return fft_matrix


def extractTimeFeatures(X_all, Y_all, **kwargs):

    
    frame_len = kwargs['frame_len']
    frame_move_len = kwargs['frame_move_len']
    
    N_signals = len(X_all)
    X_time = []
    Y_matrix = []
    frame_begin = 0
    frame_end = frame_len
    N_ft = X_all[0].shape[1]

    for i  in range(0,N_signals):
        X = X_all[i]
        Y = Y_all[i]
        frame_begin = 0
        frame_end = frame_len
        while frame_end < X.shape[1]:
            
            X_time.append(X[:, frame_begin:frame_end])
            frame_begin += frame_move_len
            frame_end = frame_begin + frame_len
            Y_matrix.append(Y)

    
    return np.array(X_time), np.array(Y_matrix)        


def main_extract_features(X_all, chosen_features, Fs, **kwargs):

    
  kwargs_outputs = {}

  if chosen_features == 'FFT':
        X_feat_all, x_f = extract_fft_features(X_all, Fs, **kwargs) 
        if kwargs["apply_xai"]:
            xai_indexes = np.load("final_xai_indexes.npy")
            X_feat_all = X_feat_all[:, xai_indexes[-kwargs["num_hidd_dim"]:]]
        
        #X_feat_all = smooth_fft(X_feat_all)
        kwargs_outputs.update({'f':x_f})
  elif chosen_features == 'STFT':
        X_feat_all, f, t = extract_stft_features(X_all, Fs, **kwargs)
        kwargs_outputs.update({'f':f})
        kwargs_outputs.update({'t':t})
  elif chosen_features == 'MelLog': 
        X_feat_all =  extract_mellog_features(X_all, Fs, **kwargs)  
  elif chosen_features == 'MelEner': 
        X_feat_all = extract_melener_features(X_all, Fs, **kwargs)          


  else:
      print("undefined feature type !!!")
      return -1,-1,-1
      
  return X_feat_all, kwargs_outputs


