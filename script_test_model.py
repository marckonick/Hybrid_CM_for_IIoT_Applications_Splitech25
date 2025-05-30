import numpy as np
import matplotlib.pyplot as plt
import data_functions as ff
import yaml
import nn_models_functions as modata
import torch
import torch.utils.data as data
import time  
import seaborn as sns
from sklearn.metrics import confusion_matrix
import copy 



def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def viterbi_cgt(network_outputs, init_prob, state_trans_prob):
    num_states, num_time_points = network_outputs.shape

    # Initialize the dynamic programming tables
    log_probs = np.zeros((num_states, num_time_points))  # log probabilities
    backpointers = np.zeros((num_states, num_time_points), dtype=int)

    # Initialization step (log to avoid underflow)
    log_probs[:, 0] = np.log(init_prob) + np.log(network_outputs[:, 0])

    # Recursion step
    for t in range(1, num_time_points):
        for s in range(num_states):
            transition_probs = log_probs[:, t-1] + np.log(state_trans_prob[:, s])
            best_prev_state = np.argmax(transition_probs)
            log_probs[s, t] = transition_probs[best_prev_state] + np.log(network_outputs[s, t])
            backpointers[s, t] = best_prev_state

    # Termination step
    best_last_state = np.argmax(log_probs[:, -1])
    best_path = [best_last_state]

    # Backtrace the best path
    for t in range(num_time_points - 1, 0, -1):
        best_last_state = backpointers[best_last_state, t]
        best_path.insert(0, best_last_state)

    return best_path

def get_viterbi_results(network_outputs, y_all, plot_on = False):
    
    # network_outputs, 
    init_prob = np.array([1-1e-5, 1e-5/2, 1e-5/2])

    change_prob = 1e-12
    state_trans_prob = np.zeros((3,3))
    state_trans_prob[0,:] = np.array([1-2*change_prob, change_prob, change_prob])
    state_trans_prob[1,:] = np.array([change_prob, 1-2*change_prob, change_prob])
    state_trans_prob[2,:] = np.array([change_prob, change_prob, 1-2*change_prob])
    
    
    best_path_cgp = viterbi_cgt(network_outputs, init_prob, state_trans_prob) 
    
    acc = sum(best_path_cgp == y_all)/len(best_path_cgp)
    print(f"Accuracy for the Viterbi classification: {100*acc}")
     
    cm_viterbi = confusion_matrix(y_all, best_path_cgp)
     
    cm_norm_viterbi = copy.copy(cm_viterbi)
    cm_norm_viterbi = np.array(cm_norm_viterbi, dtype=float)
     
    for i in range(0, n_classes):
       cm_norm_viterbi[i,:] = cm_norm_viterbi[i,:]/np.sum(cm_viterbi, axis=1)[i]


    print('\nConfusion matrix after Viterbi classification: ' )
    print(np.array_str(cm_norm_viterbi, precision=4, suppress_small=True))
    print('\n')



    if plot_on:
      linwdth = 4 
      f_size_title = 32
      f_size_axes = 30
      # Create a figure with two subplots side by side
      plt.figure(figsize=(16, 6))  # Wider figure to accommodate two plots

      # First subplot
      plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
      plt.plot(y_all, linewidth=linwdth+4, color =  'blue')# red
      plt.plot(preds_all, linewidth=linwdth, color = 'red')
      #plt.legend(["true", "instant predictions"], fontsize=f_size_legend)
      plt.xlabel("Samples", fontsize=f_size_axes)  # X-axis label
      #plt.ylabel("Class", fontsize=f_size_axes)       # Y-axis label
      plt.yticks([0, 1, 2], ["good", "broken", "heavy"], fontsize=f_size_axes)  
      plt.xticks(fontsize=f_size_axes)               # Custom x-tick size
      plt.title("Instant Classification - atmo_high Dataset", fontsize=f_size_title, y=1.02)

      # Second subplot
      plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
      plt.plot(y_all, linewidth=linwdth+3, color = 'blue')
      plt.plot(best_path_cgp, linewidth=linwdth, color = 'red')
      #plt.legend(["true", "predicted viterbi"], fontsize=f_size_legend)
      plt.xlabel("Samples", fontsize=f_size_axes)  # X-axis label
      #plt.ylabel("Class", fontsize=f_size_axes)       # Y-axis label
      plt.yticks([0, 1, 2], ["good", "broken", "heavy"], fontsize=f_size_axes) 
      plt.xticks(fontsize=f_size_axes)               # Custom x-tick size
      plt.title("Viterbi Classification - atmo_high Dataset", fontsize=f_size_title, y=1.02)
    
      # Adjust layout and display
      plt.tight_layout()
      #plt.close()
      
    return cm_viterbi, cm_norm_viterbi  
      
      
def do_cm_sns_2x(cm1, cm2, title1="Test dataset 1", title2="Test dataset 2", font_size_1=28, font_size_title=32):

    
    #font_size_1 = 28
    #font_size_title = 32
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,14))  # Wider figure to accommodate two plots
    
    # Plot first confusion matrix
    sv1 = sns.heatmap(cm1, annot=True, cmap='Blues', cbar=False, fmt='.1f', 
                     xticklabels=Y_names_list, yticklabels=Y_names_list, 
                     annot_kws={"size": font_size_1}, ax=ax1)
    sv1.set_xticklabels(sv1.get_xmajorticklabels(), fontsize=font_size_1)
    sv1.set_xlabel("Predicted labels\n", fontsize=font_size_1)
    sv1.set_yticklabels(sv1.get_ymajorticklabels(), fontsize=font_size_1)
    sv1.set_ylabel("True labels", fontsize=font_size_1)
    sv1.set_title(title1, fontsize=font_size_title, y=1.02)
    
    # Plot second confusion matrix
    sv2 = sns.heatmap(cm2, annot=True, cmap='Blues', cbar=False, fmt='.1f', 
                     xticklabels=Y_names_list, yticklabels=Y_names_list, 
                     annot_kws={"size": font_size_1}, ax=ax2)
    sv2.set_xticklabels(sv2.get_xmajorticklabels(), fontsize=font_size_1)
    sv2.set_xlabel("Predicted labels", fontsize=font_size_1)
    sv2.set_yticklabels(sv2.get_ymajorticklabels(), fontsize=font_size_1)
    sv2.set_ylabel("True labels", fontsize=font_size_1)
    sv2.set_title(title2, fontsize=font_size_title, y=1.02)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    

test_classes = ["whitenoise_low", "talking_1", "talking_2", 
"talking_3", "talking_4",  "atmo_low", "atmo_medium", "atmo_high",  "stresstest"]
 

#test_classes = ["stresstest"]


config = load_config("config_test_model.yaml")  
df_config = config['data_and_features']
mtrain_config = config['model_optimization']

save_folder = df_config['save_folder']
to_save_res = df_config['to_save_res']


good_folder = df_config['good_folder']
broken_folder = df_config['broken_folder']
heavy_folder = df_config['heavy_folder']
model_save_folder = df_config['model_save_folder']

sel_class = test_classes[0]
 
Y_all_names = {0:"good", 1:"broken", 2:"heavy"}


kwargs_arguments = {'win_len':df_config["win_len_stft"], 'overlap_l':df_config["overlap_l_stft"], 
                     'time_frame_x':df_config["time_frame_x"], 'time_frame_y':df_config["time_frame_y"], 
                      "win_len_mel":df_config["win_len_mel"], "hop_len_mel":df_config["hop_len_mel"],
                      "nfft_mle":df_config["nfft_mle"], "hop_len_mle":df_config["hop_len_mle"], 
                      "mle_n_frames":df_config["mle_n_frames"], "apply_xai":df_config["apply_xai"], 
                      "num_hidd_dim":df_config["num_hidd_dim"]}
    
    
all_accs = {}
for sel_class in test_classes:
    
    X_all, Y_all = ff.load_data_test(good_folder, broken_folder,heavy_folder, sel_class)

    chosen_features = df_config["selected_feature"] #'FFT' # STFT, TimeFrames
    Fs = df_config['Fs']
    
    X_all, kwargs_outputs = ff.main_extract_features(X_all, chosen_features, Fs, **kwargs_arguments)
    if chosen_features == "FFT":
        Y_all = np.repeat(Y_all, 13)
    
    n_classes = len(np.unique(Y_all))
    if chosen_features == "STFT" or chosen_features == "MelLog" or chosen_features == "MelEner": 
        X_all = X_all[:,None,:,:]
        in_channels = X_all.shape[1]


######################## MODEL ########################

    device = mtrain_config['device']

    model_name = model_save_folder + "model_" + chosen_features + "_"  
    #model_name = model_save_folder + "model_STFT_CNN.pt"
    if chosen_features == "FFT": 
        if kwargs_arguments["apply_xai"]:
            model_name += "DNN_XAI.pt"
            model = modata.DNN_XAI(X_all.shape[1], n_classes=n_classes, n_per_layer=[10])
        else:   
            model_name += "DNN.pt"
            model = modata.DNN(X_all.shape[1], n_classes=n_classes, n_per_layer=[64,32])
    elif chosen_features == "STFT":  
        model_name += "CNN.pt"
        model = modata.VGG_CNN(n_classes, in_channels=in_channels, n_chans1=[16,16,16,16], k_size = [3,3,3,3], padding_t='same', fc_size = 2208) # 384
    elif chosen_features == "MelLog":  
        model_name += "CNN.pt"
        model = modata.VGG_CNN(n_classes, in_channels=in_channels, n_chans1=[16,16,16,16], k_size = [3,3,3,3], padding_t='same', fc_size = 64) # 384
    elif chosen_features == "MelEner":  
        model_name += "CNN.pt"
        model = modata.VGG_CNN(n_classes, in_channels=in_channels, n_chans1=[16,16,16,16], k_size = [3,3,3,3], padding_t='same', fc_size = 2560) # 384
    
    
    model.to(device)
    
    #model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=True ))
    model.load_state_dict(torch.load("saved_models/xai_dnn_200_model_paper.pt", map_location=torch.device(device), weights_only=True ))

    batch_size = mtrain_config['batch_size']

    X_all = modata.labeled_dataset(X_all, Y_all)
    X_all = data.DataLoader(X_all, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    Y_names_list = [yy for yy in Y_all_names.values()]
    
    print(f"\nResults for the dataset: {sel_class}")
    #cm, cm_norm, acc = modata.test_model(X_all, model, n_classes, device)
    cm, cm_norm, outs, preds_all, y_all, acc = modata.test_model_get_all_outputs(X_all, model, n_classes, device)
    network_outputs = outs.transpose()
    cm_viterbi, cm_norm_viterbi = get_viterbi_results(network_outputs, y_all, plot_on = False)
    
    
    #do_cm_sns_2x(cm_norm*100, cm_norm_viterbi*100, "Instant Classification - atmo_high Dataset", "Viterbi Classification - atmo_high Dataset", font_size_1=24, font_size_title=26)


    all_accs[sel_class] = acc

    ts = str(int(time.time()))


print(all_accs)    







