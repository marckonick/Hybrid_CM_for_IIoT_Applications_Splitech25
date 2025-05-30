import numpy as np
import data_functions as ff
import nn_models_functions as modata
import time 
import torch
import torch.optim as optim
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import yaml 
import matplotlib.pyplot as plt
import captum

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)

def main():   
    
    
    config = load_config("config_train_model.yaml")  
    df_config = config['data_and_features']
    mtrain_config = config['model_optimization']

    save_folder = df_config['save_folder']
    to_save_res = df_config['to_save_res']
    to_save_model = df_config['to_save_model']
    
    t_start = time.time()
    
    
    good_folder = df_config['good_folder']
    broken_folder = df_config['broken_folder']
    heavy_folder = df_config['heavy_folder']
    model_save_folder = df_config['model_save_folder']


    X_all, Y_all = ff.load_data(good_folder, broken_folder,heavy_folder)
    Y_all_names = {0:"good", 1:"broken", 2:"heavy"}
    n_classes = len(np.unique(Y_all))


    kwargs_arguments = {'win_len':df_config["win_len_stft"], 'overlap_l':df_config["overlap_l_stft"], 
                     'time_frame_x':df_config["time_frame_x"], 'time_frame_y':df_config["time_frame_y"], 
                      "win_len_mel":df_config["win_len_mel"], "hop_len_mel":df_config["hop_len_mel"],
                      "nfft_mle":df_config["nfft_mle"], "hop_len_mle":df_config["hop_len_mle"], 
                      "mle_n_frames":df_config["mle_n_frames"], "apply_xai":df_config["apply_xai"], 
                      "num_hidd_dim":df_config["num_hidd_dim"]}

     
    chosen_features = df_config["selected_feature"] #'FFT' # STFT, TimeFrames
    Fs = df_config['Fs']
  
    
    X_all, kwargs_outputs = ff.main_extract_features(X_all, chosen_features, Fs, **kwargs_arguments)
    if chosen_features == 'FFT':
       Y_all = np.repeat(Y_all, 13)

    
    t_end = time.time()

    print(f"Elapsed time for feature extracting {chosen_features} features {t_end-t_start} seconds")



    ############ Model and optimization ###########################################
    test_size = mtrain_config['test_size']
    batch_size = mtrain_config['batch_size']


    #if chosen_features == "FFT": 
    #    X_all = np.reshape(X_all, (X_all.shape[0], X_all.shape[1]*X_all.shape[2]))
    if chosen_features == "STFT" or chosen_features == "MelLog" or chosen_features == "MelEner": 
        X_all = X_all[:,None,:,:]
        in_channels = X_all.shape[1]
     
     
    X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=test_size, random_state=42)


    X_train = modata.labeled_dataset(X_train, Y_train)
    X_test = modata.labeled_dataset(X_test, Y_test)


    X_train = data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    X_test = data.DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    ################## MODEL AND TRAINING ###################
    device = mtrain_config['device']
    

    # ***
    model_name = model_save_folder + "model_" + chosen_features + "_" 
    if chosen_features == "FFT": 
        model_name += "DNN"
        if kwargs_arguments["apply_xai"]:
            model = modata.DNN_XAI(X_all.shape[1], n_classes=n_classes, n_per_layer=[10])
        else:    
            model = modata.DNN(X_all.shape[1], n_classes=n_classes, n_per_layer=[64,32])

        
    elif chosen_features == "STFT" or chosen_features == "MelLog" or chosen_features == "MelEner":  
        model_name += "CNN" # stft 576, mellog  256, melener 640
        model = modata.VGG_CNN(n_classes, in_channels=in_channels, n_chans1=[16,16,16,16], k_size = [3,3,3,3], padding_t='same', fc_size = 2208) # 384

    
    model = model.to(device)
# torch.save(model.state_dict(), 'stft_model.pt')
    optimizer = optim.Adam(model.parameters(), lr=mtrain_config['lrate'])
    loss_fn = torch.nn.CrossEntropyLoss()
    n_epochs = mtrain_config['n_epochs']
    model.number_of_params()  # prints number of params

    model.train()
    for epoch in range(1,n_epochs+1):
        loss_train = 0.0
        for x, y in X_train:
               x = x.to(device=device)
               y = y.to(device=device)

               outputs = model(x.float())
               loss = loss_fn(outputs, y.long())

               optimizer.zero_grad()
               loss.backward()
               optimizer.step()
               loss_train  += loss.item()
        print(" Epoch ", epoch, "/",n_epochs, " loss = ", float(loss_train/len(X_train)))



########### Testing #################
    Y_names_list = [yy for yy in Y_all_names.values()]
    #np.save('Y_names_list.npy', Y_names_list)
    cm, cm_norm, _ = modata.test_model(X_test, model, n_classes, device)
    
    plt.figure(figsize=(10, 8))
    sv = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt = 'd', xticklabels=Y_names_list, yticklabels=Y_names_list)
    figure = sv.get_figure()    

    if to_save_res:
     sv = sns.heatmap(cm, annot=True, cmap='Blues', cbar=False, fmt = 'd', xticklabels=Y_names_list, yticklabels=Y_names_list)
     sv.set_xlabel("Predicted labels", fontsize = 28)
     sv.set_yticklabels(sv.get_ymajorticklabels(), fontsize = 28)
     sv.set_ylabel("True labels", fontsize = 28)
     sv.set_title("Features - " + str(chosen_features) + "\n Pure data ", fontsize=32)
     
     #figure = sv.get_figure()    
     #figure.savefig(save_folder + 'cm_' + chosen_features + ts + '_'  + '.png', dpi=400) 
     #plt.close(figure)
     
    if to_save_model:
        if kwargs_arguments["apply_xai"]:
           model_name += "_XAI" 
           
        model_name += ".pt"
        torch.save(model.state_dict(), model_name)
 
    
if __name__ == '__main__':
    main()

#cm_stft_speeds_1750_1772_1797
