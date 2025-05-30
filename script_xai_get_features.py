import numpy as np
import data_functions as ff
import nn_models_functions as modata
import time 
import torch
from sklearn.model_selection import train_test_split
import torch.utils.data as data
import yaml 
import matplotlib.pyplot as plt
from captum.attr import DeepLift, DeepLiftShap, GradientShap

"""""""""""""""
Works only for FFT features, needs to be modifed for CNN models 

"""""""""""""""

def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)
            

def get_attributions(xai_method, X_train, target):
    
    if xai_method =="deeplift":
        dl = DeepLift(model)   
    elif xai_method == "deepliftshap":
         dl = DeepLiftShap(model)   #
    elif xai_method == "gradientshap":
         dl = GradientShap(model)
    else:
        return 0
      
    model.eval()
    
    
    y_al = []
    tensor_attributions_all = []
    outs_all = []

    with torch.no_grad():      
     for x, y in X_train:

        x = x.to(device=device)
        y = y.to(device=device)
        y = y.long()

        outputs = model(x.float()) 
        
        
        tensor_attributions = dl.attribute(x.float(), target=target, baselines = torch.zeros(32, 5000))       
        
        y_al.append(y.detach().cpu().numpy())
        tensor_attributions_all.append(tensor_attributions.detach().cpu().numpy())   
        outs_all.append(outputs.detach().cpu().numpy()) 
        
    tensor_attributions_all = np.concatenate(tensor_attributions_all)
    y_al = np.concatenate(y_al)
    outs_all = np.concatenate(outs_all)
    
    
    return tensor_attributions_all, y_al, outs_all     


config = load_config("config_xai_model.yaml")  
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

############ Feature Extraction ###########################################

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


if chosen_features == "STFT" or chosen_features == "MelLog" or chosen_features == "MelEner": 
        X_all = X_all[:,None,:,:]
        in_channels = X_all.shape[1]
        
        
        
################## MODEL AND TRAINING ###################
device = mtrain_config['device']
    
    
model_name = model_save_folder + "model_" + chosen_features + "_" 
if chosen_features == "FFT": 
        model_name += "DNN.pt"

        model = modata.DNN(X_all.shape[1], n_classes=n_classes, n_per_layer=[64,32])
else:   
        model_name += "CNN.pt" # stft 576, mellog  256, melener 640
        model = modata.VGG_1D(n_classes, in_channels=in_channels, n_chans1=[16,16,16,16], k_size = [3,3,3,3], padding_t='same', fc_size = 128) # 384
   

   
model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=True ))        
      

X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=test_size, random_state=42)

X_train = modata.labeled_dataset(X_train, Y_train)
X_test = modata.labeled_dataset(X_test, Y_test)

X_train = data.DataLoader(X_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
X_test = data.DataLoader(X_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)        
     


tensor_attributions_all = []
important_indexes = []

# i -> target/label
for i in range(3):  
    tensor_attributions_temp, y_al, outs_all = get_attributions("gradientshap", X_train, i) #deeplift, deepliftshap, gradientshap
    tensor_attributions_all.append(np.mean(tensor_attributions_temp, axis=0))
    important_indexes.append(np.argsort(abs(tensor_attributions_all[-1]))) # from min to max
    
    
plt.figure()
plt.plot(abs(tensor_attributions_all[0]))
plt.plot(abs(tensor_attributions_all[1]))
plt.plot(abs(tensor_attributions_all[2]))
plt.title("Most important features for each class ")


# final 
tensor_attributions_final = np.array(tensor_attributions_all).mean(0)       
        

final_indexes = np.argsort(abs(tensor_attributions_final))

tensor_attributions_final_reducted = np.zeros_like(tensor_attributions_final)
tensor_attributions_final_reducted[final_indexes[-100:]] = tensor_attributions_final[final_indexes[-100:]]

plt.figure()
plt.plot(abs(tensor_attributions_final_reducted))
plt.title("Most important features in total")


np.save("final_xai_indexes", final_indexes)        
        
        
        
        
        