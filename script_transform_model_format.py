import numpy as np
import matplotlib.pyplot as plt
import data_functions as ff
import yaml
import nn_models_functions as modata
import torch
import torch.utils.data as data
import time  
import copy
import tensorflow as tf
from tensorflow.keras import layers

def create_keras_model(input_shape=100, n_classes=3, n_per_layer=[10, 32]):
    
    inputs = tf.keras.Input(shape=(input_shape,))
    x = layers.Dense(n_per_layer[0], activation='relu')(inputs)
    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model




def load_config(config_file):
    with open(config_file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


#test_classes = ["atmo_high", "atmo_low", "atmo_medium", "talking_1", "talking_2", 
#"talking_3", "talking_4", "whitenoise_low", "stresstest"]

test_classes = ["talking_1"]

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
    
    

for sel_class in test_classes:
    
    X_all, Y_all = ff.load_data_test(good_folder, broken_folder,heavy_folder, sel_class)

    chosen_features = df_config["selected_feature"] #'FFT' # STFT, TimeFrames
    Fs = df_config['Fs']
    X_all_time = copy.copy(X_all)
    X_all_time = ff.reshape_array(X_all)

    X_all, kwargs_outputs = ff.main_extract_features(X_all, chosen_features, Fs, **kwargs_arguments)
    
    if chosen_features == "FFT":
        Y_all = np.repeat(Y_all, 13)
    
    n_classes = len(np.unique(Y_all))
    if chosen_features == "STFT" or chosen_features == "MelLog" or chosen_features == "MelEner": 
        X_all = X_all[:,None,:,:]
        in_channels = X_all.shape[1]


######################## MODEL ########################

    device = 'cpu' #mtrain_config['device']
    

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
        model = modata.VGG_1D(n_classes, in_channels=in_channels, n_chans1=[16,16,16,16], k_size = [3,3,3,3], padding_t='same', fc_size = 1536) # 384
    elif chosen_features == "MelLog":  
        model_name += "CNN.pt"
        model = modata.VGG_1D(n_classes, in_channels=in_channels, n_chans1=[16,16,16,16], k_size = [3,3,3,3], padding_t='same', fc_size = 128) # 384
    elif chosen_features == "MelEner":  
        model_name += "CNN.pt"
        model = modata.VGG_1D(n_classes, in_channels=in_channels, n_chans1=[16,16,16,16], k_size = [3,3,3,3], padding_t='same', fc_size = 2560) # 384
    
    
    #model.to(device='cpu')
    model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu'), weights_only=True ))
    #model.load_state_dict(torch.load("xai_dnn_200_model_vrh.pt", map_location=torch.device('cpu'), weights_only=True ))
    
    batch_size = mtrain_config['batch_size']
    X_all_feat_np = copy.copy(X_all)
    Y_all_feat_np = copy.copy(Y_all)

    
    X_all = modata.labeled_dataset(X_all, Y_all)
    X_all = data.DataLoader(X_all, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    Y_names_list = [yy for yy in Y_all_names.values()]
    cm, cm_norm, acc = modata.test_model(X_all, model, n_classes, device)
    

    ts = str(int(time.time()))


freqs = kwargs_outputs['f']
xai_indexes = np.load("final_xai_indexes.npy")

selected_freqs = freqs[xai_indexes[-100:]]


ex_idx = 2014
x_ex_time = X_all_time[ex_idx:ex_idx+1]
x_ex = X_all_feat_np[ex_idx:ex_idx+1]
y_ex = Y_all_feat_np[ex_idx:ex_idx+1]

x_ex_feat = X_all_feat_np[ex_idx:ex_idx+1]


# torch prediction
y_ex_pred = torch.softmax(model(torch.tensor(x_ex).float()),1).detach().cpu().numpy()


keras_model = create_keras_model(input_shape=200)
keras_model.summary()

# Get PyTorch weights
pytorch_weights = model.state_dict()

# Transfer weights to Keras model
# First dense layer
keras_model.layers[1].set_weights([
    pytorch_weights['lin1_e.weight'].cpu().numpy().T,  # Transpose weight matrix
    pytorch_weights['lin1_e.bias'].cpu().numpy()
])

# Second dense layer
keras_model.layers[2].set_weights([
    pytorch_weights['lin2_e.weight'].cpu().numpy().T,  # Transpose weight matrix
    pytorch_weights['lin2_e.bias'].cpu().numpy()
])


y_keras_predict = keras_model.predict(X_all_feat_np)

cm, cm_norm, outputs_ll, preds_all, y_all, acc = modata.test_model_get_all_outputs(X_all, model, n_classes)
y_keras_predict_c = np.argmax(y_keras_predict, 1)

equivalence = sum((preds_all == y_keras_predict_c))/len(preds_all)
print(f"Torch Keras model equivalence: {equivalence*100}%")

keras_model.save('dnn_fft_xai_model.h5')


#########################
#########################
#########################

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open("tflite_models/dnn_fft_xai_model.tflite", "wb") as f:
    f.write(tflite_model)

# git bash 
#  xxd -i dnn_fft_xai_model.tflite > dnn_fft_xai_model.cc













