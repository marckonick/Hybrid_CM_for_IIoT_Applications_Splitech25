# Hybrid Condition Monitoring for IIoT Applications
The official repository for the paper "Hybrid Condition Monitoring for IIoT Applications" 

Paper link: https://ieeexplore.ieee.org/document/11091724
Dataset link: https://zenodo.org/records/7551261

## Description 
This paper proposes a computationally efficient method for reliable condition monitoring in the Industrial Internet of Things settings. The approach taken consists of two stages. Firstly, a neural network-based classifier is trained for fault detection and isolation task. In the following stage, an explainable artificial intelligence technique is used for identifying input features with high importance for decision-making. Then, a new classifier is trained using only a selected subset of the original features. Additionally, it is assumed that the process can be structured as a hidden Markov model. In this way, sequential probabilistic inference can be applied via the Viterbi algorithm, for improving decision reliability. The performance was evaluated on an open dataset for acoustic monitoring of electrical motors. Using only noise-free sound recordings, a hybrid classifier was trained using 4\% of the original features. The obtained results have shown that the developed model is very robust as it achieves a near-perfect classification in test datasets with different levels of noise. The entire monitoring procedure was also implemented on an ESP32 microcontroller, without any performance decrease.

## Files Description

- main_classification.py            - training model. Can be used for different features. Feature changes requres change in model configuration. 
- script_test_model.py              - script for testing saved models
- script_xai_get_features.py        - script for getting most important features 
- script_transform_model_format     - script for converting torc model to.tflite format

- data_functions.py                 - data processging functions
- nn_models_functions.py            - neural network classes

- .yaml are configuration files
