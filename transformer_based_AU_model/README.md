# Transformer-based Facial Action Unit Detection Model
+ The data preprocessing and model finetuning codes are adapted from the official [GitHub repo](https://github.com/rakutentech/FAU_CVPR2021) of the paper " Facial Action Unit Detection with Transformers" by Geethu Miriam Jacob and Bjorn Stenger.
+ The original model implementation uses Tensorflow 1.15, so necessary changes have been made to the original codes to make it runnable under Tensorflow 2.15.0.

# Dataset
+ The model is pre-trained on [**BP4D dataset**] (https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html), which is a 3D video database of spontaneous facial expressions in a diverse group of young adults.
+ We finetuned the pre-trained model on [**The Denver Intensity of Spontaneous Facial Action (DISFA)**] (http://mohammadmahoor.com/disfa/), which consists of 27 videos with real human faces in it.

# Experiment Platform
+ All experiments are done on Google Colab using both CPU and T4 GPU.

# Usage
+ To run the codes please follow the following jupyter notebooks, which contain further detailed explanations and comments about the codes.
## Data preprocessing
- preprocess.ipynb: Extraced frames from original videos and checked the AU label distributions.
## Ground truth Attention Map Generation
- attention_map_generation.ipynb: Generated ground truth attention map for each frame and each AU, concatenated the frame image, attention map, and label for further training.
## Pretrained Model Inference
-pretrained_inference.ipynb: Used the pre-trained model to do inference and evaluation on the DISFA test set. 
## Finetune and Inference
finetune_inference.ipynb: Used the pre-trained model and finetune it on the DISFA training data. We change part of the loss functions for experiments. Then the finetuned model was used to do inference and evaluation on the DISFA test set.
## Model files
- face detection/shape_predictor_68_face_landmarks.dat: The model file used for identifying the 68 landmarks of faces in an image.
- models: store the finetuned model weights
  
