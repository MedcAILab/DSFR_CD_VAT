# -*- coding: utf-8 -*-

import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
fitness_name = 'vat'
place = 'external'
save_model = True
tensorboard = True
use_cuda = torch.cuda.is_available()
seed = 666

os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True # whether use cosineLR or not
n_channels = 1
n_labels = 1 # 1 for binary segmentation, 2 for multi-class segmentation
epochs = 500
img_size = [256,256]   # determined by the size parameter of resample in preprocess
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50

pretrain = False
task_name = 'FitnessSeg'
learning_rate = 1e-3
batch_size = 1


model_name = 'UCTransNet_384'

train_dataset = '../data/newnas/'
val_dataset = '../data/newnas/'
test_dataset = '../data/newnas/'
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'


##########################################################################
# CTrans configs
##########################################################################
def get_CTranS_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads  = 4
    config.transformer.num_layers = 4
    config.expand_ratio           = 4  # MLP channel dimension expand ratio
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    # config.patch_sizes = [16,8,4,2]
    config.patch_sizes = [8,4,2,1]
    config.base_channel = 64 # base channel of U-Net
    config.n_classes = 1
    return config
