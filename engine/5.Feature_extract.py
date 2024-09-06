import gc
import h5py
import os
import numpy as np
import time
import copy
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from sklearn.cluster import KMeans
from criterion.dsc import *
import argparse
import config.Config as config
from model.UCTransNet import *
from data.dataloader import data_loaders_all
from other import judgedir
sep = os.sep

features_get = []
def hook(module, input, output):
    features_get.append(output.detach().cpu())

if __name__=='__main__':
    datapath = r'/data/preprocess_h5data/' # preprocessed h5 data
    pklpath = r'/data/segmodel.pkl' # preprocessed pkl data
    modelpath = r'/data/modelsave/' # trained model location
    resultsave = r'/data/DL_feature/' # DL feature save address
    MatchPath = r'/data/featuresave/match_data.h5' # library save address
    judgedir(resultsave)

    # Addresses
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=True, help='use GPU or not')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer for training, Adam / SGD (default)')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay for SGD / Adam')
    args = parser.parse_args()


    config_vit = config.get_CTranS_config()
    ucnet = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=[config.img_size[0],config.img_size[1]])

    if args.gpu:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        ucnet.to(device)

    # Read feature matching library
    with h5py.File(MatchPath, mode='r') as f:
        match_feature = f['f_values'][:]


    # datasets
    dataloader_all = data_loaders_all(datapath, pklpath, batch_size=16, workers=10)
    # Load model
    state_dict = torch.load(modelpath)
    ucnet.load_state_dict(state_dict)
    ucnet.eval()

    input_list = []
    pred_list = []
    true_list = []
    feature_values = np.zeros((len(dataloader_all.dataset.chooseslicepath), 512, 1, 1))
    flag = 0
    print('number: ',len(dataloader_all.batch_sampler))
    for i, data in enumerate(dataloader_all):
        gc.collect()
        t1 = time.time()
        x, y_true = data  # x=data,y_true=label
        x, y_true = x.to(device), y_true.to(device)
        with torch.set_grad_enabled(False):
            handle = ucnet.down4.register_forward_hook(hook)
            y_pred = ucnet(x)
            # Extract features
            feature_save = features_get[0]  # Feature in tensor format 1x512xMxN
            G = torch.nn.AdaptiveAvgPool2d((1, 1))
            GAP = G(feature_save)
            GAP_2 = GAP.detach().cpu().numpy()
            if dataloader_all.batch_size > 1:
                for h in range(GAP_2.shape[0]):
                    feature_values[flag, :, :, :] = GAP_2[h]
                    flag += 1
            else:
                feature_values[i, :, :, :] = GAP_2
            features_get = []
            print(i, '/ ', len(dataloader_all.batch_sampler),' ', time.time() - t1, ' s')
        handle.remove()
    volumes = feature_per_volume(
        feature_values,
        dataloader_all.dataset.chooseslicepath
    )
    # Clustering analysis
    for i, patient in enumerate(volumes):
        print(i, patient,'/n')
        features_values = volumes[patient][:]
        if features_values.shape[0]!= 1:
            features_values_2 = np.squeeze(features_values)  # Remove dimension with value 1
            # Clustering analysis Kmeans f-value cluster
            n_cluster = KMeans(n_clusters=2).fit_predict(copy.deepcopy(features_values_2))  # Return labels corresponding to each data
            max_cluster = np.argmax(np.bincount(n_cluster))  # Count the number of occurrences of 0/1 and return [number of occurrences of 0, number of occurrences of 1]. Finally return the index of the large value.
            # Extract slice features corresponding to the two clusters of clustering Kmeans Max and Min cluster f-values
            Kmeans_max_mask_f_value = features_values_2[n_cluster == max_cluster, :]
            Kmeans_min_mask_f_value = features_values_2[n_cluster!= max_cluster, :]
            # check
            if Kmeans_max_mask_f_value.shape[0] == 0:
                f_min_values = np.average(Kmeans_min_mask_f_value, axis=0)
                f_values = f_min_values.reshape((1, 512))
            elif Kmeans_min_mask_f_value.shape[0] == 0:
                f_max_values = np.average(Kmeans_max_mask_f_value, axis=0)
                f_values = f_max_values.reshape((1, 512))
            else:
                f_values = copy.deepcopy(features_values)
            # Mean operation for feature fusion average cluster f-values(N * 1 * 1024 --> 1 * 1024)
            f_max_values = np.average(Kmeans_max_mask_f_value, axis=0)
            f_max_values = f_max_values.reshape((1, 512))
            f_min_values = np.average(Kmeans_min_mask_f_value, axis=0)
            f_min_values = f_min_values.reshape((1, 512))
            # Perform fingerprint feature matching based on Euclidean distance
            max_d = np.linalg.norm(match_feature - f_max_values, ord=2)
            min_d = np.linalg.norm(match_feature - f_min_values, ord=2)
            if max_d < min_d:
                f_values = f_max_values
            else:
                f_values = f_min_values
        else:
            f_values = copy.deepcopy(features_values)

        # Save features
        patient_save = patient + '.h5'
        resultsave_temp = os.path.join(resultsave, patient_save)
        print(resultsave_temp)
        with h5py.File(os.path.join(resultsave, patient_save), 'w') as f:
            f.create_dataset('f_values', data=f_values)
        print('{}/{} dsfr feature extract {} done!'.format(i + 1, len(volumes), patient))



