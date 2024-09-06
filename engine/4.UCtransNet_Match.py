import gc
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # Activate if GPU is needed
import time
import torch
from criterion.dsc import *
import argparse
import config.Config as config
from model.UCTransNet import *
from data.dataloader import data_loaders
from criterion.Dice_loss import WeightedDiceBCE
from other import judgedir
sep = os.sep
fitness_name = config.fitness_name

features_get = []
def hook(module, input, output):
    features_get.append(output.detach().cpu())

if __name__=='__main__':

    datapath = r'/data/preprocess_h5data/' # preprocessed h5 data
    pklpath = r'/data/segmodel.pkl' # preprocessed pkl data
    modelpath = r'/data/modelsave' # trained model location
    resultsave = r'/data/featuresave' # DL feature library save address
    judgedir(resultsave)

    # Addresses
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=bool, default=True, help='use GPU or not')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer for training, Adam / SGD (default)')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay for SGD / Adam')
    args = parser.parse_args()

    # Load parameters
    config_vit = config.get_CTranS_config()

    ucnet = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels,
                       img_size=[config.img_size[0], config.img_size[1]])
    if args.gpu:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        ucnet.to(device)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    # optimizer
    if args.optim == 'SGD':
        optimizer = torch.optim.SGD(ucnet.parameters(), lr=args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(ucnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('Other optimizer is not implemented')

    # datasets
    dataloader_train, _ = data_loaders(datapath, pklpath, batch_size=1, workers=10)

    # Load model
    state_dict = torch.load(modelpath)
    ucnet.load_state_dict(state_dict)
    ucnet.eval()

    input_list = []
    pred_list = []
    true_list = []
    feature_values = np.zeros((len(dataloader_train.batch_sampler), 512, 1, 1))
    flag = 0
    for i, data in enumerate(dataloader_train):
        gc.collect()
        t1 = time.time()
        x, y_true = data
        x, y_true = x.to(device), y_true.to(device)

        with torch.set_grad_enabled(False):

            handle = ucnet.down4.register_forward_hook(hook)

            y_pred = ucnet(x)

            # Extract features
            feature_save = features_get[0]  # Feature in tensor format 1x512xMxN

            G = torch.nn.AdaptiveAvgPool2d((1, 1))
            GAP = G(feature_save)
            GAP_2 = GAP.detach().cpu().numpy()
            if dataloader_train.batch_size > 1:
                for h in range(GAP_2.shape[0]):
                    feature_values[flag, :, :, :] = GAP_2[h]
                    flag += 1
            else:
                feature_values[i, :, :, :] = GAP_2
            features_get = []
            print("{} / {}   {}s".format(i, len(dataloader_train.batch_sampler), (time.time() - t1)))
        handle.remove()

    feature_values_mean = np.average(feature_values,axis=0)
    feature_values_save = feature_values_mean.reshape((1,512))
    # Save feature library
    save_path = os.path.join(resultsave, str(len(dataloader_train)) + '_match_data.h5')
    judgedir(resultsave)
    # Write to.h5 file
    with h5py.File(save_path, mode='w') as f:
        f.create_dataset('f_values', data=feature_values_save)
    print('Match - ok')



