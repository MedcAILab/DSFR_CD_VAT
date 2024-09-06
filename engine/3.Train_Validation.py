import os
import torch
from torch.optim import lr_scheduler
import argparse
import config.Config as config
import numpy as np
from tqdm import tqdm
from criterion.dsc import *
from model.UCTransNet import *
from criterion.Dice_loss import WeightedDiceBCE
from data.dataloader import data_loaders
from other import log_loss_summary
from other import writesimple
from other import judgedir
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    datapath = r'/data/preprocess_h5data/' # preprocessed h5 data
    pklpath = r'/data/segmodel.pkl' # preprocessed pkl data
    resultsave = r'/data/modelsave' # model save location
    checkpoint_fold = resultsave + r'checkpoint/'
    checkpoint_save = checkpoint_fold + r'fitness_checkpoint.pth'
    judgedir(checkpoint_fold)

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default=datapath, help='path to dataset (images list file)')
    parser.add_argument('--pkl', default=pklpath, help='path to pkl (divide train and test datasets)')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate for training')
    parser.add_argument('--epochs', default=500, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--optim', type=str, default='Adam', help='optimizer for training, Adam / SGD (default)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight_decay for SGD / Adam')
    parser.add_argument('--gpu', type=bool, default=True, help='use GPU or not')
    args = parser.parse_args()

    datapath = args.root
    pklpath = args.pkl

    excelsavepath = os.path.join(resultsave, 'results.xlsx')
    # Load parameters
    config_vit = config.get_CTranS_config()

    ucnet = UCTransNet(config_vit, n_channels=config.n_channels, n_classes=config.n_labels, img_size=[config.img_size[0],config.img_size[1]])
    if args.gpu:
        device = ('cuda' if torch.cuda.is_available() else 'cpu')
        ucnet.to(device)
    criterion = WeightedDiceBCE(dice_weight=0.5,BCE_weight=0.5)
    best_validation_dsc = 0.0

    # optimizer
    if args.optim == 'SGD':
        params = []
        for key, value in dict(ucnet.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': args.lr * (config_vit['head']['lr_mult'] if 'head' in key else 2.0),
                                'weight_decay': args.weight_decay * (config_vit['head']['decay_mult'] if 'head' in key else 0.0)}]
                else:
                    params += [{'params': [value], 'lr': args.lr * (config_vit['head']['lr_mult'] if 'head' in key else 1.0),
                                'weight_decay': args.weight_decay * (config_vit['head']['decay_mult'] if 'head' in key else 1.0)}]
        optimizer = torch.optim.SGD(ucnet.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(ucnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError('Other optimizer is not implemented')

    # excel item save
    writesimple(excelsavepath, 'epoch', 0, 0, 'train')
    writesimple(excelsavepath, 'loss', 0, 1, 'train')
    writesimple(excelsavepath, 'epoch', 0, 0, 'test')
    writesimple(excelsavepath, 'loss', 0, 1, 'test')
    writesimple(excelsavepath, 'mean_dsc', 0, 2, 'test')
    # datasets
    step = 0
    loss_train = []
    loss_valid = []
    dataloader_train, dataloader_valid = data_loaders(datapath, pklpath, batch_size=7,workers=10) # originally 20
    loaders = {'train': dataloader_train, 'valid': dataloader_valid}

    # Resume from checkpoint
    epoch_start = args.start_epoch
    if os.path.exists(checkpoint_save):
        print("=> loading checkpoint '{}'".format(checkpoint_save))
        checkpoint = torch.load(checkpoint_save)
        ucnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        criterion = checkpoint['loss']

    for epoch in range(epoch_start, args.epochs):
        print("Epoch:{}  Lr:{:.2E}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
        progress_bar_train = tqdm(range(len(dataloader_train.batch_sampler)), total=len(dataloader_train.batch_sampler), unit='batch')
        progress_bar_valid = tqdm(range(len(dataloader_valid.batch_sampler)), total=len(dataloader_valid.batch_sampler), unit='batch')
        progress_bar = {'train': progress_bar_train, 'valid': progress_bar_valid}
        for phase in ["train", "valid"]:
            if phase == "train":
                ucnet.train()
            else:
                ucnet.eval()
            valid_pred = []
            valid_true = []
            # get data
            for i, item in enumerate(loaders[phase]):
                if phase == 'train':
                    step += 1
                data, label = item
                if torch.cuda.is_available():
                    data = data.to(device)
                    label = label.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == "train"):
                        pred = ucnet(data)
                        loss = criterion(pred, label)
                        if phase == 'valid':
                            loss_valid.append(loss.item())
                            y_pred_np = pred.detach().cpu().numpy()
                            valid_pred.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
                            y_true_np = label.detach().cpu().numpy()
                            valid_true.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
                        elif phase == 'train':
                            loss_train.append(loss.item())
                            loss.backward()
                            optimizer.step()

                # Save checkpoint
                if (step % 1000 == 0) and (phase == 'train'):
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': ucnet.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': criterion,
                    }, checkpoint_save)

                progress_bar[phase].update(1) # Update progress bar

            if phase == "train":

                log_loss_summary(loss_train, epoch)
                writesimple(excelsavepath, epoch, epoch+1, 0, 'train')
                writesimple(excelsavepath, np.mean(loss_train), epoch+1, 1, 'train')
                loss_train = []
            if phase == "valid":
                log_loss_summary(loss_valid, epoch, prefix="val_")
                # calculate mean dice
                mean_dsc = np.mean(
                    dsc_per_volume(
                        valid_pred,
                        valid_true,
                        dataloader_valid.dataset.chooseslicepath
                    )
                )
                log_scalar_summary("val_dsc", mean_dsc, epoch)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(ucnet.state_dict(), os.path.join(resultsave, "uctransnet.pt"))
                writesimple(excelsavepath, epoch, epoch+1, 0, 'test')
                writesimple(excelsavepath, np.mean(loss_valid), epoch+1, 1, 'test')
                writesimple(excelsavepath, np.mean(mean_dsc), epoch+1, 2, 'test')
                loss_valid = []
                # Training set results
    print("/nBest validation mean DSC: {:4f}/n".format(best_validation_dsc))

if __name__ == '__main__':
    main()