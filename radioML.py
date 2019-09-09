import torch
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import random
import math
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import make_interp_spline, BSpline
import scipy.interpolate as interpolate
from logger import setup_logger
import os
import random
from os import listdir
from os.path import isfile, join
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import scipy.signal
from collections import Counter
import model_p_mod as model
import _pickle as cPickle
import torch.nn.functional as F

Params_dict = {
    'lrs': [0.01, 0.03, 0.02, 0.03, 0.008, 0.01, 0.005, 0.015, 0.05, 0.08],
    'batches': [800, 100, 80, 200],
    'batches1': [1000, 1500, 800, 500, 300],
    'max_epochs': 150,
    'test_batch': 1000,
    'num_classes': 11,
    'num_distributions': 5,
    'num_repeat': 100,
}

def to_polar4D(X):
    M = np.linalg.norm(X, axis=1)
    T = np.arctan2(X[:,1,:],X[:,0,:])
    MT = np.amax(T)
    mT = np.amin(T)
    MM = np.amax(M)
    mM = np.amin(M)
    #T = np.expand_dims(2*((T-mT)/(MT-mT))-1.0+0.00001,axis=1)
    #M = ((M-mM)/(MM-mM))+0.00001
    Y = np.expand_dims(np.expand_dims(np.concatenate((np.expand_dims(T,axis=1),np.expand_dims(M,axis=1)),axis=1),axis=3),axis=2)
    return Y


def data_prep(batch_size):
#     def to_onehot(yy):
#         yy1 = np.zeros([len(yy), max(yy)+1])
#         yy1[np.arange(len(list(yy))),yy] = 1
#         return yy1
    
#     Xd = cPickle.load(open('../data/RML2016.10a_dict.pkl','rb'), encoding='latin1')
#     snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
#     X = []  
#     lbl = []
#     for mod in mods:
#         for snr in snrs:
#             X.append(Xd[(mod,snr)])
#             for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
#     X = np.vstack(X)
    
#     train_idx = np.load("../data/train_idx.npy")
#     test_idx = np.load("../data/test_idx.npy")
    
#     X_train = X[train_idx]
#     X_test = X[test_idx]
#     X_train = (X_train - np.mean(X_train) ) / np.std(X_train)
#     X_test = (X_test - np.mean(X_test) ) / np.std(X_test)

#     X_train = to_polar4D(X_train)
#     X_test = to_polar4D(X_test)
    
#     Y_train = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), train_idx)) )
#     Y_test = np.asarray(list(map(lambda x: mods.index(lbl[x][0]), test_idx)) )

    #[Batch, 2, signal_length]
    X_train = np.load("../RadioML/X_train.npy")
    X_test = np.load("../RadioML/X_test.npy")


    #One hot label for signals
    Y_train = np.load("../RadioML/Y_train.npy")
    Y_test = np.load("../RadioML/Y_test.npy")

    #max-min normalize
    X_train[:, 0, :] = (X_train[:, 0, :] - np.min(X_train[:, 0, :])) / (np.max(X_train[:, 0, :]) - np.min(X_train[:, 0, :]))
    X_train[:, 1, :] = (X_train[:, 1, :] - np.min(X_train[:, 1, :])) / (np.max(X_train[:, 1, :]) - np.min(X_train[:, 1, :]))

    X_test[:, 0, :] = (X_test[:, 0, :] - np.min(X_test[:, 0, :])) / (np.max(X_test[:, 0, :]) - np.min(X_test[:, 0, :]))
    X_test[:, 1, :] = (X_test[:, 1, :] - np.min(X_test[:, 1, :])) / (np.max(X_test[:, 1, :]) - np.min(X_test[:, 1, :]))
    
        
    M = np.expand_dims(np.linalg.norm(X_train, axis=1), axis=1)
    T = np.expand_dims(np.arctan2(X_train[:,1,:],X_train[:,0,:]), axis=1)
    X_train = np.concatenate((T, M), axis=1)
    
    X_train = np.expand_dims(np.expand_dims(X_train, axis=2), axis=4)
        
    M_ = np.expand_dims(np.linalg.norm(X_test, axis=1), axis=1)
    T_ = np.expand_dims(np.arctan2(X_test[:,1,:],X_test[:,0,:]), axis=1)
    X_test = np.concatenate((T_, M_), axis=1)
    
    X_test = np.expand_dims(np.expand_dims(X_test, axis=2), axis=4)
    
    Y_train = np.argmax(Y_train, axis=1)
    Y_test = np.argmax(Y_test, axis=1)
    
    train_loader = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy (Y_train).type(torch.LongTensor))
    train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = True)
    test_loader = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))
    test_loader_dataset = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle = False)
    
#     return train_loader_dataset, test_loader_dataset, lbl, snrs, test_idx

    return train_loader_dataset, test_loader_dataset


#Default device
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')



# def test(model, device, test_loader, lbl, snrs, test_idx):
#     test_loss = 0
#     correct = 0
#     pred_all = []
#     real_all = []
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output, losses = model(data)
#             test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
#             pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
#             pred_all.append(np.array(pred.cpu()))
#             real_all.append(np.array(target.cpu()))
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     pred_all = np.squeeze( np.concatenate(pred_all) )
#     real_all = np.concatenate(real_all)
    
#     acc = {}
#     for snr in snrs:
#         # extract classes @ SNR
#         test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
#         pred_i = pred_all[np.where(np.array(test_SNRs)==snr)]
#         real_i = real_all[np.where(np.array(test_SNRs)==snr)]
#         logger.info('SNR ' +str(snr)+' test accuracy: '+str(100. * np.mean(pred_i==real_i) )+'%.')  
        
#     test_loss /= len(test_loader.dataset)
#     logger.info('\nTest set: Average loss: {:.4f}, Overall accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))
#     return 100. * correct / len(test_loader.dataset)

def test(model, device, test_loader):
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    with torch.no_grad():
        for data, target in test_loader:
            targets = target.cpu().numpy()
            
            data, target = data.to(device), target.to(device)
            output, losses = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            #For confusion matrix
            pred_all = np.concatenate((pred_all, pred.cpu().numpy()), axis=0)
            real_all = np.concatenate((real_all, target.unsqueeze(1).cpu().numpy()), axis=0)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return 100. * correct / len(test_loader.dataset)


def classification(out,desired):
    _, predicted = torch.max(out, 1)
    total = desired.shape[0]
    correct = (predicted == desired).sum().item()
    return correct

def draw_confusion_matrix(y_true, y_pred, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    matrix = cm > 0
    final = np.zeros((cm.shape[0], cm.shape[1], 3))
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if i == j:
                final[i][j] = [84 / 255., 118 / 255., 33 / 255.]
            elif matrix[i][j] != 0:
                final[i][j] = [208 / 255., 123 / 255., 12 / 255.]
            else:
                final[i][j] = [1., 1., 1.]
            
    fig, ax = plt.subplots()
    im = ax.imshow(final, interpolation='nearest')
#     ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='MSTAR Dataset Classification',
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                if cm[i, j] < 0.01:
                    cm[i, j] = 0.01
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('temp.png', dpi=fig.dpi)
    fig.savefig('temp.eps', dpi=fig.dpi, format='eps')
    
    

max_epochs = Params_dict['max_epochs']
logger = setup_logger('JS logger')
        
logger.info(str(Params_dict))
model_name = 'radioML'
logger.info(model_name)
batches = Params_dict['batches']
lrs = Params_dict['lrs']


save_path = None
torch.manual_seed(42222222)
np.random.seed(42222222)

for b in batches:
    for lr in lrs:
        new = True
        logger.info('Batch[{b}]-LR[{lr}]'.format(b=b, lr=lr))
        manifold_net = model.ManifoldNetR(Params_dict['num_classes'], Params_dict['num_distributions'], Params_dict['num_repeat']).cuda()
        logger.info(manifold_net.name)
        init_params = manifold_net.parameters()
        model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
        logger.info(str([p.size() for p in model_parameters]))
        model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
        logger.info("Model Parameters: "+str(sum([np.prod(p.size()) for p in model_parameters])))
#         manifold_net.load_state_dict(torch.load('./save/Res-acc[98.393]-[100]-[0.008]-11class-model.ckpt'))
        optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        

        # Training...
        print('Starting training...')
        validation_accuracy = []
        highest=0
        train_generator, test_generator = data_prep(b)
#         , lbl, snrs, test_idx = data_prep(b)
        epoch_validation_history = []

        for epoch in range(max_epochs):
            print('Starting Epoch ', epoch, '...')
            loss_sum = 0
            start = time.time()
            train_acc = 0
            tot_len=0
            tot_loss=0
            for it,(local_batch, local_labels) in enumerate(train_generator):
                batch = torch.tensor(local_batch, requires_grad=True).cuda()
                optimizer.zero_grad()
                out, losses = manifold_net(batch, local_labels.cuda())
                
                
#                 train_acc += classification(out, local_labels.cuda()) 
#                 loss = criterion(out, local_labels.cuda())
                
                
                
                
                train_acc += classification(out, local_labels.cuda()) 
                loss = criterion(out, local_labels.cuda())

                ##Can customize losses
                losses = sum(losses)
                sums = torch.sum(losses)
                loss = loss + sums 
                ##

                print(loss)
                loss.backward()
                optimizer.step()
                
                tot_len += len(out)
                tot_loss += loss




            logger.info("Epoch: "+str(epoch)+" - Training accuracy: "+str(train_acc / (tot_len)*100.))
            logger.info("Model: "+str(manifold_net.name)+" - Loss: "+str(tot_loss / it))
            
#             acc = test(manifold_net, device, test_generator, lbl, snrs, test_idx)
            acc = test(manifold_net, device, test_generator)
            
#             model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
#             print(True in [True in torch.isnan(p).detach().cpu().numpy() for p in model_parameters])
            manifold_net.clear_weights()
            if acc > highest:
                highest=acc
                if save_path != None and not new:
                    os.remove(save_path)

                save_path = os.path.join('./save/', 'Radio-[{acc}]-[{batch}]-[{learning_rate}]-11class-model-[{num_distr}]-[{name}].ckpt'.format(acc = np.round(acc, 3), name=manifold_net.name, batch=b, learning_rate=lr, num_distr = Params_dict['num_distributions']))
                new = False
                torch.save(manifold_net.state_dict(), save_path)
                print('Saved model checkpoints into {}...'.format(save_path))
            logger.info("Epoch: "+str(epoch)+" - Testing accuracy is "+str(acc))
        manifold_net = model.ManifoldNetComplexRadio(Params_dict['num_classes'], Params_dict['num_distributions'], Params_dict['num_repeat']).cuda()
        optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
