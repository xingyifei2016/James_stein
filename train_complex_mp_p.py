# -*- coding: utf-8 -*-
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
import scipy.io as sio


Params_dict = {
    'lrs': [0.007, 0.005, 0.009],
    'batches': [100, 200, 100, 100],
    'max_epochs': 150,
    'test_batch': 1,
    'num_classes': 11,
    'num_distributions': 8,
    'num_repeat': 100,
}


def data_prep_11(batch_size, split=0.7):
#     data_x = []
#     data_y = []
#     for f in listdir('./data_polar'):
#         data = np.load(join('./data_polar', f))
#         label = f.split('_')[0].split('c')[1]
# #         if int(label) != 11:
#         data_x.append(data)
#         data_y.append(int(label)-1)

#     data_x = np.array(data_x)
#     data_y = np.array(data_y)
    
    
#     xshape = data_x.shape
    
#     data_x = data_x.reshape((xshape[0], xshape[1], 1, xshape[2], xshape[3]))
    
#     data_x[:, 0,...] = np.arccos(data_x[:, 0, 0,...]).reshape(data_x[:, 0,...].shape)
#     data_x[:, 1,...] = data_x[:, 4,...]
#     data_x = data_x[:, :2,...]
#     np.save('./data_polar/x_data.npy', data_x)
#     np.save('./data_polar/y_data.npy', data_y)
    data_x = np.load('./data_polar/x_data.npy')
    data_y = np.load('./data_polar/y_data.npy')
    
    
    
    
    
    
    data_set_11 = torch.utils.data.TensorDataset(torch.from_numpy(data_x).type(torch.FloatTensor), torch.from_numpy (data_y).type(torch.LongTensor))
    train_idx, test_idx = index_split(False)
    data_train = torch.utils.data.Subset(data_set_11,indices=train_idx)
    data_test = torch.utils.data.Subset(data_set_11,indices=test_idx)
    params_train = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}

    params_val = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 1}
    
    train_generator = torch.utils.data.DataLoader(dataset=data_train, **params_train)
    test_generator = torch.utils.data.DataLoader(dataset=data_test, **params_val)
    return train_generator, test_generator 


#Default device
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')


def index_split(full_or_no):
    csv_path = './chipinfo.csv' #os.path.join('./save/', 'split[{split}]-10class-model-temp.ckpt'.format(split=i))
    df = pd.read_csv(csv_path)

    training = df.loc[df['depression'] == 17]

    subclass_9 = training.loc[training['target_type'] != 'bmp2_tank']
    subclass_8 = subclass_9.loc[subclass_9['target_type'] != 't72_tank'].index.values


    class_1_train = np.array(training.loc[training['serial_num']=='c21'].index.values)
    class_3_train = np.array(training.loc[training['serial_num']=='132'].index.values)


    subclass = np.concatenate([subclass_8, class_1_train, class_3_train], axis=0)
    training = training.index

    testing = df.loc[df['depression'] == 15]

    subclass_test9 = testing.loc[testing['target_type'] != 'bmp2_tank']
    subclass_test8 = np.array(subclass_test9.loc[subclass_test9['target_type']=='t72_tank'].index.values)

#     class_1_test1 = np.array(testing.loc[testing['serial_num']=='c21'].index.values)
    class_1_test2 = np.array(testing.loc[testing['serial_num']=='9563'].index.values)
    class_1_test3 = np.array(testing.loc[testing['serial_num']=='9566'].index.values)

#     class_3_test1 = np.array(testing.loc[testing['serial_num']=='132'].index.values)
    class_3_test2 = np.array(testing.loc[testing['serial_num']=='812'].index.values)
    class_3_test3 = np.array(testing.loc[testing['serial_num']=='s7'].index.values)

    subclass_test = np.concatenate([subclass_test8, class_1_test2, class_1_test3, class_3_test2, class_3_test3], axis=0)
    
    testing = np.array(testing.index.values)
        
    if full_or_no:
        return training, testing
    else:
        return subclass, subclass_test
        
def test_val(model, device, test_loader):
    test_loss = 0
    correct = 0
    pred_all = np.array([[]]).reshape((0, 1))
    real_all = np.array([[]]).reshape((0, 1))
    with torch.no_grad():
        for data, target in test_loader:
            targets = target.cpu().numpy()
            
            data, target = data.to(device), target.to(device)
            x0, x1, x2, x3, x4, x5, output, losses = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            
            #For confusion matrix
            pred_all = np.concatenate((pred_all, pred.cpu().numpy()), axis=0)
            real_all = np.concatenate((real_all, target.unsqueeze(1).cpu().numpy()), axis=0)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return 100. * correct / len(test_loader.dataset), pred_all, real_all    


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
    
    return 100. * correct / len(test_loader.dataset), pred_all, real_all

def classification(out,desired):
    _, predicted = torch.max(out, 1)
    total = desired.shape[0]
    correct = (predicted == desired).sum().item()
    return correct


def visualize(model, data_generator):
    kk=1
    save_dict = {}
    for it,(local_batch, local_labels) in enumerate(data_generator):
        batch = torch.tensor(local_batch, requires_grad=True).cuda()
        x0, x1, x2, x3, x4, x5, x, res_loss = model(batch, None)
        label = local_labels.item()
        save_dict['c'+str(label)] = []
        
        img = x0[0, :, 0, :, :].cpu().detach().numpy()
        
        
        #H is phase V is magnitude
        #[2, H, W] phase, mag
        img = img.transpose((1, 2, 0))
        img = np.insert(img, 1, 1, axis=2)
        
        #[128, 1]
        img[:, :, 0] = img[:, :, 0] / math.pi
        img[:, :, 2] = np.sqrt(img[:, :, 2])
        
        save_dict['c'+str(label)].append(img)
        
        img1 = x1[0, :, 6:16, :, :].cpu().detach().numpy()
        img1 = img1.transpose((2, 3, 0, 1))
        img1 = np.insert(img1, 1, 1, axis=2)

        img1[:, :, 2, :] = np.sqrt(img1[:, :, 2, :])

        min_0 = np.min(img1[:, :, 0, :], keepdims=True)
        max_0 = np.max(img1[:, :, 0, :], keepdims=True)

        min_2 = np.min(img1[:, :, 2, :], keepdims=True)
        max_2 = np.max(img1[:, :, 2, :], keepdims=True)

        img1[:, :, 0, :] = (img1[:, :, 0, :] - min_0)/ (max_0-min_0)
        img1[:, :, 2, :] = (img1[:, :, 2, :] - min_2)/ (max_2-min_2)
        
        save_dict['c'+str(label)].append(img1)
        
        

        img2 = x2[0, 0:10, :, :].cpu().detach().numpy()
        img2 = img2.transpose((1, 2, 0))
        
        img2 = (img2 - np.min(img2)) / (np.max(img2) - np.min(img2))
        
        save_dict['c'+str(label)].append(img2)
        
        kk+=1

        img3 = x3[0, 0:10, :, :].cpu().detach().numpy()
        img3 = img3.transpose((1, 2, 0))
        
        img3 = (img3 - np.min(img3)) / (np.max(img3) - np.min(img3))

        save_dict['c'+str(label)].append(img3)

        kk+=1
        
        img4 = x4[0, 0:10, :, :].cpu().detach().numpy()
        imshape = img4.shape
        img4 = (img4 - np.min(img4)) / (np.max(img4) - np.min(img4))
        save_dict['c'+str(label)].append(img4)

        kk+=1

        img5 = x5[0, :, :, :].cpu().detach().numpy().reshape(x5.shape[1], 1)
        save_dict['c'+str(label)].append(img5)

        kk+=1
    sio.savemat('data_new.mat', save_dict)
    
    

def draw_confusion_matrix(y_true, y_pred, acc, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], cmap=plt.cm.Blues):
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
    fig.savefig('temp'+str(acc)+'.png', dpi=fig.dpi)
    fig.savefig('temp'+str(acc)+'.eps', dpi=fig.dpi, format='eps')
    
    
    

# Parameters for data loading
params_train = {'shuffle': False,
          'num_workers': 1}

params_val = {'batch_size': Params_dict['test_batch'],
          'shuffle': False,
          'num_workers': 1}

max_epochs = Params_dict['max_epochs']
logger = setup_logger('JS logger')
model_name = 'MSTAR'
logger.info(model_name)    
logger.info(str(Params_dict))
batches = Params_dict['batches']
lrs = Params_dict['lrs']


batches = [100, 200, 300, 500]
lrs = [0.008, 0.01]
save_path = None
torch.manual_seed(42222222)
np.random.seed(42222222)
# np.random.seed(42222222)
distr = [5]
for ss in distr:
    for b in batches:
        for lr in lrs:
            new = True
            logger.info('Batch[{b}]-LR[{lr}]'.format(b=b, lr=lr))
            manifold_net = model.Test(Params_dict['num_classes'], ss, Params_dict['num_repeat']).cuda()
            init_params = manifold_net.parameters()
            model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
            logger.info(str([p.size() for p in model_parameters]))
            model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
            params = sum([np.prod(p.size()) for p in model_parameters])
            logger.info("Model Parameters: "+str(params))
            manifold_net.load_state_dict(torch.load('./save/[98.055]-[100]-[0.008]-11class-model-[5].ckpt'))
            optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()



            # Training...
            print('Starting training...')
            validation_accuracy = []
            highest=0
            #split = 0

                #for train_idx, test_idx in data_utils.k_folds(n_splits=10, n_samples=(14557)):

            train_generator, test_generator = data_prep_11(b, 0.3)
            epoch_validation_history = []

#             for epoch in range(max_epochs):
#                 print('Starting Epoch ', epoch, '...')
#                 loss_sum = 0
#                 start = time.time()
#                 train_acc = 0
#                 tot_len=0
#                 tot_loss=0
#                 for it,(local_batch, local_labels) in enumerate(train_generator):
#                     batch = torch.tensor(local_batch, requires_grad=True).cuda()
#                     optimizer.zero_grad()
#                     out, losses = manifold_net(batch, local_labels.cuda())
#                     train_acc += classification(out, local_labels.cuda()) 
#                     loss = criterion(out, local_labels.cuda())

#                     ##Can customize losses
#                     losses = sum(losses)
#                     sums = torch.sum(losses)
#                     loss = loss + sums 
#                     ##

#                     loss.backward()
#                     optimizer.step()
#                     optimizer.zero_grad()
#                     tot_len += len(out)
#                     tot_loss += loss




#                 logger.info("Epoch: "+str(epoch)+"Training accuracy: "+str(train_acc / (tot_len)*100.))
#                 logger.info("Loss: "+str(tot_loss / it))
            visualize(manifold_net, test_generator)
#             acc, pred, real = test_val(manifold_net, device, test_generator)
            print(acc)
            manifold_net.clear_weights()
            exit()
#                 if acc > highest:
#                     highest=acc
#                     if save_path != None and not new:
#                         os.remove(save_path)
#                     draw_confusion_matrix(real, pred, acc)
#                     save_path = os.path.join('./save/', '[{acc}]-[{batch}]-[{learning_rate}]-11class-model-[{num_distr}].ckpt'.format(acc = np.round(acc, 3), batch=b, learning_rate=lr, num_distr = ss))
#                     new = False
#                     torch.save(manifold_net.state_dict(), save_path)
#                     print('Saved model checkpoints into {}...'.format(save_path))
#                 logger.info("Epoch: "+str(epoch)+"Testing accuracy is "+str(acc))
#     #                     print('Epoch Time:', end-start)
#     #                 accs.append(highest)
            manifold_net = model.ManifoldNetComplex1(Params_dict['num_classes'], ss, Params_dict['num_repeat']).cuda()
            optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

        
        
        

    
        
                            
                    
                            
            
            
    






