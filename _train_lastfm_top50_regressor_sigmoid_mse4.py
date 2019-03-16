#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 05:31:35 2019

@author: mimbres
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.backends import cudnn
import numpy as np
import argparse, os
from tqdm import trange, tqdm
from lastfm_dataloader import LastFMDataloader
from utils.plot_save_png import regression_img_save
cudnn.benchmark = True

parser = argparse.ArgumentParser(description="lastfm_model")
parser.add_argument("-s","--save_path",type=str, default="./save/exp_lastfm_top50_regressor_sigmoid_mse4/")
parser.add_argument("-e","--epochs",type=int, default= 2000)
parser.add_argument("-si","--save_interval",type=int, default=200)
parser.add_argument("-lr","--learning_rate", type=float, default = 0.001)
parser.add_argument("-b","--train_batch_size", type=int, default = 1024)
parser.add_argument("-tsb","--test_batch_size", type=int, default = 2048)
parser.add_argument("-g","--gpu",type=int, default=0)
args = parser.parse_args()


# Hyper Parameters
INPUT_DIM = 29 
EMBEDDING_DIM = 128
OUTPUT_DIM = 50 
EPOCHS = args.epochs
SAVE_INTERVAL = args.save_interval
LEARNING_RATE = args.learning_rate
TR_BATCH_SZ = args.train_batch_size
TS_BATCH_SZ = args.test_batch_size
GPU = args.gpu

# Model-save directory
MODEL_SAVE_PATH = args.save_path
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Logs
hist_trloss = list()
#hist_tracc  = list()
hist_vloss  = list()
#hist_vacc   = list()
np.set_printoptions(precision=3)

#%% MODEL
class MLP_Regressor(nn.Module):
    def __init__(self, input_d=INPUT_DIM, output_d=OUTPUT_DIM,
                 emb_d=EMBEDDING_DIM, mid_d=256): # emb_dim means the l-1 layer's dimension
        super(MLP_Regressor, self).__init__()
        self.embedding_layer  = nn.Sequential(nn.Linear(input_d, mid_d),
                                              nn.ReLU(),
                                              nn.Linear(mid_d, emb_d))
        self.final_layer = nn.Linear(emb_d, output_d)
        
    def forward(self, x): # Input:
        emb = self.embedding_layer(x)
        x = self.final_layer(emb)
        x = F.sigmoid(x)
        return emb, x

model = MLP_Regressor().cuda(GPU)
optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
scheduler = StepLR(optim, step_size=20, gamma=0.6)  


#%% Train/test methods
def Train():
    mtrain_loader = LastFMDataloader(mtrain_mode=True, normalization_factor=100, batch_size=TR_BATCH_SZ, shuffle=True)
    
    for epoch in trange(0, EPOCHS, desc='epochs', position=0, ascii=True):
        tr_iter = iter(mtrain_loader)
        #total_corrects = 0
        total_items    = 0
        total_trloss   = 0

        scheduler.step()
        for it in range(len(tr_iter)):
            index, y, x = tr_iter.next() 
            #y_numpy = y.numpy()
            y = Variable(y).cuda(GPU)
            x = Variable(x).cuda(GPU)
            
            model.train();
            _emb, y_hat = model(x)
            loss = F.mse_loss(input=y_hat, target=y)  # MSE Loss           
            #loss = F.smooth_l1_loss(input=y_hat, target=y)
            model.zero_grad()
            loss.backward()            
            optim.step()
            
            total_items += TR_BATCH_SZ
            total_trloss += loss.item()                
                
        vloss, samples = Validate()    
        
        hist_trloss.append(total_trloss/total_items)
        hist_vloss.append(vloss)
        tqdm.write("tr_epoch:{0:}  tr_loss:{1:.9f}  val_loss:{2:.9f}".format(epoch, 
                   hist_trloss[-1], hist_vloss[-1]))
        #save model and image samples
        if epoch%SAVE_INTERVAL == 0:
            regression_img_save(samples[1], samples[2], out_path=MODEL_SAVE_PATH + 'samples_' + str(epoch) + '.png')
            torch.save({'ep': epoch, 'model_state': model.state_dict(), 'trloss': hist_trloss,
                        'vloss': hist_vloss, 'opt_state': optim.state_dict()},
                        MODEL_SAVE_PATH + "check_{0:}.pth".format(epoch))
            
            
            
    return

def Validate():
    mval_loader = LastFMDataloader(mtrain_mode=False, normalization_factor=100, batch_size=TS_BATCH_SZ, shuffle=False)
    val_iter = iter(mval_loader)
    #total_corrects = 0
    total_items    = 0
    total_vloss   = 0

    for it in range(len(val_iter)):
        index, y, x = val_iter.next() 
        #y_numpy = y.numpy()
        y = Variable(y, requires_grad=False).cuda(GPU)
        x = Variable(x, requires_grad=False).cuda(GPU)
        
        model.eval();
        _emb, y_hat = model(x)
        loss = F.mse_loss(input=y_hat, target=y)  # MSE Loss           
        #loss = F.smooth_l1_loss(input=y_hat, target=y)
        
        total_items += TR_BATCH_SZ
        total_vloss += loss.item()
    
    # Collect samples to display
    samples = (x.cpu().detach().numpy()[:5,:], y.cpu().detach().numpy()[:5,:],
               y_hat.cpu().detach().numpy()[:5,:], _emb.cpu().detach().numpy()[:5,:])
    return total_vloss/total_items, samples
            

#%% Main
def main():
    Train()
    return

if __name__ == '__main__':
    main()  

            
            
            
