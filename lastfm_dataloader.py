#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 02:19:59 2019

@author: mimbres
"""

import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
#from utils.matrix_math import indices_to_one_hot

TAG_PATH = './data/lastfm_top50_tagmtx.npy'
FEAT_PATH = './data/lastfm_top50_featmtx.npy'
NUM_TRSET = 100000
# TRSET: 100,000
# TSSET: 23,204


def LastFMDataloader( mtrain_mode=True,
                      data_sel=None,
                      normalization_factor=100,
                      batch_size=1,
                      shuffle=False,
                      num_workers=4,
                      pin_memory=True):
    
    dset = LastFMDataset(mtrain_mode=mtrain_mode,
                         normalization_factor=normalization_factor,
                         data_sel=data_sel)
    dloader = DataLoader(dset,
                         batch_size=batch_size,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         pin_memory=pin_memory)
    return dloader



class LastFMDataset(Dataset):
    def __init__(self,
                 mtrain_mode=True,
                 normalization_factor=100,
                 data_sel=None):
        
        self.mtrain_mode = mtrain_mode
        self.data_sel    = data_sel # NOT IMPLEMENTED YET
        self.tag_all     = []
        self.feat_all    = []
        
        # Import data
        tag_all = np.load(TAG_PATH)
        feat_all = np.load(FEAT_PATH)
        
        # Train/test split (8:2 by default)
        if self.mtrain_mode:
            self.tag_all  = tag_all[:NUM_TRSET,:]
            self.feat_all = feat_all[:NUM_TRSET,:]
        else:
            self.tag_all  = tag_all[NUM_TRSET:,:]
            self.feat_all = feat_all[NUM_TRSET:,:]
            
        # Normalize tag probability
        self.tag_all = self.tag_all.astype(np.float32) / normalization_factor
        return None
    
    
    
    def __getitem__(self, index):
        tag = self.tag_all[index,:]
        feat = self.feat_all[index, :]
        return index, tag, feat 
    
    
    
    def __len__(self):
        return len(self.tag_all) # return the total number of items
    
    
    