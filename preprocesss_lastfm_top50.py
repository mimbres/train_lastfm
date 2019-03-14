#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:11:07 2019

@author: mimbres
"""

import pandas as pd
import numpy as np
from tqdm import trange

LASTFM_FILEPATH = './data/final_mapping.json'
OUTPUT_FILEPATH1 = './data/lastfm_top50_tagmtx.npy'
OUTPUT_FILEPATH2 = './data/lastfm_top50_featmtx.npy'
OUTPUT_FILEPATH3 = './data/lastfm_top50_track_ids.npy'
OUTPUT_FILEPATH4 = './data/lastfm_top50_tag_avail_cnt.npy'
SAVED_SCALER_FILEPATH = './data/std_scaler.sav'

TOP50A = ['rock', 'pop', 'alternative', 'indie', 'favorites', 'female vocalists',
          'Love', 'alternative rock', 'electronic', 'beautiful', 'jazz', '00s',
          'singer-songwriter', 'metal', 'male vocalists', 'Awesome', 'american', 
          'Mellow', 'classic rock', '90s', 'soul', 'chillout', 'punk', '80s', 'chill',
          'indie rock', 'folk', 'dance', 'instrumental', 'hard rock', 'oldies',
          'seen live', 'Favorite', 'country', 'blues', 'guitar', 'cool', 'british',
          'acoustic', 'electronica', '70s', 'Favourites', 'Hip-Hop', 'experimental',
          'easy listening', 'female vocalist', 'ambient', 'punk rock', 'funk', 'hardcore']
_dict = {'major': 1, 'minor': 0}



# Load .json file...
df=pd.read_json(LASTFM_FILEPATH)
num_items = len(df)

# Shuffle (we can split train/test later)
df = df.sample(frac=1).reset_index(drop=True)


# Create an empty result matrix
tag_mtx = np.zeros((num_items,50))
feat_mtx = np.zeros((num_items,29))
track_ids = np.ndarray((num_items,), dtype=object)
tag_avail_cnt = np.zeros((num_items,))


for i in trange(num_items):
    item = np.asarray(df[0][i]) # Get one item 
    
    tag_cnt = 0
    for tag in TOP50A:
        # Check availability of each tag in this item
        _idx = np.where(tag == item)[0]
        if len(_idx) is not 0: # If top50-tag available...
            tag_cnt += 1
            column_idx = _idx[0]
            #print(i, item[column_idx,:])
            tag_mtx[i,TOP50A.index(tag)] = item[column_idx,1].astype(np.float)
    
    tag_avail_cnt[i] = tag_cnt  
    track_ids[i] = df[1][i][0]
    if tag_cnt is not 0:
        _feat = np.asarray(df[1][i])
        _feat[20] = _dict.get(_feat[20]) # {'major', 'minor'} --> {0,1}
        _feat[5] = _feat[5][:4] # '2005-01-01' --> '2005'
        feat_mtx[i,:] = _feat[[4,5,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]]
        
print('max available tags =', np.max(tag_avail_cnt), '\n',
      'avg available tags =', np.mean(tag_avail_cnt[np.where(tag_avail_cnt!=0)]), '\n',
      'items with top50 unavailable =', len(np.where(tag_avail_cnt==0)[0]), '\n',
      'items with top50 available =', len(np.where(tag_avail_cnt!=0)[0]) )

'''
max available tags = 31.0 
 avg available tags = 4.705301775916366 
 items with top50 unavailable = 38595 
 items with top50 available = 123204
'''

# Reduce top50 unavailable items
tag_mtx = tag_mtx[tag_avail_cnt!=0,:]
feat_mtx = feat_mtx[tag_avail_cnt!=0,:]
track_ids = track_ids[tag_avail_cnt!=0]


# Feature normalization
import pickle
#from sklearn.preprocessing import StandardScaler
scaler = pickle.load(open(SAVED_SCALER_FILEPATH, 'rb'))
feat_mtx_new = scaler.fit_transform(feat_mtx)
feat_mtx_new[:,15] = feat_mtx[:,15]


# Save results as .npy
np.save(OUTPUT_FILEPATH1, tag_mtx.astype(np.int8))
#np.save(OUTPUT_FILEPATH2, feat_mtx.astype(np.int8))
np.save(OUTPUT_FILEPATH2, feat_mtx_new.astype(np.float32))
np.save(OUTPUT_FILEPATH3, track_ids)
np.save(OUTPUT_FILEPATH4, tag_avail_cnt.astype(np.int8))
    
            
            
            
            