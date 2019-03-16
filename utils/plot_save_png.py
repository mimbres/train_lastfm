#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 08:24:57 2019

@author: mimbres
"""
import matplotlib.pyplot as plt
import seaborn as sns



def regression_img_save(y_hats, ys, out_path):
    plt.ioff()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.figsize"] = [6,6]
    # plotting tr_loss with val_loss
    fig, ax = plt.subplots(5, sharex=True )
    for i in range(5):
        ax[i].plot(y_hats[i,:], 'o-', markersize=2, label="Ground-truth")
        ax[i].plot(ys[i,:],'o-', markersize=2, label="Predicted")
        ax[i].set_xlabel('dim.', fontsize = 'small')
        ax[i].set_ylabel('value', fontsize = 'small')
        ax[i].legend(loc=1, fontsize = 'x-small')
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close('all')
    return

def confusion_mtx_img_save(cnf_df, norm_cnf_df, filepath_cnf_png):
    plt.rcParams["figure.figsize"] = [10,4]
    sns.set(font_scale=0.4)
    plt.subplot(121)
    ax = sns.heatmap(cnf_df, annot=True, fmt="g")
    plt.subplot(122)
    ax = sns.heatmap(norm_cnf_df, annot=True, fmt=".2f")
    plt.title('y-axis = target, x-axis = prediction')
    plt.savefig(filepath_cnf_png, bbox_inches='tight', dpi=220)
    del(ax)
    plt.close('all')
    return