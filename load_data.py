#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:56:21 2022

@author: andrea
"""
import numpy as np
from open_long import do_image,plot_k_basic
import matplotlib.pyplot as plt
# load data 

def load_data_fede(file):
    with np.load(file) as data:
        kspace = data["k_space_list"]
        images = data["image_list"]
        return kspace,images
def average_k(k_spaces):
    total_kspace = np.zeros(k_spaces[0].shape,dtype='complex128')
    for k,im in zip(k_spaces,images):
        print("k-image shape",k.shape)
        total_kspace += k/np.array(len(k_spaces)).astype(complex)
        
        
        
        # subsampling phase
        total_kspace = total_kspace[:,:,::2]
        
        total_image = do_image(total_kspace)
        total_kspace = np.squeeze(total_kspace)
        #plot_k_basic(total_kspace,total_image)
        
        return total_kspace, total_image
    
if __name__ == "__main__":  
    kspace,images = load_data_fede("long_images_averaged.npz")
    
    total_kspace, total_image = average_k(kspace)
    # %%
    plt.figure(figsize=(8,10))
    plt.title('image')
    plt.imshow(abs(total_image), cmap='gray', origin='lower')
    plt.axis('off')
    
    plt.figure()
    plt.hist(total_image.ravel(),200)
    plt.yscale("log")
    
    # %% 
   
    
   
    print(total_image[:,115:130])
    sig = total_image[:,60:68]
    bg =total_image[:,0:8]
    plt.bar(np.arange(96),sig.max(axis=-1),alpha=0.7,label="signal?")
    plt.bar(np.arange(96),bg.max(axis=-1),alpha=0.7,label="noise")
    plt.legend()
    plt.yscale("log")
    plt.title("original")
