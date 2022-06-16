#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 12:41:06 2022

@author: andrea
"""

import twixtools

#filename = "RAW/meas_MID00298_FID79321_gre_1H_2_5x2_5x(4+1)mm.dat"
import twixtools
import numpy as np
import matplotlib.pyplot as plt
import os

import glob

# import file list
os.chdir("/media/andrea/DATA/giove/long_19F/")
filelist = []
for file in glob.glob("*.dat"):
    filelist.append(file)
    
filelist.sort()
#%%
def ifftnd(kspace, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, ifftn
    if axes is None:
        axes = range(kspace.ndim)
    img = fftshift(ifftn(ifftshift(kspace, axes=axes), axes=axes), axes=axes)
    img *= np.sqrt(np.prod(np.take(img.shape, axes)))
    return img


def fftnd(img, axes=[-1]):
    from numpy.fft import fftshift, ifftshift, fftn
    if axes is None:
        axes = range(img.ndim)
    kspace = fftshift(fftn(ifftshift(img, axes=axes), axes=axes), axes=axes)
    kspace /= np.sqrt(np.prod(np.take(kspace.shape, axes)))
    return kspace

def rms_comb(sig, axis=1):
    return np.sqrt(np.sum(abs(sig)**2, axis))


def get_kspace(twix):
    # sort all 'imaging' mdbs into a k-space array
    image_mdbs = [mdb for mdb in twix[-1]['mdb'] if mdb.is_image_scan()]
    
    n_line = 1 + max([mdb.cLin for mdb in image_mdbs])

# assume that all data were acquired with same number of channels & columns:
    n_channel, n_column = image_mdbs[0].data.shape
    norm=len(image_mdbs)/n_line
    kspace = np.zeros([n_line, n_channel, n_column], dtype=np.complex64)
    for mdb in image_mdbs:
        kspace[mdb.cLin] += mdb.data/norm
    
    print('\nk-space shape', kspace.shape)
    print('\nk-space averages', norm)
    return kspace

def do_image(kspace):
    image = ifftnd(kspace, [0,-1])
    image = rms_comb(image)
    return image
    


# %% 
def plot_k_basic(kspace,image):
    plt.figure(figsize=[12,8])
    plt.subplot(121)
    plt.title('k-space')
    plt.imshow(abs(kspace[:,0])**0.2, cmap='gray', origin='lower')
    plt.axis('off')
    
    
    plt.subplot(122)
    plt.title('image')
    plt.imshow(abs(image), cmap='gray', origin='lower')
    plt.axis('off')



# %%

# def read_data(file):
#     try:
#         print(f"reading from file:\n ",file)
#         twix = twixtools.read_twix(file)
#         print(f"Averaging k-space:\n ")
#         kspace = get_kspace(twix)
        
#         image = do_image(kspace)
#         return kspace,image
#     except: 
#         return("Did not find image")

# from concurrent.futures import ThreadPoolExecutor, as_completed

# # suppose the files contains th 16k file names

# future_to_file = {}
# k_spaces = []
# images = []   

# with ThreadPoolExecutor(max_workers=4) as executor:
#     for file in filelist:
#         future = executor.submit(read_data, file)
#         future_to_file[future] = file
    
#     for future in as_completed(future_to_file):
#         file = future_to_file[future]
#         kspace,image = future.result()
#         if kspace != 'Did not find image':
#             k_spaces.append((file, kspace))
#             images.append((file, image))
            

# %%

k_spaces = []
images = []   
for file in filelist:
    print(f"reading from file:\n ",file)
    twix = twixtools.read_twix(file)
    print(f"Averaging k-space:\n ")
    kspace = get_kspace(twix)
    k_spaces.append(kspace)
    image = do_image(kspace)
    images.append(image)
    twix = 0 
    

#%% total image
total_kspace = np.zeros(kspace.shape)
for k,im in zip(k_spaces,images):
   print("k-image shape",k.shape)
   total_kspace += np.array(k)/len(k_spaces)
   plot_k_basic(k,im)

total_image = do_image(total_kspace)

plot_k_basic(total_kspace,total_image)

np.savez("long_images_averaged.npz",k_space_list=k_spaces,image_list=images)
# %%
#plot_k_basic(kspace,image)
# twix0 = twixtools.read_twix(filelist[1])
# dict0 = twix0[-1]['hdr']['MeasYaps']['sSliceArray']['asSlice']
# twix1 = twixtools.read_twix(filelist[78])
# dict1 = twix1[-1]['hdr']['MeasYaps']['sSliceArray']['asSlice']


#for mdb in twix[-1]['mdb'][::8000]:
    #print('line: %3d; flags:'%(mdb.cLin), mdb.get_active_flags())


