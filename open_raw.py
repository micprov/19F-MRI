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
os.chdir("/media/andrea/DATA/giove/RAW/")
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

    kspace = np.zeros([n_line, n_channel, n_column], dtype=np.complex64)
    for mdb in image_mdbs:
        kspace[mdb.cLin] = mdb.data
    
    print('\nk-space shape', kspace.shape)
    return kspace

def do_image(kspace):
    image = ifftnd(kspace, [0,-1])
    image = rms_comb(image)
    return image
    

def read_file(filename,INFO_ONLY=True, PLOT=False):
# parse the twix file
    twix = twixtools.read_twix(filename)

# twix is a list of measurements:
    print('\nnumber of separate scans (multi-raid):', len(twix))
    amplitude = twix[-1]['hdr']['Spice']['TransmitterReferenceAmplitude']
    
    sliceinfo = twix[-1]['hdr']['MeasYaps']['sAAInitialOffset']['SliceInformation']
    
    ReadFoV = twix[-1]['hdr']['Config']['ReadFoV']
    PhaseFoV = twix[-1]['hdr']['Config']['PhaseFoV']
     
    rot = sliceinfo['dInPlaneRot'] if 'dInPlaneRot' in sliceinfo.keys() else 0
    
    d = list(sliceinfo['sNormal'].values()) if 'sNormal' in sliceinfo.keys() else [0]
    
    
    fov=[ReadFoV,PhaseFoV]
    if INFO_ONLY:
        return amplitude,d,fov,rot
    #print("here!")
    kspace = get_kspace(twix)
    image = do_image(kspace)

# reconstruct an image and show the result:
    if PLOT:
        plt.figure(figsize=[12,8])
        plt.subplot(121)
        plt.title('k-space')
        plt.imshow(abs(kspace[:,0])**0.2, cmap='gray', origin='lower')
        plt.axis('off')
        
        
        plt.subplot(122)
        plt.title('image')
        plt.imshow(abs(image), cmap='gray', origin='lower')
        plt.axis('off')


#twix[-1]['hdr']['Meas']['FrameOfReference']
   
    print('amplidute trasmitter: ' , amplitude)
    return kspace, image, amplitude
# %% 
def read_slice_info(filename):
    twix = twixtools.read_twix(filename)
    
    #data = twix[-1]['hdr']['MeasYaps']['sAAInitialOffset']['SliceInformation']
    data = twix[-1]['hdr']['MeasYaps']['sAAInitialOffset']['SliceInformation']
    return data

# %%
amplidude_list = []
d_list = []
fov_list = []
rot_list =[]
for filename in filelist:
    amplitude,d,fov,rot = read_file(filename)
    amplidude_list.append(amplitude)
    d_list.append(d)
    #fov_list.append(fov)
    rot_list.append(rot)

# twix0 = twixtools.read_twix(filelist[1])
# dict0 = twix0[-1]['hdr']['MeasYaps']['sSliceArray']['asSlice']
# twix1 = twixtools.read_twix(filelist[78])
# dict1 = twix1[-1]['hdr']['MeasYaps']['sSliceArray']['asSlice']

# slicepos0 = [list(i['sPosition'].values()) for i in dict0  ]
# slicepos1 = [list(i['sPosition'].values()) for i in dict1  ]
# #remove_empty_keys
# dict0 = {k: v for k, v in dict0.items() if v}
# dict1 = {k: v for k, v in dict1.items() if v}

# #flatten dictionary in dictionary
# dict0 = {k: str(v) for k, v in dict0.items() if isinstance(v, dict)}
# dict1 = {k: str(v) for k, v in dict1.items() if isinstance(v, dict)}

# set1 = set(dict0.items())
# set2 = set(dict1.items())
# diff = dict(set1 - set2)

from pandas import DataFrame

df = DataFrame({'file_name': filelist, 'amplitude': amplidude_list, "plane rotation":rot_list,"slice normal": d_list})

df.to_excel('test.xlsx', sheet_name='sheet1', index=False)
