import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'DejaVu Sans', 'serif': ['Computer Modern'], 'size':15})
rc('figure', facecolor='w')
import astropy.io.fits as fits
from astropy.io import ascii
import math, os

#optional dependencies
from distutils.spawn import find_executable
if find_executable('latex'): rc('text', usetex=True)
else: rc('text', usetex=False)

def linear_regression(x, y):
	"""
	Input:  x, y: 1D arrays
	Output: [m, c], [m_err, c_err]: slope and intercept best fit and error
	"""
	N = len(x)
	x, y = np.array(x), np.array(y)

	A = np.array([[np.sum(x**2), np.sum(x)], \
	              [np.sum(x), N]])
	a = np.array([np.sum(x*y), np.sum(y)])

	fit = np.dot(np.linalg.inv(A), a)

	sig_sq = np.sum(y - (fit[0]*x + fit[1]))**2/(N + 2)
	m_err = np.sqrt(N*sig_sq/(N*np.sum(x**2) - (np.sum(x))**2))
	c_err = np.sqrt(sig_sq*np.sum(x**2)/(N*np.sum(x**2) - (np.sum(x))**2))
	err = np.array([m_err, c_err])

	return fit, err


def emission(data, **kwargs):
    thres = kwargs.get('thres', 1)
    
    x, y = np.array(data[0]), np.array(data[1])
    Npix = len(x)
    
    med = np.median(y)
    std = np.std(y)
    cut = med + thres*std
    
    count_cut = []
    for i in range(Npix):
        if y[i] >= cut:
            count_cut.append(i)
    
    emission = []
    arr = []
    for i in range(len(count_cut)-1):
        if (count_cut[i+1] - count_cut[i]) == 1:
            arr.append(count_cut[i])
        else:
            arr.append(count_cut[i])
            if len(arr) > 5:
                emission.append(np.array(arr))
            arr = []
            
    plt.figure(figsize=[16,5])
    plt.step(x, y, color='k')
    plt.axhline(cut, color='r', label=r'$\tilde{x} + %s \sigma$'%(thres))
    plt.axhline(med, color='b', label=r'$\tilde{x}$')
    
    plt.xlabel('Pixel')
    plt.ylabel('Count (ADU)')
    plt.xlim(min(x), max(x))
    if 'ylim' in kwargs:
        plt.ylim(kwargs.get('ylim'))
    if 'title' in kwargs:
        plt.title(kwargs.get('title'), fontsize=20)
    plt.legend(loc='upper right')
    if 'save' in kwargs:
        plt.savefig(kwargs.get('save'))
    plt.show()
    
    return np.array(emission)


def centroid(data, feat_idx, **kwargs):
    
    x, y = np.array(data[0]), np.array(data[1])
    Npix = len(x)
    
    max_idx = []
    for feat in feat_idx:
        for idx in feat:
            if y[idx] == max(y[feat]):
                max_idx.append(idx)
                
    plt.figure(figsize=[16,5])
    plt.plot(x, y, color='k')
    for idx in max_idx:
        plt.axvline(x[idx], color='g')
    
    plt.xlabel('Pixel')
    plt.ylabel('Count (ADU)')
    if 'xlim' in kwargs:
        plt.xlim(kwargs.get('xlim'))
    else:
        plt.xlim(min(x), max(x))
    if 'ylim' in kwargs:
        plt.ylim(kwargs.get('ylim'))
    if 'title' in kwargs:
        plt.title(kwargs.get('title'), fontsize=20)
    if 'save' in kwargs:
        plt.savefig(kwargs.get('save'))
    plt.show()
                
    return np.array(max_idx)