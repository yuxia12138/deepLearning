import sys

tools = sys.modules[__name__]

import numpy as np
import torch
from PIL import Image
from IPython import display
from matplotlib import pyplot as plt



def use_svg_display():
    '''use svg style'''
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    '''set figure size'''
    use_svg_display()
    tools.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    '''set axes'''
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
        fmts=('-','m--','g-.','r:'), figsize=(3.5, 2.5), axes=None):

        if legend is None:
            legend = []
        set_figsize(figsize)
        axes = axes if axes else tools.plt.gca()

        '''
        if X only have one axis ,return True
        '''
        def has_one_axis(X):
            return (hasattr(X, 'ndim') and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], '__len__'))
        
        if has_one_axis(X):
            X = [X]
        if Y is None:
            X, Y = [[]] * len(X), X
        elif has_one_axis(Y):
            Y = [Y]
        if len(X) != len(Y):
            X = X * len(Y)
        axes.cla()

        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)
        set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

            
