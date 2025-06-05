# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:09:48 2022

@author: carlo
"""
import matplotlib.pyplot as plt
import numpy as np

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def violin_plot(ax, data, color_names, alpha_violin = 1, s_box = 20, s_ind = 20):
    
    
    points = [np.random.uniform(-0.2,0.2,len(data[x0])) + x0 for x0 in range(0,len(data))]
    np.random.seed(0)
    
    
    parts = plt.violinplot(data, positions = np.arange(0,len(points),1), widths = 0.8,
                            showmeans = False, showmedians = False, showextrema = False)
    for i,pc in enumerate(parts['bodies']):
        pc.set_facecolor(color_names[i])
        pc.set_edgecolor('none')
        pc.set_alpha(alpha_violin)
    
    
    quartile1 = np.zeros(len(data))
    medians = np.zeros(len(data))
    quartile3 = np.zeros(len(data))
    whiskersMin = np.zeros(len(data))
    whiskersMax = np.zeros(len(data))
    
    for i in range(0,len(data)):
        quartile1[i], medians[i], quartile3[i] = np.percentile(data[i], [25, 50, 75])
        whiskersMin[i], whiskersMax[i] = np.min(data[i]), np.max(data[i])
    
    
    
    inds = np.arange(0,len(data),1)
    plt.scatter(inds, medians, marker='o', color='white', s=s_box, zorder=3)
    plt.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    plt.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)
    
    for i in range(0, len(data)):
        plt.scatter(points[i], data[i], s = s_ind, c = 'black',  edgecolors = 'none', alpha = 0.6, zorder = 2)
        ax.bar(x=[i], height = [0,0], yerr = [(0,0), (0,0)], color = color_names[i])
    
    return(ax)