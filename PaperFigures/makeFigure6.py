import os
import numpy as np
from robustnessMeasures import SatisfyTypeI
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe
import matplotlib
import pandas
import seaborn.apionly as sns

class Formulation:
    def __init__(self):
        self.name = None
        self.optim = None
        self.reeval_1000 = None
        self.MORDMobjs = None
        self.Satisfy1 = None
        self.bestIndices = None
        
def makeFigure6():
    new_WP1 = calcRobustness('new_WP1')
    
    cbar = 'inferno'
    labels = [r'$J_{Flood}$' + '\n(m above 11.25 m)',\
        r'$J_{Hydro}$' + '\n(Gwh/day)',\
        r'$J_{Max Def}$' + '\n' + r'$\mathregular{(m^3/s)}$']
    precision = [2,0,0]
    cbar_axes = [0.85, 0.15, 0.05, 0.7]
        
    sns.set_style("darkgrid")
    fig = plt.figure()
    ax = fig.add_subplot(111)      

    # create newlabels so they aren't appended to labels each time
    newlabels = []
    
    table = pandas.DataFrame(new_WP1.reeval_1000,columns=labels)
    mins = np.min(new_WP1.reeval_1000,0)
    maxs = np.max(new_WP1.reeval_1000,0)
    # round number of significant digits shown on objective labels
    for k in range(len(labels)):
        if precision[k] != 0:
            newlabels.append(str(np.round(mins[k],precision[k])) + '\n' + labels[k])
        else:
            newlabels.append(str(int(mins[k]))+ '\n' + labels[k])
        
        # don't show negative sign on maximization objectives
        if mins[k] < 0:
            newlabels[k] = newlabels[k][1:]
        
    parallel_coordinate(fig, ax, table, new_WP1.Satisfy1[:,4], mins, maxs, cbar,\
        newlabels, precision, cbar_axes, new_WP1.bestIndices)
        
    fig.set_size_inches([8.725, 7.7375])
    fig.savefig('Figure5.pdf')
    fig.clf()
    
    return None
        
def calcRobustness(name):
    formulation = getFormulations(name, 1000, 5)
    thresholds = np.array([2.15, -25, 350])
    
    # calculate satisficing Type I metric
    formulation.Satisfy1 = SatisfyTypeI(formulation.MORDMobjs, thresholds)
        
    # write output to file
    np.savetxt(formulation.name + '_Satisfy1.txt', formulation.Satisfy1, delimiter = ' ')
    
    # calculate indices of most robust solutions on each objective individually, and all together
    formulation.bestIndices = []
    formulation.bestIndices.append(np.argmax(formulation.Satisfy1[:,0])) # most robust for flooding
    formulation.bestIndices.append(np.argmax(formulation.Satisfy1[:,1])) # most robust for hydropower
    formulation.bestIndices.append(np.argmax(formulation.Satisfy1[:,2])) # most robust for deficit
    formulation.bestIndices.append(np.argmax(formulation.Satisfy1[:,4])) # most robust for all requirements
    
    return formulation

def reformatData(name, nSamples, nObjs, colOrder):
    numPts = len(os.listdir('./../MORDMobjs'))
    objs = np.zeros([numPts, nSamples, nObjs])
    
    for i in range(numPts):
        objs[i,:,:] = np.loadtxt('./../MORDMobjs/MORDM_' + name + '_thinned_Soln' + str(int(i+1)) + '.obj')
    
    # re-order objectives to be WP1 Flood Vul, WP1 Hydro, WP1 Max Def
    temp_objs = objs[:,:,colOrder]
    objs = temp_objs

    return objs
    
def getFormulations(name, nsamples, nobjs):
    formulation = Formulation()
    formulation.name = name
    
    formulation.reeval_1000 = np.loadtxt('./../' + name + '_thinned_re-eval_1000.csv',skiprows=1,delimiter=',')
    
    colOrder = [4, 0, 3] # flood obj, hydro obj, max deficit obj
    temp_objs = formulation.reeval_1000[:,colOrder]
    formulation.reeval_1000 = temp_objs

    formulation.MORDMobjs = reformatData(name, nsamples, nobjs, colOrder)
    
    formulation.optim = np.loadtxt('./../' + name + '_thinned.csv',skiprows=1,delimiter=',')
    colOrder = [2, 0, 1] # flood obj, hydro obj, deficit obj
    temp_optim = formulation.optim[:,colOrder]
    formulation.optim = temp_optim
    
    # replace re-evaluated objective values with values from optimization if that objective optimized
    formulation.reeval_1000[:,0] = formulation.optim[:,0]
    formulation.reeval_1000[:,1] = formulation.optim[:,1]
    
    return formulation
    
def parallel_coordinate(fig, ax1, table, shade, mins, maxs, cbar, \
    xlabels, precision, cbar_axes, indices):
      
    newShade = np.copy(shade)
    minShade = np.min(shade)
    maxShade = np.max(shade)
    for i in range(len(shade)):
        newShade[i] = (shade[i]-minShade)/(maxShade-minShade)
    
    toplabels = []
    # round number of significant digits shown on objective labels
    for i in range(len(xlabels)):
        if precision[i] != 0:
            toplabels.append(str(np.round(maxs[i],precision[i])))
        else:
            toplabels.append(str(int(maxs[i])))
        if maxs[i] < 0:
            # don't show negative sign on maximization objectives
            toplabels[i] = toplabels[i][1:]
        
    cmap = matplotlib.cm.get_cmap(cbar)
    scaled = table.copy()
    index = 0
    for column in table.columns:
        scaled[column] = (table[column] - mins[index]) / (maxs[index] - mins[index])
        index = index + 1
    
    index = 0
    for k, solution in enumerate(scaled.iterrows()):
        ys = solution[1]
        xs = range(len(ys))
        if k not in indices:
            ax1.plot(xs, ys, c=cmap(newShade[index]), linewidth=2, alpha=0.5)
        else:
            # make line for most robust solution thicker and opaque
            ax1.plot(xs, ys, c=cmap(newShade[index]), linewidth=2, \
                path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()], \
                    zorder=np.shape(table)[0])
            
        index = index + 1
    
    ax1.set_xticks(np.arange(0,np.shape(table)[1],1))
    ax1.set_xlim([0,np.shape(table)[1]-1])
    ax1.set_ylim([0,1])
    ax1.set_xticklabels(xlabels,fontsize=14)
    ax1.tick_params(axis='y',which='both',labelleft='off',left='off',right='off')
    ax1.tick_params(axis='x',which='both',top='off',bottom='off')
    
    sm = matplotlib.cm.ScalarMappable(cmap=cmap)
    sm.set_array([minShade,maxShade])
    cbar = fig.colorbar(sm, ax=ax1, orientation='horizontal', ticks=np.arange(0.10,0.35,0.05))
    cbar.ax.tick_params(axis='x',labelsize=14)
    cbar.ax.set_xticklabels(['10','15','20','25','30'])
    fig.axes[-1].set_xlabel('Percent of SOWs in which all criteria are met',fontsize=16)
    
    # make subplot frames invisible
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    
    # draw in axes
    for i in np.arange(0,np.shape(table)[1],1):
        ax1.plot([i,i],[0,1],c='k')
    
    # create twin y axis to put x tick labels on top
    ax2 = ax1.twiny()
    ax2.set_xticks(np.arange(0,np.shape(table)[1],1))
    ax2.set_xlim([0,np.shape(table)[1]-1])
    ax2.set_ylim([0,1])
    ax2.set_xticklabels(toplabels,fontsize=14)
    ax2.tick_params(axis='y',which='both',labelleft='off',left='off',right='off')
    ax2.tick_params(axis='x',which='both',top='off',bottom='off')
    
    # make subplot frames invisible
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    
    return ax1, fig
    
makeFigure5()
