import numpy as np
import os
from matplotlib import pyplot as plt
import matplotlib
import seaborn.apionly as sns

# plot "CDFs" of each metric for each solution across sampled SOWs
class Formulation:
    def __init__(self):
        self.name = None
        self.MORDMobjs = None
        self.baseObjs = None
        self.normObjs = None

def makeFigureS1():
    nsamples = 1000
    nobjs = 5
    
    new_WP1 = getFormulations('new_WP1_thinned', nsamples, nobjs)
    colOrder = [0, 3, 4] # hydro obj, max deficit obj, flood obj
    figOrder = [3, 4, 2]
    thresholds = np.array([-25, 350.0, 2.15])
    objs = [r'$J_{Hydro}$' + ' (Gwh/day)', r'$J_{Max Def}$' + ' ' + r'$\mathregular{(m^3/s)}$', r'$J_{Flood}$' + ' (m above 11.25 m)']
    cmap = matplotlib.cm.get_cmap('Blues_r')

    p = np.zeros([nsamples])
    for i in range(nsamples):
        p[i] = 100*(i+1.0)/(nsamples+1.0)
        
    sns.set_style("darkgrid")
    fig = plt.figure()
    for i in range(len(objs)):
        ax = fig.add_subplot(2,2,figOrder[i])
        for k in range(np.shape(new_WP1.MORDMobjs)[0]):
            #if new_WP1.normObjs[k,0] == np.min(new_WP1.normObjs[:,0]):
            x = np.sort(new_WP1.MORDMobjs[k,:,colOrder[i]])
            l1, = ax.step(x,p,c=cmap(new_WP1.normObjs[k,0]))
            
        ax.plot([thresholds[i],thresholds[i]],[0,100],c='k',linewidth=2)
        ax.set_xlabel(objs[i],fontsize=18)
        ax.tick_params(axis='both',labelsize=14)
        
        if i == 0:
            ax.set_xticks(np.arange(-70,10,20))
            ax.set_xticklabels(-np.arange(-70,10,20))
            
        if i == 1:
            ax.set_xticks(np.arange(0,500,100))
            ax.set_xticklabels(np.arange(0,500,100))
            ax.set_xlim([0,450])
            
        ax.set_ylabel('Cumulative Percent\n of Sampled SOWs',fontsize=18)
    
    ax = fig.add_subplot(2,2,1)
    ax.scatter(new_WP1.baseObjs[:,0],new_WP1.baseObjs[:,2],s=200*(new_WP1.normObjs[:,1]+0.05),\
        edgecolor=cmap(new_WP1.normObjs[:,0]),facecolor=cmap(new_WP1.normObjs[:,0]))
    pt1 = ax.scatter([],[],s=200*0.05,facecolor='#6baed6',edgecolor='#6baed6')
    pt2 = ax.scatter([],[],s=200*1.05,facecolor='#6baed6',edgecolor='#6baed6')
    
    ax.set_yticks(np.arange(0,3,0.5))
    ax.set_xticks(np.arange(-46,-22,4))
    ax.set_xticklabels(-np.arange(-46,-22,4))
    ax.set_xlim([-46,-26])
    ax.tick_params(axis='both',labelsize=14)
    ax.set_xlabel(r'$J_{Hydro}$' + ' (Gwh/day)',fontsize=18)
    ax.set_ylabel(r'$J_{Flood}$' + ' (m above 11.25 m)',fontsize=18)
    ax.set_title('Performance in Base SOW',fontsize=18)
    legend = ax.legend([pt1, pt2], [str(int(np.round(np.min(new_WP1.baseObjs[:,1]),0))), \
        str(int(np.round(np.max(new_WP1.baseObjs[:,1]),0)))], scatterpoints=1, \
        title=r'$J_{Deficit^2}$'+ ' ' + r'$\mathregular{(m^3/s)^2}$', \
        fontsize=18, loc='upper right', frameon=True)
    plt.setp(legend.get_title(), fontsize=18)
    
    fig.subplots_adjust(bottom=0.25, hspace=0.3, wspace=0.3)
    sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap('Blues'))
    sm.set_array([np.min(-new_WP1.baseObjs[:,0]),np.max(-new_WP1.baseObjs[:,0])])
    cbar_ax = fig.add_axes([0.1,0.1,0.8,0.05])
    cbar = fig.colorbar(sm, cax=cbar_ax,orientation='horizontal')
    cbar.ax.tick_params(labelsize=14)
    fig.axes[-1].set_xlabel(r'$J_{Hydro}$' + ' (Gwh/day) in Base SOW',fontsize=18)
    
    fig.set_size_inches([14.775, 10.0875])
    fig.savefig('FigureS1.pdf')
    fig.clf()
    
    return None

def reformatData(name, nSamples, nObjs):
    numPts = len(os.listdir('./../MORDMobjs/'))
    objs = np.zeros([numPts, nSamples, nObjs])
    
    for i in range(numPts):
        objs[i,:,:] = np.loadtxt('./../MORDMobjs/MORDM_' + name + '_Soln' + str(int(i+1)) + '.obj')

    return objs
    
def getFormulations(name, nsamples, nobjs):
    formulation = Formulation()
    formulation.name = name
    formulation.MORDMobjs = reformatData(name, nsamples, nobjs)
    formulation.baseObjs = np.loadtxt('./../' + name + '.csv',skiprows=1,delimiter=',')
    formulation.normObjs = np.zeros(np.shape(formulation.baseObjs))
    for i in range(np.shape( formulation.normObjs)[0]):
         formulation.normObjs[i,:] = (formulation.baseObjs[i,:] - np.min(formulation.baseObjs,0)) / \
             (np.max(formulation.baseObjs,0) - np.min(formulation.baseObjs,0))
        
    return formulation
    
makeFigureS1()
