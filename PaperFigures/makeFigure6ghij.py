from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import seaborn.apionly as sns
        
class Solution:
    def __init__(self):
        self.HanoiLev = None
        self.Hydro = None
        self.Deficit = None
        
def makeFigure6ghij(formulation, LHsamples, paramBounds, normSamples, thresholds):
    # scenarios for event plots
    scenarios = ['Mean','C1','Std','All3_Flood']
    titles = ['SOW Trajectory 1','SOW Trajectory 2','SOW Trajectory 3','SOW Trajectory 4']
    returnPds = 100.0
    pctiles = 1/returnPds
    IDs = pctiles*1000
    colors = ['#e31a1c','#fb9a99','#f7f7f7','#a6cee3','#1f78b4']
    ylabel = 'Water Level (m)'
    solnNo = formulation.bestFloodSoln.solnNo
    ymax = 18.0
    
    sns.set()
    fig = plt.figure()
    # plot 100-yr event for each pathway through the SOW space
    for k in range(4): # 4 trajectories
        # load simulations from most robust solution across scenarios and find year of 100-yr event
        ax = fig.add_subplot(1,4,k+1)
        
        for i in range(5): # 5 pts along trajectory
            soln = getSoln(solnNo, scenarios[k], i+1)
            yvalues = soln.HanoiLev
            maxFloods = np.max(soln.HanoiLev,1)
            year = np.argsort(maxFloods)[::-1][int(IDs)-1]
            yvalues = yvalues[year,:]
                
            ax.plot(range(0,365),yvalues,c=colors[-(1+i)],linewidth=2)
            ax.fill_between(range(0,365),yvalues,color=colors[-(1+i)],zorder=5-i)
                
        ax.plot([0,364],[13.4,13.4],c='k',linewidth=2) # dike height

        ax.set_xlim([0,364])
        ax.set_ylim([0,ymax])
        ax.set_xticks([45,137,229,319])
        ax.set_xticklabels(['Jun','Sep','Dec','Mar'],fontsize=18)
            
        if k == 0:
            ax.set_ylabel(ylabel,fontsize=22)
            ax.tick_params(axis='y',labelsize=18)
        else:
            ax.tick_params(axis='y',labelleft='off')
            
        ax.set_title(titles[k], fontsize=22)
                
    fig.suptitle('Water level time series during 100-yr flood with most robust solution for flooding',fontsize=22)
    fig.set_size_inches([26.4, 6.1])
    fig.savefig('Figure6ghij.pdf')
    fig.clf()
            
    return None
    
def getSoln(solnNo, scenario, point):
    soln = Solution()
    soln.HanoiLev, soln.Hydro, soln.Deficit = resortSimulations('./../Simulations/new_WP1_thinned_proc' \
        + str(solnNo) + '_' + scenario + 'Trajectory_Pt' + str(point) + '.obj')
    
    return soln

def resortSimulations(filename):
    rawSims = np.loadtxt(filename)
    rawSims = np.transpose(np.reshape(rawSims,[1000,3*365]))
    HanoiLev = np.zeros([1000,365])
    Hydro = np.zeros([1000,365])
    Deficit = np.zeros([1000,365])
    for i in range(np.shape(HanoiLev)[1]):
        HanoiLev[:,i] = rawSims[3*i,:]
        Hydro[:,i] = rawSims[3*i+1,:]
        Deficit[:,i] = rawSims[3*i+2,:]
        
    Deficit = np.where(Deficit>0, Deficit, 0) # remove negative deficits (there shouldn't be any)
    
    return (HanoiLev, Hydro, Deficit)