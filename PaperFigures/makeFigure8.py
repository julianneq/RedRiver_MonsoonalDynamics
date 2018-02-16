import numpy as np
from matplotlib import pyplot as plt
import seaborn.apionly as sns
import os

def makeFigure8(result1, result2, paramBounds):

    modelNames = ['ACCESS1','bcc-csm1','BNU-ESM','CanESM2','CCSM4','CESM1','CMCC','CNRM-CM5',\
        'CSIRO','GFDL','GISS','HadGEM2','inmcm4','IPSL','MIROC','MPI-ESM',\
        'MRI','NorESM1']#,'EC-EARTH','FGOALS']
    markers = ['o','v','^','<','>','8','s','p','+','*','h','H','x','D','d','|','_','.']#,\
        #'1','2']
        
    RCPs = ['rcp26','rcp45','rcp60','rcp85']
    colors = ['#fecc5c','#fd8d3c','#f03b20','#bd0026']
    
    models = [f for f in os.listdir('./../Multipliers') if f[-17::] == '_Qmultipliers.txt' \
           and f != 'Historical_Qmultipliers.txt']
    numModels = len(models)
    
    # load time series of multipliers for all climate models
    multipliers = np.zeros([numModels,2099-1976+1-30+1,3])
    for i in range(numModels):
        multipliers[i,:,:] = np.loadtxt('./../Multipliers/' + models[i],skiprows=1)
        
    # make rectangles for RCP legend
    boxes = []
    for i in range(len(RCPs)):
        boxes.append(plt.Rectangle((0,0), 1, 1, fc=colors[i], ec='none'))
    
    mu_min = 0.95
    mu_max = 1.05
    C1_min = 0.7
    C1_max = 1.5
    std_min = 0.7
    std_max = 1.5
    
    # find normalized values of mins and maxs
    y_min = []
    y_max = []
    x_min = (mu_min - paramBounds[0][0])/(paramBounds[0][1] - paramBounds[0][0])
    x_max = (mu_max - paramBounds[0][0])/(paramBounds[0][1] - paramBounds[0][0])
    y_min.append((C1_min - paramBounds[2][0])/(paramBounds[2][1] - paramBounds[2][0]))
    y_max.append((C1_max - paramBounds[2][0])/(paramBounds[2][1] - paramBounds[2][0]))
    y_min.append((std_min - paramBounds[1][0])/(paramBounds[1][1] - paramBounds[1][0]))
    y_max.append((std_max - paramBounds[1][0])/(paramBounds[1][1] - paramBounds[1][0]))
    
    # 6 panel w/ C1, std and mean multipliers
    sns.set()
    fig = plt.figure()
    indices = [4,49,94] # indices corresponding to decades 1980-2009, 2025-2054 and 2070-2099
    for n in range(2):
        for j in range(3):
            ax = fig.add_subplot(2,3,n*3+j+1)
            ax = ShadeBoundaries(ax, x_min, x_max, y_min[n], y_max[n], result1, result2, paramBounds, [mu_min, mu_max], n)
            for i in range(numModels):
                # find marker for model j
                for k, name in enumerate(modelNames):
                    if name in models[i]:
                        marker=markers[k]
                        label=name
                        
                if marker == '+' or marker == '|' or marker == '_' or marker == 'x':
                    linewidth=2
                else:
                    linewidth=1
                        
                for k, rcp in enumerate(RCPs):
                    if rcp in models[i]:
                        color = colors[k]
                        
                if n == 0:
                    ax.scatter(multipliers[i,indices[j],0],multipliers[i,indices[j],1],edgecolor=color,\
                        facecolor=color,marker=marker,label=label,s=100,linewidth=linewidth)
                    ax.set_ylim([C1_min,C1_max])
                else:
                    ax.scatter(multipliers[i,indices[j],0],multipliers[i,indices[j],2],edgecolor=color,\
                        facecolor=color,marker=marker,label=label,s=100,linewidth=linewidth)
                    ax.set_ylim([std_min,std_max])
                    
            
            plt.scatter(1,1,edgecolor='k',facecolor='w',s=100,linewidth=2,label='Base SOW')
                
            if j == 0:
                ax.tick_params(axis='y',labelsize=18)
                if n == 0:
                    ax.legend(boxes,['RCP 2.6','RCP 4.5','RCP 6.0','RCP 8.5'],\
                        loc='upper right',ncol=1, fontsize=22)
                    ax.set_ylabel(r'$m_{C_1}$', fontsize=26, rotation='horizontal')
                    ax.yaxis.set_label_coords(-0.2,0.5)
                else:
                    ax.set_ylabel(r'$m_{\sigma}$', fontsize=26, rotation='horizontal')
                    ax.yaxis.set_label_coords(-0.2,0.5)
            else:
                ax.tick_params(axis='y',labelleft='off')
                
            if n == 1:
                ax.tick_params(axis='x',labelsize=18)
                ax.set_xlabel(r'$m_{\mu}$',fontsize=26)
            else:
                ax.tick_params(axis='x',labelbottom='off')
                ax.set_title(str(1976+indices[j]) + '-' + str(1976+30+indices[j]-1),fontsize=22)
            
            ax.set_xlim([mu_min,mu_max])
        
    handles, labels = plt.gca().get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]    
    fig.subplots_adjust(bottom=0.25)
    fig.legend(handles, labels, loc='lower center',ncol=6,fontsize=22,frameon=True)
    fig.set_size_inches([24, 12.55])
    fig.savefig('Figure11.pdf')
    fig.clf()
    
    # make yr-by-yr figures for GIFs
    for j in range(2099-1976+1-30+1):
        fig = plt.figure()
        for n in range(2):
            ax = fig.add_subplot(1,2,n+1)
            ax = ShadeBoundaries(ax, x_min, x_max, y_min[n], y_max[n], result1, result2, paramBounds, [mu_min, mu_max], n)
            for i in range(numModels):
                # find marker for model j
                for k, name in enumerate(modelNames):
                    if name in models[i]:
                        marker=markers[k]
                        label=name
                        
                if marker == '+' or marker == '|' or marker == '_' or marker == 'x':
                    linewidth=2
                else:
                    linewidth=1
                        
                for k, rcp in enumerate(RCPs):
                    if rcp in models[i]:
                        color = colors[k]
                        
                if n == 0:
                    ax.scatter(multipliers[i,j,0],multipliers[i,j,1],edgecolor=color,\
                        facecolor=color,marker=marker,label=label,s=100,linewidth=2)
                    ax.set_ylim([C1_min,C1_max])
                else:
                    ax.scatter(multipliers[i,j,0],multipliers[i,j,2],edgecolor=color,\
                        facecolor=color,marker=marker,label=label,s=100,linewidth=2)
                    ax.set_ylim([std_min,std_max])
                    
            plt.scatter(1,1,edgecolor='k',facecolor='w',s=100,linewidth=2,label='Base SOW')
                
            if n == 0:
                ax.set_ylabel(r'$m_{C_1}$', fontsize=36)
            else:
                ax.legend(boxes,['RCP 2.6','RCP 4.5','RCP 6.0','RCP 8.5'],\
                    loc='upper right',ncol=1, fontsize=26,frameon=True)
                ax.set_ylabel(r'$m_{\sigma}$', fontsize=36)
            
            ax.tick_params(axis='both',labelsize=26)
            ax.set_xlabel(r'$m_{\mu}$',fontsize=36)
            #ax.set_title(str(1976+j) + '-' + str(1976+30+j-1),fontsize=22)
            ax.set_xlim([mu_min,mu_max])
            
        #handles, labels = plt.gca().get_legend_handles_labels()
        #labels, ids = np.unique(labels, return_index=True)
        #handles = [handles[i] for i in ids]    
        fig.subplots_adjust(bottom=0.3)
        #legend2 = fig.legend(handles, labels, loc='lower center',ncol=6, fontsize=26, \
        #    title='Model', frameon=True)
        #plt.setp(legend2.get_title(), fontsize=30)
        fig.suptitle(str(1976+j) + '-' + str(1976+30+j-1), fontsize=30)
        fig.set_size_inches([24, 12.55])
        fig.savefig('GIF_PNGs/Both/' + str(1976+j) + '-' + str(1976+30+j-1) + '_Q.png')
        fig.clf()
        
    return None
    
def ShadeBoundaries(ax, x_min, x_max, y_min, y_max, result1, result2, paramBounds, mu, n):
    
    x = np.array([x_min, x_max])
    y0 = np.array([y_min]*2)
    y3 = np.array([y_max]*2)
    if n == 1:
        y1 = np.array([(1/result2.params[2])*(-result2.params[0] - result2.params[1]*x[0] - result2.params[3]*0.5), \
            (1/result2.params[2])*(-result2.params[0] - result2.params[1]*x[1] - result2.params[3]*0.5)]) # left dividing line
        y2 = np.array([(1/result1.params[2])*(np.log(0.95/0.05) - result1.params[0] - result1.params[1]*x[0] - result1.params[3]*0.5), \
            (1/result1.params[2])*(np.log(0.95/0.05) - result1.params[0] - result1.params[1]*x[1] - result1.params[3]*0.5)]) # right dividing line
    else:
        y1 = np.array([(1/result2.params[3])*(-result2.params[0] - result2.params[1]*x[0] - result2.params[2]*0.5), \
            (1/result2.params[3])*(-result2.params[0] - result2.params[1]*x[1] - result2.params[2]*0.5)]) # left dividing line
        y2 = np.array([(1/result1.params[3])*(np.log(0.95/0.05) - result1.params[0] - result1.params[1]*x[0] - result1.params[2]*0.5), \
            (1/result1.params[3])*(np.log(0.95/0.05) - result1.params[0] - result1.params[1]*x[1] - result1.params[2]*0.5)]) # right dividing line
            
    C0 = y0*(paramBounds[2][1] - paramBounds[2][0]) + paramBounds[2][0]
    C1 = y1*(paramBounds[2][1] - paramBounds[2][0]) + paramBounds[2][0]
    C2 = y2*(paramBounds[2][1] - paramBounds[2][0]) + paramBounds[2][0]
    C3 = y3*(paramBounds[2][1] - paramBounds[2][0]) + paramBounds[2][0]
    
    if n==1:
        ax.fill_between(mu,C0,C1,facecolor='#1f78b4',edgecolor='none') # successes
        ax.fill_between(mu,C0,C2,facecolor='#1f78b4',edgecolor='none') # successes
        ax.fill_between(mu,C1,C3,facecolor='#fb9a99',edgecolor='none') # left failures
    else:
        ax.fill_between(mu,C0,C1,facecolor='#fb9a99',edgecolor='none') # left failures
        ax.fill_between(mu,C1,C2,facecolor='#1f78b4',edgecolor='none') # successes
    
    ax.fill_between(mu,C2,C3,facecolor='#fb9a99',edgecolor='none') # right failures    
    
    return ax
