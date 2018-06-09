import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def makeFigure9(formulation, thresholds, paramBounds):

    # uncertain factors
    LHsamples = pd.read_csv('LHsamples.txt', sep=' ', \
        names=['mu','sigma','amp1','phi1','amp2','phi2','ag','aqua','other','Dshift','evap'])
        
    # normalize LHsamples
    for i, col in enumerate(LHsamples.columns):
        LHsamples[col] = (LHsamples[col] - paramBounds[i,0])/(paramBounds[i,1] - paramBounds[i,0])
        
    # color maps for plotting
    dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)
    class_cmap = mpl.colors.ListedColormap(np.array([[251,154,153],[31,120,180]])/255.0)
    
    #############################################################################################       
    # fit logistic regression to hydro successes of best hydro solution
    dta = pd.read_csv('../MORDMobjs/MORDM_new_WP1_thinned_Soln' + str(formulation.bestHydroSoln.solnNo) + '.obj',\
        sep=' ', names=['Hydro','Def2','AvgDef','MaxDef','Flood'])
    dta = pd.concat([LHsamples,dta],axis=1)
    predictors = dta.columns.tolist()[0:3]
    result2 = fitLogit(dta, dta.Hydro, thresholds[1], predictors)

    # grid and dividing line
    xgrid = np.arange(-0.1,1.1,0.01)
    ygrid = np.arange(-0.1,1.1,0.01)
    levels = [0.0, 0.5, 1.0]
    
    # make figure of hydro dot plots
    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(2,2,1)
    plotContourMap(ax, result2, 'x3', LHsamples, dta, class_cmap, dot_cmap, xgrid, ygrid, levels, \
        'mu', 'sigma', r'$m_{\mu}$', r'$m_{\sigma}$', np.arange(0.1,0.95,0.2), np.arange(0.1,0.95,0.2), \
        ['0.96','0.98','1.00','1.02','1.04'], ['0.6','0.8','1.0','1.2','1.4'], [0.5,0.5,0.5], False)
    ax = fig.add_subplot(2,2,2)
    plotContourMap(ax, result2, 'x2', LHsamples, dta, class_cmap, dot_cmap, xgrid, ygrid, levels, \
        'mu', 'amp1', r'$m_{\mu}$', r'$m_{C_1}$', np.arange(0.1,0.95,0.2), np.arange(0.1,0.95,0.2), \
        ['0.96','0.98','1.00','1.02','1.04'], ['0.6','0.8','1.0','1.2','1.4'], [0.5,0.5,0.5], False)
        

    # fit logistic regression to deficit successes of best deficit solution
    dta = pd.read_csv('../MORDMobjs/MORDM_new_WP1_thinned_Soln' + str(formulation.bestDefSoln.solnNo) + '.obj',\
        sep=' ', names=['Hydro','Def2','AvgDef','MaxDef','Flood'])
    dta = pd.concat([LHsamples,dta],axis=1)
    predictors = dta.columns.tolist()[0:1] + dta.columns.tolist()[6:7] + dta.columns.tolist()[8:9]
    result3 = fitLogit(dta, dta.MaxDef, thresholds[2], predictors)
    
    # dividing line
    levels = [0.0, 0.75, 1.0]

    # add deficit dot plots to figure
    ax = fig.add_subplot(2,2,3)
    plotContourMap(ax, result3, 'x3', LHsamples, dta, class_cmap, dot_cmap, xgrid, ygrid, levels, \
        'mu', 'ag', r'$m_{\mu}$', r'$m_{ag}$', np.arange(0.1,0.95,0.2), np.arange(0.1,0.95,0.2), \
        ['0.96','0.98','1.00','1.02','1.04'], ['0.6','0.8','1.0','1.2','1.4'], [0.5,0.5,0.5], False)
    ax = fig.add_subplot(2,2,4)
    plotContourMap(ax, result3, 'x2', LHsamples, dta, class_cmap, dot_cmap, xgrid, ygrid, levels, \
        'ag', 'other', r'$m_{ag}$', r'$m_{o}$', np.arange(0.1,0.95,0.2), np.arange(0.25/4.5,4.3/4.5,1/4.5), \
        ['0.6','0.8','1.0','1.2','1.4'], ['0.75','1.75','2.75','3.75','4.75'], [0.5,0.5,0.5], True)
    
    fig.subplots_adjust(wspace=0.3)
    fig.set_size_inches([14.5, 12.3])
    fig.savefig('Figure7.pdf')
    fig.clf()
        
    return None
    
def fitLogit(dta, variable, threshold, predictors):
    # define successes (1) and failures (0)
    dta['Success'] = (variable < threshold).astype(int)
    # concatenate intercept column of 1s
    dta['Intercept'] = np.ones(np.shape(dta)[0])
    # get columns of predictors
    cols = dta.columns.tolist()[-1:] + predictors
    #fit logistic regression
    logit = sm.Logit(dta['Success'], dta[cols])
    result = logit.fit()
    
    return result
    
def plotContourMap(ax, result, constant, LHsamples, dta, contour_cmap, dot_cmap, xgrid, ygrid, levels, \
    xvar, yvar, xlabel, ylabel, xticks, yticks, xticklabels, yticklabels, base, tranpose):
    
    # find probability of success for x=xgrid, y1var=ygrid and y2var=1
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()
    if constant == 'x3': # 3rd predictor held constant at base value
        grid = np.column_stack([np.ones(len(x)),x,y,np.ones(len(x))*base[2]])
    elif constant == 'x2': # 2nd predictor held constant at base value
        grid = np.column_stack([np.ones(len(x)),x,np.ones(len(x))*base[1],y])
    else: # 1st predictor held constant at base value
        grid = np.column_stack([np.ones(len(x)),np.ones(len(x))*base[0],x,y])
        
    z = result.predict(grid)
    Z = np.reshape(z, np.shape(X))
            
    if tranpose != True:
        contourset = ax.contourf(X, Y, Z, levels, cmap=contour_cmap)
        ax.scatter(LHsamples[xvar].values,LHsamples[yvar].values, \
            c=dta['Success'].values,edgecolor='none',cmap=dot_cmap)
        ax.set_xlim(np.min(X),np.max(X))
        ax.set_ylim(np.min(Y),np.max(Y))
        ax.set_xlabel(xlabel,fontsize=24)
        ax.set_ylabel(ylabel,fontsize=24)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
    else:
        contourset = ax.contourf(Y, X, Z, levels, cmap=contour_cmap)
        ax.scatter(LHsamples[yvar].values,LHsamples[xvar].values, \
            c=dta['Success'].values,edgecolor='none',cmap=dot_cmap)
        ax.set_xlim(np.min(Y),np.max(Y))
        ax.set_ylim(np.min(X),np.max(X))
        ax.set_xlabel(ylabel,fontsize=24)
        ax.set_ylabel(xlabel,fontsize=24)
        ax.set_xticks(yticks)
        ax.set_xticklabels(yticklabels)
        ax.set_yticks(xticks)
        ax.set_yticklabels(xticklabels)
        
    ax.tick_params(axis='both',labelsize=18)
    
    return contourset
