import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

def makeFigure7abde(formulation, thresholds, paramBounds):

    # uncertain factors
    LHsamples = pd.read_csv('LHsamples.txt', sep=' ', \
        names=['mu','sigma','amp1','phi1','amp2','phi2','ag','aqua','other','Dshift','evap'])
        
    # normalize LHsamples
    for i, col in enumerate(LHsamples.columns):
        LHsamples[col] = (LHsamples[col] - paramBounds[i,0])/(paramBounds[i,1] - paramBounds[i,0])
    
    # points for trajectories along each parameter individually
    xpts = [[np.arange(0.95,1.075,0.025),np.arange(0.95,1.075,0.025),np.arange(0.5,1.75,0.25)],\
        [np.arange(0.95,1.075,0.025),np.arange(0.95,1.075,0.025),np.arange(0.5,1.75,0.25)],\
        [np.arange(0.95,1.075,0.025),np.arange(0.95,1.075,0.025),np.arange(0.5,1.75,0.25)]]
    ypts = [[np.arange(0.5,1.75,0.25),np.arange(0.5,1.75,0.25),np.arange(0.5,1.75,0.25)],\
        [np.arange(0.5,1.75,0.25),np.arange(0.5,1.75,0.25),np.arange(0.5,1.75,0.25)],\
        [np.arange(0.5,1.75,0.25),np.arange(0.5,5.125,1.125),np.arange(0.5,5.125,1.125)]]
        
    params = [[[0,2],[0,1],[1,2]],[[0,2],[0,1],[1,2]],[[0,6],[0,8],[6,8]]]
        
    # normalized pts
    norm_xpts = []
    norm_ypts = []
    for j in range(3):
        for k in range(3):
            norm_xpts.append((xpts[j][k] - paramBounds[params[j][k][0],0]) / (paramBounds[params[j][k][0],1] - paramBounds[params[j][k][0],0]))
            norm_ypts.append((ypts[j][k] - paramBounds[params[j][k][1],0]) / (paramBounds[params[j][k][1],1] - paramBounds[params[j][k][1],0]))
        
    # color maps for plotting
    dot_cmap = mpl.colors.ListedColormap(np.array([[227,26,28],[166,206,227]])/255.0)
    class_cmap = mpl.colors.ListedColormap(np.array([[251,154,153],[31,120,180]])/255.0)
    contour_cmap = mpl.cm.get_cmap('RdBu')
    
    #############################################################################################
    # fit logistic regression to flood successes of best flood solution
    dta = pd.read_csv('../MORDMobjs/MORDM_new_WP1_thinned_Soln' + str(formulation.bestFloodSoln.solnNo) + '.obj',\
        sep=' ', names=['Hydro','Def2','AvgDef','MaxDef','Flood'])
    dta = pd.concat([LHsamples,dta],axis=1)
    predictors = dta.columns.tolist()[0:3]
    result1 = fitLogit(dta, dta.Flood, thresholds[0], predictors)
    
    # find sample points for C1 and Std in direction of greatest variability
    sampleX1, sampleX2, sampleX3 = getSamplePts(result1, [0.5, 0.5, 0.5])
        
    # grid and dividing line
    xgrid = np.arange(-0.1,1.1,0.01)
    ygrid = np.arange(-0.1,1.1,0.01)
    class_levels = [0.0, 0.95, 1.0]
    contour_levels = np.arange(0.0,1.05,0.1)
    
    sns.set()
    fig = plt.figure()

    # plot dot plots and 95% probability contour
    ax = fig.add_subplot(2,2,1)
    plotContourMap(True, ax, result1, 'x2', LHsamples, dta, class_cmap, dot_cmap, \
        xgrid, ygrid, class_levels, norm_xpts[1], norm_ypts[1], sampleX1, sampleX3, \
        'mu', 'amp1', r'$m_{\mu}$', r'$m_{C_1}$', np.arange(0.1,0.95,0.2), np.arange(0.1,0.95,0.2), \
        ['0.96','0.98','1.00','1.02','1.04'], ['0.6','0.8','1.0','1.2','1.4'], [0.5,0.5,0.5])
    ax = fig.add_subplot(2,2,3)
    plotContourMap(True, ax, result1, 'x3', LHsamples, dta, class_cmap, dot_cmap, \
        xgrid, ygrid, class_levels, norm_xpts[0], norm_ypts[0], sampleX1, sampleX2, \
        'mu', 'sigma', r'$m_{\mu}$', r'$m_{\sigma}$', np.arange(0.1,0.95,0.2), np.arange(0.1,0.95,0.2), \
        ['0.96','0.98','1.00','1.02','1.04'], ['0.6','0.8','1.0','1.2','1.4'], [0.5,0.5,0.5])
        
    # plot probability contours of logistic regression
    ax = fig.add_subplot(2,2,2)
    contourset = plotContourMap(False, ax, result1, 'x2', LHsamples, dta, contour_cmap, dot_cmap, \
        xgrid, ygrid, contour_levels, norm_xpts[1], norm_ypts[1], sampleX1, sampleX3, \
        'mu', 'amp1', r'$m_{\mu}$', r'$m_{C_1}$', np.arange(0.1,0.95,0.2), np.arange(0.1,0.95,0.2), \
        ['0.96','0.98','1.00','1.02','1.04'], ['0.6','0.8','1.0','1.2','1.4'], [0.5,0.5,0.5])
    ax = fig.add_subplot(2,2,4)
    plotContourMap(False, ax, result1, 'x3', LHsamples, dta, contour_cmap, dot_cmap, \
        xgrid, ygrid, contour_levels, norm_xpts[0], norm_ypts[0], sampleX1, sampleX2, \
        'mu', 'sigma', r'$m_{\mu}$', r'$m_{\sigma}$', np.arange(0.1,0.95,0.2), np.arange(0.1,0.95,0.2), \
        ['0.96','0.98','1.00','1.02','1.04'], ['0.6','0.8','1.0','1.2','1.4'], [0.5,0.5,0.5])
    
    fig.subplots_adjust(wspace=0.3,hspace=0.3,right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(contourset, cax=cbar_ax)
    cbar_ax.set_ylabel('Probability of Success',fontsize=20)
    yticklabels = cbar.ax.get_yticklabels()
    cbar.ax.set_yticklabels(yticklabels,fontsize=18)
    fig.set_size_inches([12.67, 10.61])
    fig.savefig('Figure6abde.pdf')
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
    
def getSamplePts(result, base):
    # parametric equations of line perpendicular to dividing line [b0 + b1x1 + b2x2 + b3x3 = ln(p/(1-p))]
    # and passing through (0.5,0.5,0.5):
    # x1 = 1 + b1*t
    # x2 = 1 + b2*t
    # x3 = 1 + b3*t

    # minimum number of slope units above (1,1,1) to hit max x1, max x2 or max x3
    t1 = np.abs((1.0 - base[0])/result.params[1])
    t2 = np.abs((1.0 - base[1])/result.params[2])
    t3 = np.abs((1.0 - base[2])/result.params[3])
    t = np.min([t1, t2, t3])
    
    x1s = []
    x2s = []
    x3s = []
    for i in range(2):
        x1s.extend([base[0] + result.params[1]*t*(i/2 - (i-1))])
        x2s.extend([base[1] + result.params[2]*t*(i/2 - (i-1))])
        x3s.extend([base[2] + result.params[3]*t*(i/2 - (i-1))])
        
    x1s.extend([base[0]])
    x2s.extend([base[1]])
    x3s.extend([base[2]])
    
    # minimum number of slope units below (1,1,1) to hit min x1, max x2 or max x3
    t1 = np.abs(-base[0]/result.params[1])
    t2 = np.abs(-base[1]/result.params[2])
    t3 = np.abs(-base[2]/result.params[3])
    t = np.min([t1, t2, t3])
    
    for i in range(2):
        x1s.extend([base[0] - result.params[1]*t*(0.5+i/2)])
        x2s.extend([base[1] - result.params[2]*t*(0.5+i/2)])
        x3s.extend([base[2] - result.params[3]*t*(0.5+i/2)])
    
    return x1s, x2s, x3s
    
def plotContourMap(dots, ax, result, constant, LHsamples, dta, contour_cmap, dot_cmap, xgrid, ygrid, levels, \
    xpts, ypts, sampleXs, sampleYs, xvar, yvar, xlabel, ylabel, xticks, yticks, xticklabels, yticklabels, base):
    
    # find probability of success for x=xgrid, y1var=ygrid and y2var=1
    X, Y = np.meshgrid(xgrid, ygrid)
    x = X.flatten()
    y = Y.flatten()
    if constant == 'x3': # 3rd predictor held constant at base value
        grid = np.column_stack([np.ones(len(x)),x,y,np.ones(len(x))*base[2]])
    elif constant == 'x2': # 2nd predictor held constant at base value
        grid = np.column_stack([np.ones(len(x)),x,np.ones(len(x))*base[2],y])
    else: # 1st predictor held constant at base value
        grid = np.column_stack([np.ones(len(x)),np.ones(len(x))*base[2],x,y])
        
    z = result.predict(grid)
    Z = np.reshape(z, np.shape(X))
            
    contourset = ax.contourf(X, Y, Z, levels, cmap=contour_cmap)
    if dots == True:
        ax.scatter(LHsamples[xvar].values,LHsamples[yvar].values, \
            facecolor=dta['Success'].values,edgecolor='none',cmap=dot_cmap)
        
    ax.set_xlim(np.min(X),np.max(X))
    ax.set_ylim(np.min(Y),np.max(Y))
    ax.set_xlabel(xlabel,fontsize=24)
    ax.set_ylabel(ylabel,fontsize=24)
    ax.tick_params(axis='both',labelsize=18)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    
    if dots == True:
        # plot point varying only one dimension
        ax.scatter([base[0]]*len(ypts), ypts, c='k', marker='o', s=50)
        ax.scatter(xpts, [base[1]]*len(xpts), c='k', marker='o', s=50)
        
        # plot points in direction of greatest variability    
        ax.scatter(sampleXs, sampleYs, marker='o', c='k', s=50)
    
    return contourset
