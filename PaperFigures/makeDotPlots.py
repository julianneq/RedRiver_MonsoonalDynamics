from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.api as sm
from makeFigure6abde import makeFigure6abde
from makeFigure6ghij import makeFigure6ghij
from makeFigure7 import makeFigure7
from makeFigure8 import makeFigure8

class Soln:
    def __init__(self):
        self.solnNo = None
        self.objs = None

class Formulation:
    def __init__(self):
        self.name = None
        self.Satisfy1 = None
        self.bestFloodSoln = Soln()
        self.bestHydroSoln = Soln()
        self.bestDefSoln = Soln()
        self.mostRobustSoln = Soln()

def makeDotPlots():
    # load formulation and LHsamples
    new_WP1 = getFormulations('new_WP1')
    LHsamples = np.loadtxt('LHsamples.txt')
    paramBounds = np.loadtxt('uncertain_params.txt',usecols=[1,2])
    
    # normalize samples on [0,1] for consistent dimensions across objectives
    normSamples = np.zeros(np.shape(LHsamples))
    for i in range(np.shape(normSamples)[1]):
        normSamples[:,i] = (LHsamples[:,i] - paramBounds[i,0])/(paramBounds[i,1] - paramBounds[i,0])
    
    # specify primary parameters of importance for each objective
    thresholds = [2.15, -25, 350]
    
    # make Figures 6 and 7
    makeFigure6abde(new_WP1, thresholds, paramBounds)
    makeFigure6ghij(new_WP1, LHsamples, paramBounds, normSamples, thresholds)
    makeFigure7(new_WP1, thresholds, paramBounds)
    
    # find boundary equations for compromise solution
    LHsamples_df = pd.read_csv('LHsamples.txt', sep=' ', \
        names=['mu','sigma','amp1','phi1','amp2','phi2','ag','aqua','other','Dshift','evap'])
    for i, col in enumerate(LHsamples_df.columns):
        LHsamples_df[col] = (LHsamples_df[col] - paramBounds[i,0])/(paramBounds[i,1] - paramBounds[i,0])
        
    dta = pd.read_csv('../MORDMobjs/MORDM_new_WP1_thinned_Soln' + str(new_WP1.mostRobustSoln.solnNo) + '.obj',\
        sep=' ', names=['Hydro','Def2','AvgDef','MaxDef','Flood'])
    dta = pd.concat([LHsamples_df,dta],axis=1)
    predictors = dta.columns.tolist()[0:3]
    result1 = fitLogit(dta, dta.Flood, thresholds[0], predictors)
    result2 = fitLogit(dta, dta.Hydro, thresholds[1], predictors)
    
    # make Figure 8
    makeFigure8(result1, result2, paramBounds)
    
    return None
    
def getFormulations(name):
    formulation = Formulation()
    formulation.name = name
    
    formulation.Satisfy1 = np.loadtxt(name + '_Satisfy1.txt')
    formulation.bestFloodSoln.solnNo = np.argmax(formulation.Satisfy1[:,0])+1
    formulation.bestHydroSoln.solnNo = np.argmax(formulation.Satisfy1[:,1])+1
    formulation.bestDefSoln.solnNo = np.argmax(formulation.Satisfy1[:,2])+1
    formulation.mostRobustSoln.solnNo = np.argmax(formulation.Satisfy1[:,4])+1
    
    # load objective values for select solutions and convert deficit from fraction to pct
    formulation.bestFloodSoln.objs = np.loadtxt('./../MORDMobjs/MORDM_new_WP1_thinned_Soln' + \
        str(formulation.bestFloodSoln.solnNo) + '.obj')
    formulation.bestHydroSoln.objs = np.loadtxt('./../MORDMobjs/MORDM_new_WP1_thinned_Soln' + \
        str(formulation.bestHydroSoln.solnNo) + '.obj')
    formulation.bestDefSoln.objs = np.loadtxt('./../MORDMobjs/MORDM_new_WP1_thinned_Soln' + \
        str(formulation.bestDefSoln.solnNo) + '.obj')
    formulation.mostRobustSoln.objs = np.loadtxt('./../MORDMobjs/MORDM_new_WP1_thinned_Soln' + \
        str(formulation.mostRobustSoln.solnNo) + '.obj')
        
    return formulation
    
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
    
makeDotPlots()