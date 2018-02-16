import numpy as np
    
def SatisfyTypeI(Objs, thresholds):
    '''Calculates the percent of SOWs in which specified Satisficing Criteria\n
    are met.'''
    SatisfyMatrix = np.zeros([np.shape(Objs)[0],np.shape(Objs)[1],2*len(thresholds)-1])
    for i in range(np.shape(SatisfyMatrix)[0]):
        for j in range(np.shape(SatisfyMatrix)[1]):
            for k in range(len(thresholds)):
                # determine if each individual objective threshold is met
                if Objs[i,j,k] < thresholds[k]:
                    SatisfyMatrix[i,j,k] = 1
                    
            # determine if series of combined metrics are met
            if Objs[i,j,0] < thresholds[0] and Objs[i,j,1] < thresholds[1]:
                SatisfyMatrix[i,j,len(thresholds)] = 1
            if Objs[i,j,0] < thresholds[0] and Objs[i,j,1] < thresholds[1] and Objs[i,j,2] < thresholds[2]:
                SatisfyMatrix[i,j,len(thresholds)+1] = 1
                
    satisfaction = np.mean(SatisfyMatrix,1)
    
    return satisfaction
