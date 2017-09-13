def transformResponse(response, #response vector
                      tt='Regression' #operation type
                      ):
    import numpy as np
    N=len(response)
    response=np.reshape(response,(len(response),1))
    respMat=np.array([])
    classes=np.array([])
    if tt=='Regression': respMat=np.asmatrix(response)
    if tt=='Classification':
        K=len(np.unique(response[:,0]))
        if K==2: respMat=np.asmatrix(response)
        if K>2:
            respMat=np.zeros((N,K))
            classes=np.sort(np.unique(response))
            def loadValues(x):
                respMat[x,np.where(classes==response[x,0])]=1
                return(respMat[x,:])
            respMat=np.apply_along_axis(loadValues,1,np.arange(N).reshape((N,1)))
    return {'Response':response,'respMat':respMat,'Classes':classes}
