def transformOutput(A, #activations
                    CL, #classes
                    tt='Regression', #operation type
                    ):
    AL=A[len(A)-1]
    N,K=AL.shape
    yhat=np.array([])
    if tt=='Regression': yhat=AL
    if tt=='Classification':
        if K==1:
            yhat=np.zeros((N,1))
            yhat[AL[:,0]>0.5]=1
        if K>2:
            yhat=np.apply_along_axis(lambda x:CL[np.argmax(x)],1,AL)
            yhat=yhat.reshape((N,1))
    return {'yhat':yhat,'yhatMat':AL}
