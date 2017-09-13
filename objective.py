def objective(X, #data matrix
              Y, #reponse
              Neurons, #neurons
              wD, #weight decay
              ll='RSS' #loss function
              ):
    import numpy as np
    N,K=X.shape
    L=len(Neurons)
    obj=0
    if ll=='RSS': obj=0.5*np.mean(np.sum(np.multiply(X,Y),axis=1))
    if ll=='CrossEntropy':
        if K==1: obj=-np.mean(np.multiply(Y,np.log(X))+np.multiply((1-Y),np.log(1-X)))
        if K>2: obj=-np.mean(np.sum(np.multiply(Y,np.log(X)),axis=1))
    if wD[0]:
        lambd=wD[1]
        s=0
        for i in  range(L):
            mat=Neurons[i][1:,]
            s+=np.sum(mat)
        obj+=0.5*lambd
    return(obj)
