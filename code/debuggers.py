def gradchecker(W,b,delta,deltaHat,A,Zhat,resp,wD,
                tt,ll,outF,active,
                tol,bnList):
    L=len(A)
    l=np.random.choice(range(1,L))
    i=np.random.choice(W[l].shape[0])
    j=np.random.choice(W[l].shape[1]) #indice for biases and weight
    N,K=resp.shape
    BN=bnList['BN']
    
    X=A[0]
    ####1. Weight gradient
    #Numerical gradient
    W_ij=W[l][i,j]
    #right side
    W[l][i,j]=W_ij+tol
    fp=forwardProp(W,b,X,outF,active,bnList)
    mat=fp['A'][L-1]
    estPlus=objective(mat,resp,W,wD,ll)
    #left side
    W[l][i,j]=W_ij-tol
    fp=forwardProp(W,b,X,outF,active,bnList)
    mat=fp['A'][L-1]
    estMinus=objective(mat,resp,W,wD,ll)
    #gradient
    gradEst=(estPlus-estMinus)/(2*tol)
    ####Analytical gradient
    error1=np.float64()
    toCheck=np.mean(np.multiply(A[l-1][:,i].reshape((N,1)),delta[l][:,j].reshape((N,1))))
    if wD[0]: toCheck+=wD[1]*W_ij
    error1=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
    W[l][i,j]=W_ij
    
    if ~BN: 
        #####2. Bias gradient
        #Numerical gradient
        b_ij=b[l][:,j].copy()
        #right side
        b[l][:,j]=b_ij+tol
        fp=forwardProp(W,b,X,outF,active,bnList)
        mat=fp['A'][L-1]
        estPlus=objective(mat,resp,W,wD,ll)
        #left side
        b[l][:,j]=b_ij-tol
        fp=forwardProp(W,b,X,outF,active,bnList)
        mat=fp['A'][L-1]
        estMinus=objective(mat,resp,W,wD,ll)
        #gradient
        gradEst=(estPlus-estMinus)/(2*tol)
        ####Analytical gradient
        error2=np.float64()
        toCheck=np.mean(delta[l][:,j])
        error2=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
        b[l][:,j]=b_ij
        error=np.array([i,j,error1,error2])
    
    if BN:
        g_j=bnList['gammas'][l][:,j].copy()
        b_j=bnList['betas'][l][:,j].copy()
        #####2. gamma gradient
        #right side
        bnList['gammas'][l][:,j]=g_j+tol
        fp=forwardProp(W,b,X,outF,active,bnList)
        mat=fp['A'][L-1]
        estPlus=objective(mat,resp,W,wD,ll)
        #left side
        bnList['gammas'][l][:,j]=g_j-tol
        fp=forwardProp(W,b,X,outF,active,bnList)
        mat=fp['A'][L-1]
        estMinus=objective(mat,resp,W,wD,ll)
        #gradient
        gradEst=(estPlus-estMinus)/(2*tol)
        #analytical gradient
        toCheck=np.mean(np.multiply(Zhat[l][:,j].reshape((N,1)),deltaHat[l][:,j].reshape((N,1))))
        error2=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
        bnList['gammas'][l][:,j]=g_j
        #####3. beta gradient
        #right side
        bnList['betas'][l][:,j]=b_j+tol
        fp=forwardProp(W,b,X,outF,active,bnList)
        mat=fp['A'][L-1]
        estPlus=objective(mat,resp,W,wD,ll)
        #left side
        bnList['betas'][l][:,j]=b_j-tol
        fp=forwardProp(W,b,X,outF,active,bnList)
        mat=fp['A'][L-1]
        estMinus=objective(mat,resp,W,wD,ll)
        #gradient
        gradEst=(estPlus-estMinus)/(2*tol)
        #analytical gradient
        toCheck=np.mean(deltaHat[l][:,j])
        error3=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
        bnList['betas'][l][:,j]=b_j
        error=np.array([i,j,error1,error2,error3])
    return(error)

def objective(X,Y,Neurons,wD,ll):
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
        for i in  range(1,L):
            mat=Neurons[i][1:,]
            s+=np.sum(mat)
        obj+=0.5*lambd
    return(obj)
