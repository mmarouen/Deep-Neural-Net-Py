#Performs forward propagation

def forwardProp(weights,biases,X,outF,activation,bnVars,popStats=None):
    
    #load FP parameters
    L=len(weights) #layers count
    BN=bnVars['BN']
    gammas=bnVars['gammas']
    betas=bnVars['betas']
    
    #init response
    X=np.asmatrix(X)
    A=[]
    A.append(X)
    Zhat=[]
    Zhat.append(0)
    sigma2=[]
    sigma2.append(0)
    mu=[]
    mu.append(0)
    
    for i in range(1,L):
        if ~BN:Z=np.dot(A[i-1],weights[i])+biases[i]
        if BN:
            Z=np.dot(A[i-1],weights[i])
            mu.append(np.mean(Z,axis=0))
            mu_Z=mu[i].copy()
            if popStats is not None: mu_Z=popStats['mu'][i]
            demean=Z-mu_Z
            sigma2.append(np.mean(np.power(demean,2),axis=0)+1e-6)
            sigma2_Z=np.asanyarray(sigma2[i].copy(),dtype=float)
            if popStats is not None: sigma2_Z=popStats['sigma2'][i]
            Zhat.append(demean/np.sqrt(sigma2_Z))
            Z=np.multiply(Zhat[i],gammas[i])+betas[i]
        Z=np.asarray(Z,dtype=float)
        if i<(L-1):
            if activation == 'sigmoid': A1=1/(1+np.exp(-Z))
            if activation == 'tanh': A1=1.7159*np.tanh((2/3)*Z)
            if activation == 'ReLU':
                A1=Z.copy()
                A1[Z<0]=0
            if activation == 'Linear': A.append(Z)
            A.append(A1)
        if i==(L-1):
            if outF=='Identity': A.append(Z)
            if outF=='Sigmoid': A.append(1/(1+np.exp(-Z)))
            if outF=='Tanh': A.append(np.tanh(Z))
            if outF=='Softmax':
                K=Z.shape[1]
                if K==1: A.append(1/(1+np.exp(-Z)))
                if K>2: A.append(Softmax(Z))
    popStats={'mu':mu,'sigma2':sigma2}
    return {'A':A,'Zhat':Zhat,'popStats':popStats}

def backProp(A,weight,biases,resp,ll,outF,
             activation,rr,wD,trW,
             optimization,beta1,beta2,
             momentum,momentum_b,momentum_g,momentum_betas,
             rmsprop,rmsprop_b,rmsprop_g,rmsprop_betas,
             tol,t,BNVars):
    
    #load parameters
    N,K=resp.shape
    L=len(A)
    BN=BNVars['BN']
    Z_hat=BNVars['Zhat']
    sigma2=BNVars['sigma2']
    gammas=BNVars['gammas']
    betas=BNVars['betas']
    
    #init lists
    delta=[None]*L
    dltaHat=[None]*L
    gradRaw=[None]*L
    gradRaw_b=[None]*L
    grdRaw_gammas=[None]*L
    grdRaw_betas=[None]*L
    tmp=[None]*L
    tmp_b=[None]*L
    tmp_g=[None]*L
    tmp_betas=[None]*L
    traceW=[None]*L
    gradUpdate=[None]*L
    
    for i in reversed(range(1,L)):
        if i==(L-1):
            mat0=A[i].copy()
            deltaL=mat0-resp
            if (ll=='RSS' and outF=='Identity') or (ll=='CrossEntropy' and outF=='Softmax'): dltaHat[i]= deltaL
            if ll=='RSS' and outF=='Sigmoid': dltaHat[i]=np.multiply(mat0,np.multiply((1-mat0),deltaL))
            if ll=='RSS' and outF=='Tanh': dltaHat[i]=np.multiply(deltaL,(1-np.square(mat0)))
            if ll=='RSS' and outF=='Softmax': 
                mat1=np.zeros(mat0.shape)
                mat2=deltaL
                for p in range(K):
                    for j in  range(K): 
                        mat1[:,p]=mat1[:,p]+np.multiply(mat0[:,j],np.multiply(mat2[:,j],delt(p,j)-mat0[:,p]))
                dltaHat[i]=mat1.copy()
        if i<(L-1):
            mat1=A[i].copy()
            mat2=np.dot(delta[i+1],weight[i+1].T)
            if activation == 'tanh': dltaHat[i]=1.7159*(2/3)*np.multiply(1-np.square(mat1/1.7159),mat2)
            if activation == 'sigmoid':dltaHat[i]=np.multiply(mat1,np.multiply(1-mat1,mat2))
            if activation == 'linear': dltaHat[i]=mat2
            if activation == 'ReLU': 
                mat1[mat1>0]=1
                mat1[mat1==0]=0
                dltaHat[i]=np.multiply(mat1,mat2)
        if ~BN: delta[i]=dltaHat[i]
        if BN:
            Z=np.asmatrix(Z_hat[i])
            D=np.asmatrix(dltaHat[i])
            dg=np.mean(np.multiply(Z,D),axis=0)
            db=np.mean(D,axis=0)
            m1=np.multiply(Z,dg)
            m1=D-m1
            m1=m1-db
            s2=np.asarray(sigma2[i].copy(),dtype=float)
            mult=gammas[i]/np.sqrt(s2)
            grdRaw_gammas[i]=dg
            grdRaw_betas[i]=db
            delta[i]=np.multiply(m1,mult)
        
        #update gradients
        gradRaw[i]=(1/N)*np.dot(A[i-1].T,delta[i])
        if wD[0]:
            lambd=wD[1]
            gradRaw[i]+=lambd*weight[i]
        gradRaw_b[i]=np.mean(delta[i],axis=0)
        
        #GD
        if optimization =='GD': 
            gradMat=gradRaw[i].copy()
            gradMat_b=gradRaw_b[i].copy()
            if BN:
                gradMat_g=grdRaw_gammas[i]
                gradMat_betas=grdRaw_betas[i]
        
        #momentum
        if optimization in ['Adam','Momentum']:
            momentum[i]=beta1*momentum[i]+(1-beta1)*gradRaw[i]
            momentum_b[i]=beta1*momentum_b[i]+(1-beta1)*gradRaw_b[i]
            gradMat=momentum[i].copy()
            gradMat_b=momentum_b[i].copy()            
            if BN:
                momentum_g[i]=beta1*momentum_g[i]+(1-beta1)*grdRaw_gammas[i]
                momentum_betas[i]=beta1*momentum_betas[i]+(1-beta1)*grdRaw_betas[i]
                gradMat_g=momentum_g[i].copy()
                gradMat_betas=momentum_betas[i].copy()
        
        #rmsprop
        if optimization in ['Adam','RMSProp']:
            rmsprop[i]=beta2*rmsprop[i]+(1-beta2)*(np.square(gradRaw[i]))
            gradMat=gradRaw[i]/np.sqrt(rmsprop[i]+tol)
            if BN:
                rmsprop_g[i]=beta1*rmsprop_g[i]+(1-beta1)*grdRaw_gammas[i]
                rmsprop_betas[i]=beta1*rmsprop_betas[i]+(1-beta1)*grdRaw_betas[i]
                gradMat_g=rmsprop_g[i].copy()
                gradMat_betas=rmsprop_betas[i].copy()
        
        #Adam
        if optimization == 'Adam':
            momentum_cor=np.zeros(gradRaw[i].shape)
            momentum_cor_b=np.zeros(gradRaw_b[i].shape)
            momentum_cor_g=np.zeros(grdRaw_gammas[i].shape)
            momentum_cor_betas=np.zeros(grdRaw_betas[i].shape)
            rmsprop_cor=np.zeros(gradRaw[i].shape)
            rmsprop_cor_b=np.zeros(gradRaw_b[i].shape)
            rmsprop_cor_g=np.zeros(grdRaw_gammas[i].shape)
            rmsprop_cor_betas=np.zeros(grdRaw_betas[i].shape)
            momentum_cor=momentum[i]/(1-beta1**t)
            rmsprop_cor=rmsprop[i]/(1-beta2**t)
            momentum_cor_b=momentum_b[i]/(1-beta1**t)
            rmsprop_cor_b=rmsprop_b[i]/(1-beta2**t)
            gradMat=momentum_cor/np.sqrt(rmsprop_cor+tol)
            gradMat_b=momentum_cor_b/np.sqrt(rmsprop_cor_b+tol)
            if BN:
                momentum_cor_g=momentum_g[i]/(1-beta1**t)
                rmsprop_cor_g=rmsprop_g[i]/(1-beta2**t)
                momentum_cor_betas=momentum_betas[i]/(1-beta1**t)
                rmsprop_cor_betas=rmsprop_betas[i]/(1-beta2**t)
                gradMat_g=momentum_cor_g/np.sqrt(rmsprop_cor_g+tol)
                gradMat_betas=momentum_cor_betas/np.sqrt(rmsprop_cor_betas+tol)
        
        #coefficients update
        tmp[i]=weight[i]-rr*gradMat
        tmp_b[i]=biases[i]-rr*gradMat_b
        if BN:
            tmp_g[i]=gammas[i]-rr*gradMat_g
            tmp_betas[i]=betas[i]-rr*gradMat_betas
        if trW:
            weightAbsMean=np.mean(np.abs(tmp[i]))
            biasAbsMean=np.mean(np.abs(tmp_b[i]))
            weightAbsUpdateMean=np.mean(np.abs(tmp[i]-weight[i]))
            biasAbsUpdateMean=np.mean(np.abs(tmp_b[i]-biases[i]))
            traceW[i]=np.array([weightAbsUpdateMean/weightAbsMean,biasAbsUpdateMean/biasAbsMean]).reshape((2,1))
            gradUpdate[i]=rr*gradMat
    return{'W':tmp,'biases':tmp_b,'D':delta,'Dhat':dltaHat,'g':tmp_g,'b':tmp_betas,
           'tr':traceW,'gradUpdate':gradUpdate,
           'momentum':momentum,'momentum_b':momentum_b,'momentum_g':momentum_g,'momentum_betas':momentum_betas,
           'rmsprop':rmsprop,'rmsprop_b':rmsprop_b,'rmsprop_g':rmsprop_g,'rmsprop_betas':rmsprop_betas}        
