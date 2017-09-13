def backProp(A, #activations 
            Neurons, #neurons 
            resp, #reponse matrix
            rr, #learning rate
            wD, #weight decay
            trW, #debugging: trace weights
            tol, #tolerance value
            t, #time
            ll='RSS', #loss function 
            outF='Identity', #final layer output
            activation='sigmoid', #hidden layer activation
            optimization='GD', #optimization algorithm
            beta1=0.9,beta2=0.99, #Momentum & RMSprop parameters
            momentum=None, #momentum data
            rmsprop=None #rmsprop data
            ):
    N,K=resp.shape
    L=len(A)
    delta=[None]*L
    gradRaw=[None]*L
    tmp=[None]*(L-1)
    traceW=[None]*(L-1)
    gradUpdate=[None]*(L-1)
    for i in reversed(range(1,L)):
        if i==(L-1):
            mat0=A[i]
            deltaL=mat0-resp
            if (ll=='RSS' and outF=='Identity') or (ll=='CrossEntropy' and outF=='Softmax'): delta[i]= deltaL
            if ll=='RSS' and outF=='Sigmoid': delta[i]=np.multiply(mat0,np.multiply((1-mat0),deltaL))
            if ll=='RSS' and outF=='Tanh': delta[i]=np.multiply(deltaL,(1-np.square(mat0)))
            if ll=='RSS' and outF=='Softmax': 
                mat1=np.zeros(mat0.shape)
                mat2=deltaL
                for p in range(K):
                    for j in  range(K): mat1[:,p]=mat1[:,p]+np.multiply(mat0[:,j],np.multiply(mat2[:,j],delt(p,j)-mat0[:,p]))
                delta[i]=mat1
        if i<(L-1):
            mat1=A[i][:,1:] 
            mat2=np.dot(delta[i+1],Neurons[i][1:,:].T)
            if activation == 'tanh': delta[i]=1.7159*(2/3)*np.multiply(1-np.square(mat1/1.7159),mat2)
            if activation == 'sigmoid':delta[i]=np.multiply(mat1,np.multiply(1-mat1,mat2))
            if activation == 'linear': delta[i]=mat2
            if activation == 'ReLU': 
                mat1[mat1>0]=1
                mat1[mat1==0]=0
                delta[i]=np.multiply(mat1,mat2)
        gradRaw[i]=(1/N)*np.dot(A[i-1].T,delta[i])
        
        if optimization =='GD': gradMat=gradRaw[i]
        if optimization in ['Adam','Momentum']:
            momentum[i-1]=beta1*momentum[i-1]+(1-beta1)*gradRaw[i]
            gradMat=momentum[i-1]
        if optimization in ['Adam','RMSProp']:
            rmsprop[i-1]=beta2*rmsprop[i-1]+(1-beta2)*(np.square(gradRaw[i]))
            gradMat=gradRaw[i]/np.sqrt(rmsprop[i-1]+tol)
        momentum_cor=np.zeros(gradRaw[i].shape)
        rmsprop_cor=np.zeros(gradRaw[i].shape)
        if optimization == 'Adam':
            momentum_cor=momentum[i-1]/(1-beta1**t)
            rmsprop_cor=rmsprop[i-1]/(1-beta2**t)
            gradMat=momentum_cor/np.sqrt(rmsprop_cor+tol)
        tmp[i-1]=Neurons[i-1]-rr*gradMat
        if trW:
            weightAbsMean=np.mean(np.abs(tmp[i-1][2:,:]))
            biasAbsMean=np.mean(np.abs(tmp[i-1][1,:]))
            weightAbsUpdateMean=np.mean(np.abs(tmp[i-1][2:,:]-Neurons[i-1][2:,:]))
            biasAbsUpdateMean=np.mean(np.abs(tmp[i-1][1,:]-Neurons[i-1][1,:]))
            traceW[i-1]=np.array([weightAbsUpdateMean/weightAbsMean,biasAbsUpdateMean/biasAbsMean]).reshape((2,1))
            gradUpdate[i-1]=rr*gradMat
    return{'Neurons':tmp,'D':delta,'tr':traceW,'gradUpdate':gradUpdate,'Momentum':momentum,'RMSProp':rmsprop}  
