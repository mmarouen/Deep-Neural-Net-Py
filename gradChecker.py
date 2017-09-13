def gradchecker(Neurons, #neurons
                Delta, #gradients
                A, #activations
                response, #response matrix
                wD, #weight decay
                tt='Regression', #operation type
                ll='RSS', #loss function 
                outF='Identity', #final layer activation 
                activation='sigmoid', #hidden layer activation
                tol=1e-5):
    L=len(A)
    l=np.random.choice(L-1)
    i=np.random.choice(Neurons[l].shape[0])
    bias=False
    if i==0: bias=True
    j=np.random.choice(Neurons[l].shape[1])
    X=A[0][:,1:]
    N=len(X)
    ###compute numerical gradient
    W_ij=Neurons[l][i,j]
    #right side
    Neurons[l][i,j]=W_ij+tol
    fp1=forwardProp(Neurons,X,outF,activation)
    mat1=fp1['A'][L-1]
    estPlus=objective(mat1,response,Neurons,wD,ll)
    #left side
    Neurons[l][i,j]=W_ij-tol
    fp2=forwardProp(Neurons,X,outF,activation)
    mat2=fp2['A'][L-1]
    estMinus=objective(mat2,response,Neurons,wD,ll)
    #gradient
    gradEst=(estPlus-estMinus)/(2*tol)
    ####Analytical gradient
    error=np.float64()
    toCheck=np.mean(np.multiply(A[l][:,i].reshape((N,1)),Delta[l+1][:,j].reshape((N,1))))
    if wD[0] and not(bias): toCheck+=wD[1]*W_ij
    error=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
    return(error)
