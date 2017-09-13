def forwardProp(neurons, #neurons
                X, #input data
                outF='Identity', #final layer activation
                activation='sigmoid' #hidden layers activation
                ):
    import numpy as np
    L=len(neurons)+1 #layers count
    A=[]
    X=np.asmatrix(X)
    N=X.shape[0]
    A.append(np.hstack((np.ones((X.shape[0],1)),X)))
    for i in range(1,L):
        Z=np.dot(A[i-1],np.asmatrix(neurons[i-1]))
        if i<(L-1):
            if activation == 'sigmoid': A1=1/(1+np.exp(-Z))
            if activation == 'tanh': A1=1.7159*np.tanh((2/3)*Z)
            if activation == 'ReLU':
                A1=Z
                A1[Z<0]=0
            if activation == 'Linear': A.append(Z)
            A.append(np.hstack((np.ones((N,1)),A1)))
        if i==(L-1):
            if outF=='Identity': A.append(Z)
            if outF=='Sigmoid': A.append(1/(1+np.exp(-Z)))
            if outF=='Tanh': A.append(np.tanh(Z))
            if outF=='Softmax':
                K=X.shape[1]
                if K==1: A.append(1/(1+np.exp(-Z)))
                if K>2: A.append(Softmax(Z))
    return {'A':A}
