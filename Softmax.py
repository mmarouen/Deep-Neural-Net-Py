def Softmax(X):
    import numpy as np
    X=np.matrix(X)
    eps=1e-15
    Eps=1-eps
    M=np.max(X)
    prod=np.apply_along_axis(lambda x: np.exp(x-M)/np.sum(np.exp(X-M),axis=1),0,X)
    prod[prod>Eps]=Eps
    prod[prod<eps]=eps
    return(prod)
