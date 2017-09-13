def init(X, #input data
        Y, #response matrix
        layers=4, #count of layers
        activation='sigmoid', #activation function
        optimization='GD' #optimization algorithm
        ):
    import numpy as np
    X=np.asmatrix(X)
    K=Y.shape[1]
    cte=1
    momentum=[]
    rmsprop=[]
    neurons=[]
    layers=np.append(layers,K)
    if layers[0]==0: layers=np.append(X.shape[1],K)
    if layers[0]>0: layers=np.append(X.shape[1],layers)
    layers=layers.reshape(1,layers.shape[0])
    L=layers.shape[1]
    if activation=='ReLU': cte=2
    neurons=list(map(lambda x:np.random.randn(layers[0,x-1]+1,layers[0,x])*np.sqrt(cte/X.shape[1])*0.01,np.arange(1,L)))
    if optimization in ['Adam','Momentum']:
        momentum=list(map(lambda x:np.zeros(layers[0,x-1]+1,layers[0,x]),np.arange(1,L)))
    if optimization in ['Adam','RMSProp']:
        rmsprop=list(map(lambda x:np.zeros(layers[0,x-1]+1,layers[0,x]),np.arange(1,L)))
    return{'Neurons':neurons,'RMSProp':rmsprop,'Momentum':momentum}
