#init the network
def init(layers,X,Y,activation,optimization,BN):
    X=np.asmatrix(X)
    K=Y.shape[1]
    cte=1
    momentum=[]
    momentum_b=[]
    momentum_g=[]
    momentum_betas=[]
    rmsprop=[]
    rmsprop_b=[]
    rmsprop_g=[]
    rmsprop_betas=[]
    gammas=[]
    betas=[]
    
    layers=np.append(layers,K)
    if layers[0]==0: layers=np.append(X.shape[1],K)
    if layers[0]>0: layers=np.append(X.shape[1],layers)
    layers=layers.reshape(1,layers.shape[0])
    L=layers.shape[1]
    if activation=='ReLU': cte=2
    weights=[float(0)]+list(map(lambda x:np.random.randn(layers[0,x-1],layers[0,x])*np.sqrt(cte/X.shape[1])*0.01,np.arange(1,L)))
    biases=[float(0)]+list(map(lambda x:np.zeros(layers[0,x]),np.arange(1,L)))
    
    if optimization in ['Adam','Momentum']:
        momentum=[float(0)]+list(map(lambda x:np.zeros((layers[0,x-1],layers[0,x])),np.arange(1,L)))
        momentum_b=[float(0)]+list(map(lambda x:np.zeros((layers[0,x-1],layers[0,x])),np.arange(1,L)))
        if BN:
            momentum_g=momentum_b.copy()
            momentum_betas=momentum_b.copy()
    if optimization in ['Adam','RMSProp']:
        rmsprop=[float(0)]+list(map(lambda x:np.zeros((layers[0,x-1],layers[0,x])),np.arange(1,L)))
        rmsprop_b=[float(0)]+list(map(lambda x:np.zeros((layers[0,x-1],layers[0,x])),np.arange(1,L)))
        if BN:
            rmsprop_g=rmsprop_b.copy()
            rmsprop_betas=rmsprop_b.copy()
    if BN:
        gammas=[float(0)]+list(map(lambda x:np.ones(layers[0,x]),np.arange(1,L)))
        betas=[float(0)]+list(map(lambda x:np.zeros(layers[0,x]),np.arange(1,L)))
    bnList={'BN':BN,'gammas':gammas,'betas':betas}
    
    return{'W':weights,'b':biases,'L':layers,'In':X,
              'rmsprop':rmsprop,'rmsprop_b':rmsprop_b,'rmsprop_g':rmsprop_g,'rmsprop_betas':rmsprop_betas,
              'momentum':momentum,'momentum_b':momentum_b,'momentum_g':momentum_g,'momentum_betas':momentum_betas,
              'bnList':bnList}

def transformResponse(response,tt='Regression'):
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

def transformOutput(A,tt,CL):
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

def Softmax(X):
    X=np.asmatrix(X)
    eps=1e-15
    Eps=1-eps
    M=np.max(X)
    prod=np.apply_along_axis(lambda x: np.exp(x-M)/np.sum(np.exp(X-M),axis=1),0,X)
    prod[prod>Eps]=Eps
    prod[prod<eps]=eps
    return(prod)

def delt(i,j):
    return(int(i==j))

def predictNN(model,X):
    args=model['agrs']
    out=forwardProp(model['W'],model['b'],X,args['outF'],args['active'],args['bnVars'],model['popStats'])
    out2=transformOutput(out['A'],args['tt'],model['CL'])
    return(out2['yhat'])

