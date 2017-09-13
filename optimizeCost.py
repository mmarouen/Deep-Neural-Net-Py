#neural network optimizer
def OptimizeCost(Neurons,resp,X,respT=None,XT=None, #input data
             tt='Regression', #operation type: 'Regression' or 'Classification'
             ll='RSS', #loss function 'RSS' or 'Classification'
             outF='Identity', #final layer output function
             activation='sigmoid', #hidden layer activation: 'tanh', 'identity', 'sigmoid', 'ReLU'
             rr=0.01, #learning rate
             minib=None, #minibatch size in power of 2
             wD=None,gradCheck=False,traceObj=False,traceW=False,Epochs=None,Tolerance=1e-6,   
             optimization='GD',beta1=0.9,beta2=0.99,momentum=None,rmsprop=None
             ):
    r=1
    t=0
    import numpy as np
    error=None
    resp0=np.asmatrix(resp['respMat'])
    response=np.asmatrix(resp['Response'])
    CL=resp['Classes']
    resp1=resp0
    resp2=np.zeros(resp1.shape)
    lossCurve=np.array([])
    lossCurve2=np.array([])
    score=np.array([])
    score2=np.array([])
    weightsRatio=[]
    weightTune=[]
    
    if traceObj and XT is not None and respT is not None:
        XT=np.asmatrix(XT)
        out=transformResponse(respT,tt)
        respT0=out['respMat']
    if traceW:
        for i in range(len(Neurons)): weightsRatio[i]=0
    N=len(X) #total observations
    
    if minib is None: m=N #compute minibatch size
    else: m=2**minib
    
    if m == N: NB=1 #compute number of batches 
    else:
        if N%m==0: NB=N//m
        else: NB=N//m+1
    error=np.array(np.repeat(-1,10),dtype=float)
    X=np.asmatrix(X)
    resp0=np.asmatrix(resp0)
    #cond=False
    while True:
        resp2=np.zeros(resp1.shape)
        
        for b in range(NB):
            t+=1
            #load mini batch data
            if NB==1: b_size=N
            else:
                if b != NB: b_size=m
                else: b_size=N-(b-1)*m
            idx=range((b-1)*m,(b-1)*m+b_size)       
            Xb=np.asmatrix(X[idx,:])
            yb=np.asmatrix(resp0[idx,:])
            
            #forward propagation
            FP=forwardProp(Neurons,Xb,outF,activation)
            A=FP['A']
            #backprop
            BP=backProp(A,Neurons,yb,ll=ll,outF=outF,activation=activation,
                        rr=rr,wD=wD,trW=traceW,optimization=optimization,
                        beta1=beta1,beta2=beta2,momentum=momentum,rmsprop=rmsprop,
                        tol=Tolerance,t=t)
            #update network response
            rsp2=transformOutput(A,tt,CL)
            resp2[idx,:]=np.asmatrix(rsp2['yhatMat'])
            
            #gradient check
            if gradCheck and np.any(r==np.array([500,700,1000,2000,3000,5000,6000,7000,8000,9000])):
                check=gradchecker(Neurons,BP['D'],A,yb,wD,tt,ll,outF,activation,Tolerance)
                error[np.where(np.array([500,700,1000,2000,3000,5000,6000,7000,8000,9000])==r)]=check
            
            #weights update
            Neurons=BP['Neurons']
            #gradient update
            momentum=BP['Momentum']
            rmsprop=BP['RMSProp']
            
        if traceObj:#compute cost
            FP_obj=forwardProp(Neurons,X,outF,activation)
            A_obj=FP_obj['A']
            obj=objective(A_obj[len(A_obj)-1],resp0,Neurons,wD,ll)
            lossCurve=np.append(lossCurve,obj)
            out=transformOutput(A_obj,CL,tt)
            respTr=np.asmatrix(out['yhat'])
            if tt=='Classification': score=np.append(score,np.mean(respTr==np.asmatrix(response)))
            if tt=='Regression': score=np.append(score,1-obj)
            if XT is not None and respT is not None:
                FP_obj=forwardProp(Neurons,XT,outF,activation)
                A_obj=FP_obj['A']
                obj=objective(A_obj[len(A_obj)-1],respT0,Neurons,wD,ll)
                lossCurve2=np.append(lossCurve2,obj)
                out=transformOutput(A_obj,CL,tt)
                respTe=np.asmatrix(out['yhat'])
                if tt=='Classification': score2=np.append(score2,np.mean(respTe==np.asmatrix(respT)))
                if tt=='Regression': score=np.append(score,1-obj)
        
        if traceW: #trace weights evolution
            for i in range(len(Neurons)): weightsRatio[i]=np.hstack(weightsRatio[i],BP['tr'][i])
        
        if r%1000==0: print(r)
        if np.mean(np.abs(resp2-resp1)).mean() <Tolerance or r>Epochs: break

        r+=1
        resp1=resp2
    if traceW:
        for i in range(len(Neurons)): weightsRatio[i]=np.hstack(weightsRatio[i],BP['tr'][i])
        weightTune=weightsRatio + BP['gradUpdate']
    
    FP=forwardProp(Neurons,X,outF,activation)    
    
    trans=transformOutput(FP['A'],CL,tt)
    return{'yhat':trans['yhat'],'A':A,'Neurons':Neurons,'Delta':BP['D'],'epochs':r,'yhatMat':trans['yhatMat'],
          'grad':error,'trainLoss':lossCurve,'testLoss':lossCurve2,'trainScores':score,'testScores':score2,
          'wTune':weightTune}
