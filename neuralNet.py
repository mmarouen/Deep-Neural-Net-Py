def neuralNet(Input,response,InputTest=None,respTest=None, #input data
              DW=6, #neurons vector
              opType='Regression', #'Regression' or 'Classification' 
              loss='RSS', loss function 'RSS' or 'CrossEntropy'
              outputFunc='Identity', output function 
              epochs=None, # count of epochs
              tol=1e-7, #tolerance
              activationFunc='sigmoid', #activation function
              rate=0.01, #learning rate
              optAlg='GD', #optimization alg: 'GD', 'Momentum', 'RMSProp'
              beta1=0.9, #beta1 for momentum
              beta2=0.99, #beta2 for RMSProp
              mini_batch=None, #mini batch size in exponent of 2
              weightDecay=False,lambd=None, #weight decay
              probs=1, #dropout factor
              gradientCheck=False,traceObj=False,traceWeights=False #debugging
              ):
    import time
    t1=time.clock()
    
    if epochs==None: epochs=15000
    rsp=transformResponse(response,opType)
    active=init(Input,rsp['respMat'],DW,activationFunc,optAlg)
    optimize=OptimizeCost(active['Neurons'],rsp,Input,respT=respTest,XT=InputTest,
                          tt=opType,ll=loss,outF=outputFunc,activation=activationFunc,
                          rr=rate,minib=mini_batch,
                          wD=[weightDecay,lambd],gradCheck=gradientCheck,traceObj=traceObj,
                          traceW=traceWeights,Epochs=epochs,Tolerance=tol,
                          optimization=optAlg,beta1=beta1,beta2=beta2,
                          momentum=active['Momentum'],rmsprop=active['RMSProp'])
    t2=time.clock()
    L={'DW':DW,'tt':opType,'ll':loss,'outF':outputFunc,'active':activationFunc,'rr':rate,'wD':weightDecay,
       'lambd':lambd,'epochs':epochs}
    return{'yhat':optimize['yhat'],'yhatMat':optimize['yhatMat'],'y':response,'CL':rsp['Classes'],'Neurons':optimize['Neurons'],
           'Delta':optimize['Delta'],'A':optimize['A'],'r':optimize['epochs'],'gradCheck':optimize['grad'],
           'trainLoss':optimize['trainLoss'],'testLoss':optimize['testLoss'],
           'trainScores':optimize['trainScores'],'testScores':optimize['testScores'],
           'trWeights':optimize['wTune'],'duration':t2-t1,'args':L
    }
