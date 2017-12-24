import numpy as np

def neuralNet(Input,response,InputTest=None,respTest=None,
              DW=6,
              opType='Regression',loss='RSS',outputFunc='Identity',
              epochs=None,
              tol=1e-7,
              activationFunc='sigmoid',
              rate=0.01,
              optAlg='GD',
              beta1=0.9,
              beta2=0.99,
              mini_batch=None,
              weightDecay=False,lambd=None,
              probs=1,
              batchNorm=False,
              gradientCheck=False,traceObj=False,traceWeights=False):
    import time
    t1=time.clock()
    
    if epochs==None: epochs=15000
    rsp=transformResponse(response,opType)
    active=init(DW,Input,rsp['respMat'],activationFunc,optAlg,batchNorm)
    optimize=OptimizeCost(active['W'],active['b'],rsp,Input,respTest,InputTest,
                          opType,loss,outputFunc,activationFunc,
                          rate,mini_batch,
                          [weightDecay,lambd],gradientCheck,traceObj,traceWeights,epochs,tol,
                          optAlg,beta1,beta2,
                          active['momentum'],active['momentum_b'],active['momentum_g'],active['momentum_betas'],
                          active['rmsprop'],active['rmsprop_b'],active['rmsprop_g'],active['rmsprop_betas'],
                          active['bnList'])
    
    t2=time.clock()
    L={'DW':DW,'tt':opType,'ll':loss,'outF':outputFunc,'active':activationFunc,'rr':rate,'wD':weightDecay,
       'lambd':lambd,'epochs':epochs,'bnVars':active['bnList']}
    return{'yhat':optimize['yhat'],'yhatMat':optimize['yhatMat'],'y':response,'CL':rsp['Classes'],
           'W':optimize['W'],'b':optimize['b'],'popStats':optimize['popStats'],
           'Delta':optimize['Delta'],'A':optimize['A'],'r':optimize['epochs'],
           'gradCheck':optimize['grad'].T,
           'trainLoss':optimize['trainLoss'],'testLoss':optimize['testLoss'],
           'trainScores':optimize['trainScores'],'testScores':optimize['testScores'],
           'trWeights':optimize['wTune'],'duration':t2-t1,'args':L
    }
