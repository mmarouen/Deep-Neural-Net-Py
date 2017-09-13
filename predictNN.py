def predictNN(model,#neural network model
              X #input matrix
              ):
    args=model['agrs']
    out=forwardProp(model['Neurons'],X,args['outF'],args['active'])
    out2=transformOutput(out['A'],args['tt'],model['CL'])
    return(out2['yhat'])
