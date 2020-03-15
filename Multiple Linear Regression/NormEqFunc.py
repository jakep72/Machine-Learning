import numpy as np

def NormEqFunc(X,y):
    
    theta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
    
    ypred = np.dot(X,theta)
    return(theta,ypred)

 

