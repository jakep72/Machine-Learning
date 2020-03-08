import numpy as np
import matplotlib.pyplot as plt

X = np.array(([1,1,1,1,1],[1,2,3,4,5],[3,2,4,6,15]),dtype = float)
X = np.transpose(X)
m = len(X)

slopes = np.array([25,.5,15], dtype = float)
y = np.zeros(5)

y = np.dot(X,slopes)

theta = np.zeros(3)

alpha = .01
num_iter = 10000
J = np.zeros(num_iter)

for i in range(num_iter):
    h  = np.dot(X,theta)
    tempT0 = theta[0] - ((alpha/m)*sum(h-y))
    tempT1 = theta[1] - ((alpha/m)*sum((h-y)*X[:,1]))
    tempT2 = theta[2] - ((alpha/m)*sum((h-y)*X[:,2]))
    theta[0] = tempT0
    theta[1] = tempT1
    theta[2] = tempT2
    J[i] = (1/2*m)*sum((np.dot(X,theta)-y)**2)
    
ypred = np.dot(X,theta)


plt.plot(y,ypred)   