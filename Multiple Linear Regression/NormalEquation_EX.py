import numpy as np
import matplotlib.pyplot as plt

X = np.array(([1,1,1,1,1],[1,2,3,4,5],[3,2,4,6,15]),dtype = float)
X = np.transpose(X)
m = len(X)

slopes = np.array([25,.5,15], dtype = float)
y = np.zeros(5)

y = np.dot(X,slopes)

theta = np.dot(np.linalg.inv(np.dot(np.transpose(X),X)),np.dot(np.transpose(X),y))
    
ypred = np.dot(X,theta)


plt.plot(y,ypred)   

