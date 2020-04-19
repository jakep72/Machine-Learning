import numpy as np

#create example input array
X = np.array([1,2,3,2.5])

#create example weight array
W = np.array([[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]])

#create example bias array
b = np.array([2, 3, 0.5])

#vector notation: output = W*X^T+B
output = np.dot(W,np.transpose(X))+b

print(output) 





