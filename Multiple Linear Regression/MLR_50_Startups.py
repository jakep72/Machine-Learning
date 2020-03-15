import pandas as pd
import numpy as np
from NormEqFunc import NormEqFunc
import matplotlib.pyplot as plt

#import data found at https://www.kaggle.com/karthickveerakumar/startup-logistic-regression/version/1
data = pd.read_csv('50_startups.csv')

#one hot encode the 'State' column
data = pd.concat([data,pd.get_dummies(data['State'],prefix = 'state')],axis = 1)
data.drop(['State'],axis =1, inplace = True)

#split data and convert to numpy arrays
X = data[['R&D Spend','Administration','Marketing Spend','state_California','state_Florida','state_New York']].values
y = data[['Profit']].values

#Use the Normal Equation to calculate the weights and return predicted profit
results = NormEqFunc(X,y)

#split results into 
ypred = results[1]
theta = results[0]

#calculate the average absolute error
aveabserr = np.mean(abs(y-ypred)/y)

#plot actual profits vs predicted
plt.plot(y,ypred)




