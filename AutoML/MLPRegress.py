import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split


def MLPRegress(data, yname, testsize, numlayers, layersize):
    
    try:
        data = pd.read_csv(data,sep=",")
    except Exception:
        data = pd.read_excel()
        
    architect = (layersize,)*numlayers
    data = data.dropna()
    yframe = data[yname]
    Xframe = data.drop([yname], axis = 1)
    X = Xframe.values
    y = yframe.values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= testsize)
    mlp = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=architect,max_iter=15000))
    mlp.fit(X_train,y_train)
    
    train_score = mlp.score(X_train, y_train)
    test_score = mlp.score(X_test,y_test)
    ypredtest = mlp.predict(X_test)
    ypredtrain = mlp.predict(X_train)
    
    actual = "Actual"+" "+yname
    predicted = "Predicted"+" "+yname
    
    train_data = pd.DataFrame(data = X_train, columns = Xframe.columns)
    train_data[actual] = y_train
    train_data[predicted] = ypredtrain
    test_data = pd.DataFrame(data = X_test, columns = Xframe.columns)
    test_data[actual] = y_test
    test_data[predicted] = ypredtest
    
    return (train_score,test_score,train_data,test_data)
    
    

