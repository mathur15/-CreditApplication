import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone,pcolor,colorbar,plot,show

data = pd.read_csv('Credit_Card_Applications.csv')
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

sc = MinMaxScaler()
X = sc.fit_transform(X)

som  = MiniSom(x=10,y=10,input_len=15)
som.random_weights_init(data = X)
som.train_random(data = X,num_iteration=100)

bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r', 'g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,w[1]+0.5,
    markers[y[i]],
    markeredgecolor = colors[y[i]],
    markerfacecolor = 'None',
    markersize = 10,
    markeredgewidth = 2)
show()

#Get a dictionary of the winner neurons
mappings = som.win_map(X)
#Get the data related to the box coordinate
#coordinates from the visualization
frauds = np.concatenate((mappings[(3,6)], mappings[(2,7)]), axis= 0)
frauds = sc.inverse_transform(frauds)







