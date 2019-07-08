#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 19:32:47 2019

@author: marcocianciotta
"""

import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

#Standardizzazione dei dati
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values

def centra():
   
    for i in range (0,4):
        x[:,i]=x[:,i]-x.mean(0)[i]
        

centra()

U,s,Vt= np.linalg.svd(x)
print('----------------------------------------------------')

print('U')

print(U)
print('----------------------------------------------------')

print('s')
print(s)
print('----------------------------------------------------')

print('Vt')
print(Vt)

Cov= np.cov(x, rowvar=False)
print('----------------------------------------------------')

print('COvarianza')
print(Cov)

w, v =np.linalg.eig(Cov)
print('----------------------------------------------------')

print('Eigenvalues')
print(w)
print('----------------------------------------------------')

print('Eigenvectors')
print(v)

# Standardizing the features
x = StandardScaler().fit_transform(x)

#The original data has 4 column (sepal length, sepal width, petal length, and petal width)
#Now we projects the original data which is 4 dimensional into 2-D
# Bisogna far presente che dopo la riduzione di dimensionalità, di solito non c'è un significato
# particolare assengato a ciascuna componente principale.
# I nuovi compomnenti sono solo le due dimensioni principali della variazione.


#Visualizzazione dei dati 2-D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

#concatenazione del target

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)



fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#Visualizzazione dei dati 3-D
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

#concatenazione del target

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

#Visualizzazione dei dati 2-D

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d') 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#Scree Plot    



fig = plt.figure(figsize=(8,5))
sing_vals = np.arange(4) + 1
plt.plot(sing_vals, w, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
leg = plt.legend(['Eigenvalues from Covariance Matrix'], loc='best', borderpad=0.3, 
                 shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                 markerscale=0.4)
leg.get_frame().set_alpha(0.4)
plt.show()


print('----------------------------------------------------')
