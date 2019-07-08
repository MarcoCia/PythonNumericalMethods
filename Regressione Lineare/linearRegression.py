#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 21:19:44 2019

@author: marcocianciotta
"""

from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
import numpy as np

X, y= make_regression(n_samples=100, n_features=1, noise=0.4, bias=50)




def ShowLine(theta0, theta1, X, y):
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100


    xplot = np.linspace(min_x, max_x, 1000)
    yplot = theta0 + theta1 * xplot



    plt.plot(xplot, yplot, color='r', label='Regression Line')

    plt.scatter(X,y)
    plt.axis([-10, 10, 0, 200])
    plt.show()



def ipotesi(theta0, theta1, x):
    '''
    L'equazione lineare è la forma standard che rappresenta
    appunto una retta su un grafico dove theta0 rappresenta il gradiente
    e theta1 rappresenta l'intercetta.
    '''
    return theta0 + (theta1*x) 

def cost(theta0, theta1, X, y):
    costValue = 0 
    for (xi, yi) in zip(X, y):
        costValue += 0.5 * ((ipotesi(theta0, theta1, xi) - yi)**2)
    return costValue




def derivata(theta0, theta1, X, y):
    '''
    il nostro obiettivo è quindi calcolare il punto più piccolo nel grafico dove l'errore 
    è appunto il più piccolo. (minimizzazione della funzione costo)
    per fare questo dobbiamo considerare cosa accade sul fondo del grafico 
    (- il gradiante è zero).
    Ovvero per minimizzare la cost function dobbiamo portare il gradiente a zero.
    Il gradiente è dato dalla derivata della funzione e dslla derivata parziale della funzione.
    '''
    dtheta0 = 0
    dtheta1 = 0
    for (xi, yi) in zip(X, y):
        dtheta0 += ipotesi(theta0, theta1, xi) - yi
        dtheta1 += (ipotesi(theta0, theta1, xi) - yi)*xi

    dtheta0 /= len(X)
    dtheta1 /= len(X)

    return dtheta0, dtheta1

def AggiornaParametri(theta0, theta1, X, y, alpha):
    '''
    dobbiamo ora effettuare un update dei parametri di theta in modo da minimizzare l'errore
    '''
    dtheta0, dtheta1 = derivata(theta0, theta1, X, y)
    '''
    dopo aver calcolato il gradiente dobbiamo aggiornare i parametri 
    mediante la regola del gradient update (le due righe seguenti)
    '''
    theta0 = theta0 - (alpha * dtheta0)
    theta1 = theta1 - (alpha * dtheta1)
    '''
    nelle due righe precedenti viene citata 'alpha' che si chiama Learning Rate, 
    che è un piccolo numero che consente di aggiornare i parametri di una piccola quantita.
    Come detto sopra, stiamo cercando quindi di aggiornare il gradiente in modo tale 
    che diventi il più possibile vicino a zero.
    Il Learning Rate aiuta quindi 
    '''

    return theta0, theta1
'''
dobbiamo ripetere gli step- cercare gli errori, calcolare le derivate, aggiornare i parametri
finchè l'errore è il più piccolo possibile.
'''
    

def RegressioneLineare(X, y):
    '''
    theta0 e theta1 sono parametri variabili che ovviamente
    devono essere inizializzate (in maniera random inizialmente).
    Proprio per questo come si vede nella prima immagine, la linea
    è disegnata in maniera random
    '''
    theta0 = np.random.rand()
    theta1 = np.random.rand()
    
    for i in range(0, 1000):
        if i % 100 == 0:
            ShowLine(theta0, theta1, X, y)
        #print(cost(theta0, theta1, X, y))
        theta0, theta1 = AggiornaParametri(theta0, theta1, X, y, 0.005)



    


RegressioneLineare(X, y)