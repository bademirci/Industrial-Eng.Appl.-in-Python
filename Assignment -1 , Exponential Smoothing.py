# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:12:02 2021

@author: ergun
"""
import numpy as np 

def getName():
    #TODO: Add your full name instead of Lionel Messi

    return "Batuhan Demirci"

def getStudentID():
    #TODO: Replace X's with your student ID. It should stay as a string and should have exactly 9 digits in it.

    return "070190155"


def MAPE(y , y_pred):
    return round(100*np.nanmean(np.abs(y - y_pred)/np.abs(y)),2)

def exponential_smoothing(x, alpha, l_zero , mape=False):
    #TODO: Implement your function here

    x_forecast = np.full_like(x, np.nan)
    x_forecast[1] = (alpha * x[0] + (1 - alpha) * l_zero)

    for t in range(2, len(x)):
        x_forecast[t] = (alpha * x[t - 1] + (1 - alpha) * x_forecast[t - 1])
    if mape is True:

        return (x_forecast,MAPE(x,x_forecast)) if 0 < alpha < 1  else "Invalid Alpha"


    return x_forecast if 0 < alpha < 1  else "Invalid Alpha"



