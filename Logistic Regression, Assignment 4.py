import numpy as np
import pandas as pd
data=pd.read_csv("HW4data.csv")

def getName():
    #TODO: Add your full name instead of Lionel Messi
    return "Batuhan Demirci"

def getStudentID():
    #TODO: Replace X's with your student ID. It should stay as a string and should have exactly 9 digits in it.
    return "070190155"

def standardize(X):
    return (X - X.mean())/X.std(), X.mean(), X.std()

#Define your functions here if necessary
def predict(X,beta0,beta1):
    output = 1 / (1 + np.exp(-(beta0 + np.dot(beta1, X))))
    return output

def gradient_descent(data,num_iter,alpha,random_seed):
    X=np.array(data['X'])
    y=np.array(data['y'])
    X,muX,sdX = standardize(X)
    #Do not standardize y!!!!!
    np.random.seed(random_seed)
    #beta values are initialized here. Don't reinitialize beta values again!!
    beta0 = np.random.rand()
    beta1 = np.random.rand()
    J_list = []

    #write your own code here

    num_var = X.shape[0]

    for i in range(num_iter):

        j = (-1 / num_var) * np.sum((y * np.log(predict(X, beta0, beta1))) + (1 - y) *
                                    np.log(1 - predict(X, beta0, beta1)))
        J_list.append(j)

        gradient_beta1 = 1/num_var * np.sum(np.dot(X, (predict(X, beta0, beta1)-y)))
        gradient_beta0 = 1/num_var * np.sum((predict(X, beta0, beta1) - y))

        beta1 -= alpha * gradient_beta1
        beta0 -= alpha * gradient_beta0

        J_list.append(j)

    return J_list, beta0, beta1
