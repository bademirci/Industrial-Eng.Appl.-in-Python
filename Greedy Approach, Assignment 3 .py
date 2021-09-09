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

def generate_sol(current_sol,num_cities):
    current_sol_copy = current_sol.copy()
    idx1 ,idx2 = np.random.choice(num_cities, 2)
    current_sol_copy[idx2] , current_sol_copy[idx1] = current_sol_copy[idx1] , current_sol_copy[idx2] 
    return current_sol_copy

def calculate_objective(coord, solution):
    a1 = coord[solution,:]
    altered_sol = np.append(solution[1:] , solution[0])
    a2 = coord[altered_sol,:]
    return np.sum(np.sqrt(np.sum((a1-a2)**2, axis=1)))

def tsp_greedy_multiple(coords, num_iter, num_rep, random_seed, best_obj=False):
    np.random.seed(random_seed) # please do not change this line and do not assign a seed again.
    #TODO: Implement your function here
    for x in range(num_rep):
        x_old = np.random.permutation(range(len(coords)))
        f_old = calculate_objective(coords, x_old)
        for i in range(num_iter):
            x_new = generate_sol(x_old, len(coords))
            f_new = calculate_objective(coords, x_new)

            if f_old > f_new:
                x_old = x_new
                f_old = f_new
    if best_obj==True:
        return x_old,f_old
    else:
        return x_old
