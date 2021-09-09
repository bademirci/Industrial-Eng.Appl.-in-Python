import numpy as np
import pandas as pd


def getName():
    # TODO: Add your full name instead of Lionel Messi
    return "Batuhan Demirci"


def getStudentID():
    # TODO: Replace X's with your student ID. It should stay as a string and should have exactly 9 digits in it.
    return "070190155"


# You can define your own other necessary functions here
def p_y(y):
    p_y = []
    for i in np.sort(y.unique()):
        pyi = len(y[y == i]) / len(y)
        p_y.append(pyi)
    return p_y


def mean_std_corr_for_each_class(x, y):
    means = []
    for i in np.sort(y.unique()):
        means_i = x[y == i].mean()
        means.append(means_i)
    std1 = x["x1"].std()
    std2 = x["x2"].std()
    corr = x.corr().iloc[1, 0]
    return means, std1, std2, corr


def calculateGaussianProbability(x1, x2, std1, std2, muy1, muy2, corr):
    expo = np.exp(-(np.power(std2, 2) * np.power(x1 - muy1, 2) + np.power(std1, 2) * np.power(x2 - muy2, 2) - 2 * corr * std1 * std2 * (
                x1 - muy1) * (
                x2 - muy2)) / (2 * (1 - np.power(corr,2)) * np.power(std1, 2) * np.power(std2, 2)))
    return 1 / (2 * np.pi * std1 * std2 * (np.sqrt(1 - np.power(corr, 2)))) * expo


def calculateClassProbabilities(x, x1, x2, means1, means2, std1, std2, class_probs, corr):
    probabilities = []
    for i in range(len(class_probs)):
        numerator = 1

        mean_ij_for_x1 = means1[i][0]
        mean_ij_for_x2 = means2[i][1]
        fxji = calculateGaussianProbability(x1, x2, std1, std2, mean_ij_for_x1, mean_ij_for_x2, corr=corr)
        numerator = numerator * fxji *class_probs[i]
        probabilities.append(numerator)
    return probabilities / np.sum(probabilities, axis=0)


def predict(x, x1, x2, means1, means2, std1, std2, class_probs, corr):
    probabilities = calculateClassProbabilities(x, x1, x2, means1, means2, std1, std2,
                                                class_probs, corr)
    Class = (probabilities[1] >= 0.5).astype("int")
    return Class


def compute_accuracy(prediction, test_y):
    return (1 - np.sum(np.abs(prediction - test_y)) / len(prediction)) * 100





def not_so_naive_bayes(train, test):
    x = train.drop("y", axis=1)
    y = train["y"]
    x1 = test["x1"]
    x2 = test["x2"]
    means, std1, std2, corr = mean_std_corr_for_each_class(x, y)
    class_probs = p_y(y)

    prediction = predict(x, x1, x2, means, means, std1, std2, class_probs, corr)
    list(zip(prediction, test["y"]))
    accuracy = compute_accuracy(prediction, test["y"])

    return accuracy, prediction
