'''
    Arlind Stafaj
    CISC5800 - Machine Learning
    Dr. Leeds
    Final Project - Bayes Classification: Gaussian distribution 
'''

from featureSelection import FilterMethod, WrapperMethod
import numpy as np
import pandas as pd
import time

trainData = np.array(pd.read_csv('SPECTF.train'))
testData = np.array(pd.read_csv('SPECTF.test'))
np.seterr(divide='ignore')

start_time = time.time()


class BayesClassifier_Gaussian:

    def __init__(self, xTrain, yTrain):
        self.X = xTrain
        self.y = yTrain
        self.params = self.getParams()
        self.priors = self.getPriors()
    '''
        A function called getParams that takes in a data set and returns the learned mean
        and standard deviation for each class.
    '''

    def getParams(self):
        classInd = 0
        params = []
        numClasses = np.unique(self.y)
        while classInd < numClasses.size:
            x = self.X[classInd == self.y.reshape(self.X.shape[0])]
            mean = np.mean(x, axis=0)
            # if variance used instead of std there is lower percentage accuracy
            std = np.std(x, axis=0, ddof=1)
            params.append([mean, std])
            classInd += 1
        self.params = np.array(params)
        return np.array(params)

    """
        A function called getPriors that takes in a data
        set and returns the priorprobability of each class.
    """

    def getPriors(self):
        unique_elements, counts_elements = np.unique(
            self.y, return_counts=True)
        priors = np.asarray((unique_elements, counts_elements))
        result = []
        for i in priors[1]:
            result.append(i / self.y.size)
        self.priors = np.array(result)
        return np.array(result)

    '''
        A function called getLabels that takes in posting times for multiple users as well as
        the learned parameters for the likelihoods and prior, and return the most probably class for
        each user.
    '''

    def gaussPDF(self, x, mu, sig):
        return np.exp(-(x-mu)**2 / (2 * sig)) / np.sqrt(2 * np.pi * sig)

    def getLabels(self, xTest: np.array):
        labelsOut = np.zeros(0)
        # for each data point
        for dataInd in range(len(xTest)):
            classProbs = np.zeros(0)
            # for each class
            for classInd in range(len(self.priors)):
                currProb = self.gaussPDF(
                    xTest[dataInd], self.params[classInd, 0], self.params[classInd, 1])
                currProb = np.sum(np.log(currProb)) + \
                    np.log(self.priors[classInd])
                classProbs = np.append(classProbs, currProb)
            maxValue = np.max(classProbs)
            labelsOut = np.append(labelsOut, np.where(
                np.array(classProbs) == maxValue))

        return labelsOut

    '''
        A function called evaluateBayes that takes in classifier parameters for likelihoods
        and priors, and a set of labels and feature values, and returns the percent of input data
        correctly classified.
    '''

    def evaluateBayes(self, xTest, yTest):
        labels = self.getLabels(xTest)
        counter = 0
        for i in range(labels.size):
            if labels[i] == yTest[i]:
                counter += 1
        return (counter / labels.size) * 100


bayesGaussian = BayesClassifier_Gaussian(trainData[:, 1:], trainData[:, 0])
print("Test data performance accuracy: --> ",
      bayesGaussian.evaluateBayes(testData[:, 1:], testData[:, 0]))
print("Train data performance accuracy: --> ", bayesGaussian.evaluateBayes(
    trainData[:, 1:], trainData[:, 0]))


filterM = FilterMethod(trainData)
print("\n**********     FILTER METHOD     **********\n")

filterM.printFeaturePCC()
print("\n**********     Perfomance of filter method on train Data     *********")
filterM.computeFilterMethod(BayesClassifier_Gaussian, trainData)
print("**********     Perfomance of filter method on Test Data     *********")
filterM.testFilterMethod(BayesClassifier_Gaussian, testData)

wrapperM = WrapperMethod(trainData)
print("\n**********     WRAPPER METHOD     **********\n")
print("**********     Perfomance of filter method on train Data     *********")
wrapperM.computerWrapperMethod(BayesClassifier_Gaussian)
print("**********     Perfomance of filter method on Test Data     *********")
wrapperM.testWrapperMethod(BayesClassifier_Gaussian, testData)
print("--- %s seconds ---" % (time.time() - start_time))
