# MachineLearningProject

There are four python files: 
	1) BayesGaussian.py
	2) BayesUniform.py
	3) BayesRayleigh.py
	4) featureSelection.py
	5) svm.py

There is one text file explaining dataset:
	1) SPECTF.names.txt

There are two files containing train and test dataset:
	1) SPECTF.train
	2) SPECTF.test

featureSelection.py :
		This file contains the classes and code that runs feature selection.

	FilterMethod - 
	I implemented a filter method feature selection class
	- it is initialized with the full train dataset which is stored. 
	- printFeature() default function which prints out the number of features with an f in front so 	f1 = feature 1, f2 = feature 2 … fn = feature n. 
	- computeFilterMethod(algorithm, testData) is a function that takes the algorithm class as a parameter 	and the data you are testing performance of filter selection with. It prints out accuracies as features 	are added based on importance of feature in determining class. It also prints out total features used to 	compute best result 
	- testFilterMethod(algorithm, testData) is a function that takes the algorithm and data you want to test	with the stored train data from initialization. 
	
	WrapperMethod -
	The wrapper method class was implemented as an improvement to the filter method class.
	- It is initialized with the full train data which is stored. 
	- computeWrapperMethod(algorithm) is a class function that computes the best performing features 	based on wrapper method algorithm and prints results as they are tested. It then returns the best 	features 
	- testWrapperMethod(algorithm, testData) is a function that takes the algorithm and data you want to 	test with the stored train data from initialization. 

BayesGaussian.py
		This file contains class code and the main code computed for Bayes classifier with a gaussian probability distribution function. 
	
	BayesClassifier_Gaussian -
	This is the main class that implements the Bayes classifier algorithm with a gaussian PDF
	- It is initialized with the x train data and y train data 
	- getParams() is a function that returns the parameters learned from train dataset (mean and standard deviation)
	- getPriors() if a function that returns the priors of the train dataset from each class 
	- gaussPDF() is the function that computes the gaussian pdf on each x_i 
	- getLabels(xTest) is a function that returns the labels predicted for xTest on a based on trainData stored.  
	- evaluateBayes(xTest, yTest) is a function that with xTest and yTest input gives you the accuracy trainData predicts them correct. 

BayesUniform.py	
	Same layout as BayesGaussian class with differences only in computation algorithm 

BayesRayleigh.py
	Same layout as BayesGaussian class with differences only in computation algorithm 

svm.py 
	Uses sklearn library to run data on svm algorithm
