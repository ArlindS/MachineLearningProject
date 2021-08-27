import pandas as pd
import numpy as np


class FilterMethod:

    def __init__(self, trainData):
        self.trainData = trainData
        self.features = []
        for i in range(self.trainData.shape[1]-1):
            self.features.append('f' + str(i))

        # Compute r using the PCC method
        self.r = []
        for x in self.trainData[:, 1:].T:
            self.r.append(self.pcc(x, self.trainData[:, 0]))
        self.r = np.absolute(self.r)
        self.r_sorted = sorted(self.r, reverse=True)

        self.indexes = []  # store indexes of r sorted

        for i in range(len(self.features)):
            self.indexes.append(np.where(self.r == self.r_sorted[i])[0][0])

    def pcc(self, x, y):
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        cov_x_y = np.sum((x - mean_x) * (y - mean_y))
        stdv_x = np.sum(np.square(x - mean_x))
        stdv_y = np.sum(np.square(y - mean_y))
        correlation = cov_x_y / np.sqrt(stdv_x * stdv_y)

        return correlation

    def printFeaturePCC(self):
        print("\nFeatures Rated in terms of importance: \n")
        for i in range(len(self.features)):
            print(self.features[np.where(self.r == self.r_sorted[i])[0][0]],
                  ': ',  self.r_sorted[i])

    def computeFilterMethod(self, algorithm, testData):
        # xtrain = self.trainData[:, 1:]
        xtestD = testData[:, 1:]
        ytestD = testData[:, 0]
        predict = []

        # new_xtrain = np.array(xtrain[:, self.indexes[0]])
        xtest = np.array(xtestD[:, self.indexes[0]])
        bayes = algorithm(xtest, ytestD)
        predict.append(bayes.evaluateBayes(xtest, ytestD))

        for i in range(1, len(self.features)):

            # initialize with first column as ranked by r
            # new_xtrain = np.array(xtrain[:, self.indexes[0]])
            xtest = np.array(xtestD[:, self.indexes[0]])

            # add another column in order of indexes from r
            for n in range(1, i+1):
                # new_xtrain = np.column_stack((new_xtrain, xtrain[:, self.indexes[n]]))
                xtest = np.column_stack((xtest, xtestD[:, self.indexes[n]]))

            # print(xtest)
            bayes = algorithm(xtest, ytestD)
            acc = bayes.evaluateBayes(xtest, ytestD)
            predict.append(acc)

        print('\nAccuracies by adding features: \n', predict)

        print('\nHighest Accuracy is has ',
              predict.index(max(predict)) + 1, ' attributes. These are = ', max(predict))
        print('Features for highest accuracy: ',
              self.indexes[:predict.index(max(predict)) + 1], '\n')
        print('-----------------------------------------------------')

    def testFilterMethod(self, algorithm, testData):
        xtrain = self.trainData[:, 1:]
        ytrain = self.trainData[:, 0]
        xtestD = testData[:, 1:]
        ytestD = testData[:, 0]
        predict = []

        new_xtrain = np.array(xtrain[:, self.indexes[0]])
        xtest = np.array(xtestD[:, self.indexes[0]])
        bayes = algorithm(xtest, ytestD)
        predict.append(bayes.evaluateBayes(xtest, ytestD))

        for i in range(1, len(self.features)):

            # initialize with first column as ranked by r
            new_xtrain = np.array(xtrain[:, self.indexes[0]])
            xtest = np.array(xtestD[:, self.indexes[0]])

            # add another column in order of indexes from r
            for n in range(1, i+1):
                new_xtrain = np.column_stack(
                    (new_xtrain, xtrain[:, self.indexes[n]]))
                xtest = np.column_stack((xtest, xtestD[:, self.indexes[n]]))

            # print(xtest)
            bayes = algorithm(new_xtrain, ytrain)
            acc = bayes.evaluateBayes(xtest, ytestD)
            predict.append(acc)

        print('\nAccuracies by adding features: \n', predict)

        print('\nHighest Accuracy is has ',
              predict.index(max(predict)) + 1, ' attributes. These are = ', max(predict))
        print('Features for highest accuracy: ',
              self.indexes[:predict.index(max(predict)) + 1], '\n')
        print('-----------------------------------------------------')


class WrapperMethod:

    def __init__(self, trainData):
        self.trainData = trainData
        self.indexes = []

    def computerWrapperMethod(self, algorithm):
        ztrainx = self.trainData[:, 1:]
        # toremovetrain - used to keep track of column features left
        toremovetrain = self.trainData[:, 1:]
        # toaddtrain - used to calculate which features to add
        toaddtrain = np.array([])

        pred = []  # store max accuracy from each test
        indexes = []  # store index of column with max accuracy
        greedy = True
        print('\n-----------------------------------------------------')
        while toremovetrain.shape[1] > 0 and greedy:
            accuracies = []
            # test performance on adding feature to select best
            for i in range(toremovetrain.shape[1]):
                # when toaddtrain is of size zero can't use np.column_stack so initialize
                if (toaddtrain.size == 0):
                    toaddtrain = np.array(ztrainx[:, i])
                    toaddtrain = toaddtrain.reshape(-1, 1)
                else:
                    toaddtrain = np.column_stack(
                        (toaddtrain, toremovetrain[:, i]))

                bayes = algorithm(toaddtrain, self.trainData[:, 0])
                acc = bayes.evaluateBayes(toaddtrain, self.trainData[:, 0])
                accuracies.append(acc)

                # remove test data
                if toaddtrain.shape[1] == 1:
                    toaddtrain = np.array([])
                else:
                    toaddtrain = np.delete(
                        toaddtrain, toaddtrain.shape[1]-1, 1)

            # add best performing feature while there is a better one
            currentMax = max(accuracies) if accuracies else 0
            storedMax = max(pred) if pred else 0
            if currentMax >= storedMax:
                pred.append(max(accuracies))
                if toaddtrain.size == 0:
                    toaddtrain = toremovetrain[:, np.where(
                        accuracies == max(np.array(accuracies)))[0][0]]
                else:
                    toaddtrain = np.column_stack((toaddtrain, toremovetrain[:, np.where(
                        accuracies == max(np.array(accuracies)))[0][0]]))

                index = np.where(accuracies == max(np.array(accuracies)))[0][0]
                for i in range(ztrainx.T.shape[0]):
                    if np.array_equal(toremovetrain.T[index], ztrainx.T[i]):
                        indexes.append(i)
                print(indexes, '\n  percent accuracy:', max(accuracies))
                toremovetrain = np.delete(toremovetrain, index, 1)
            else:
                greedy = False

        print('-----------------------------------------------------')
        print('The wrapper method feature selection using Sequential Forward Selection: \n ',
              indexes[:pred.index(max(pred))+1])
        self.indexes = indexes[:pred.index(max(pred))+1]
        print('The maximum accuracy is: ', max(pred),
              'at ', pred.index(max(pred))+1, ' iterations')
        print('-----------------------------------------------------\n')
        return indexes[:pred.index(max(pred))+1]

    def testWrapperMethod(self, algorithm, testData):
        xtrain = self.trainData[:, 1:]
        xtest = testData[:, 1:]
        if len(self.indexes) > 0:
            new_xtest = xtest[:, 0]
            new_xtrain = xtrain[:, 0]
            for i in range(1, len(self.indexes)):
                new_xtest = np.column_stack(
                    (new_xtest, xtest[:, self.indexes[i]]))
                new_xtrain = np.column_stack(
                    (new_xtrain, xtrain[:, self.indexes[i]]))

            bayes2 = algorithm(new_xtrain, self.trainData[:, 0])
            print(bayes2.evaluateBayes(new_xtest, testData[:, 0]))

        else:
            print("\nRUN computeWrapperMethod(algorithm) first!\n")
