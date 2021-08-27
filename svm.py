from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd

trainData = np.array(pd.read_csv('SPECTF.train'))
testData = np.array(pd.read_csv('SPECTF.test'))
np.seterr(divide='ignore')

clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(trainData[:, 1:], trainData[:, 0])
print(clf.score(testData[:, 1:], testData[:, 0]))
